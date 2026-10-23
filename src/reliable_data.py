"""Reliable score: snapshot khả dụng, prediction set, nhãn suy yếu kinh doanh, cửa sổ trượt train/test.

Định nghĩa giữ nguyên POC (research/2026-10-10_reliable_score, reliable_score_poc/README.md):
- Prediction set tại cohort t: có đơn giao thành công ở cả 3 snapshot t-2..t, trung bình >= min_orders
  đơn/tháng, thâm niên >= min_tenure tháng.
- Nhãn bad tại t: baseline = tổng don_ptc_l1m 3 snapshot t-2..t; tương lai = tổng H snapshot t+1..t+H;
  hệ số thị trường m = Σ tương lai / Σ baseline trên tập khó S0; bad = 1 nếu tương lai = 0 hoặc
  (tương lai / baseline) / m < threshold.
- Nhãn tạm (partial): chỉ có k < H snapshot tương lai. Tương lai và baseline cùng quy về trung bình
  tháng, m tính trên đúng k tháng đó. Dùng để xem trước performance tháng test mới nhất.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

from src.data import minio_fs, norm_cus_id


def shift(snap: int, k: int) -> int:
    return int((pd.Timestamp(str(snap)) + pd.DateOffset(months=k)).strftime("%Y%m%d"))


def partitions(fs, bucket: str, source: str) -> set[int]:
    infos = fs.get_file_info(pafs.FileSelector(f"{bucket}/{source}", allow_not_found=True))
    return {int(i.base_name.split("=")[1]) for i in infos if i.type == pafs.FileType.Directory and i.base_name.startswith("date=")}


def available_snapshots(cfg: dict) -> tuple[list[int], list[int]]:
    """(snapshot có đủ mọi nguồn feature, snapshot có sales/l1m dùng cho nhãn)."""
    fs, bucket = minio_fs(cfg)
    parts = {s: partitions(fs, bucket, s) for s in cfg["feature_sources"]}
    order_src = cfg["reliable"]["orders_source"]
    full = sorted(set.intersection(*parts.values()))
    orders = sorted(parts.get(order_src) or partitions(fs, bucket, order_src))
    return full, orders


def monthly_orders(cfg: dict, snaps: list[int]) -> pd.DataFrame:
    """Bảng cus_id × snapshot số đơn giao thành công; shop không có dòng = 0 đơn qua VTP."""
    fs, bucket = minio_fs(cfg)
    src, col = cfg["reliable"]["orders_source"], cfg["reliable"]["orders_column"]
    dset = ds.dataset(f"{bucket}/{src}", filesystem=fs, format="parquet", partitioning="hive")
    t = dset.to_table(columns=["cus_id", col, "date"], filter=ds.field("date").isin(snaps)).to_pandas()
    t["cus_id"] = norm_cus_id(t["cus_id"])
    o = t.pivot_table(index="cus_id", columns="date", values=col, aggfunc="sum")
    return o.reindex(columns=snaps).fillna(0).clip(lower=0).astype("float32")


def tenure(cfg: dict, snap: int) -> pd.Series:
    """Thâm niên (tháng); cụm nghi ngày hợp tác mặc định (~59,26 tháng tại 20260801) coi như thiếu."""
    fs, bucket = minio_fs(cfg)
    dset = ds.dataset(f"{bucket}/features/demographic/final", filesystem=fs, format="parquet", partitioning="hive")
    t = dset.to_table(columns=["cus_id", "duration"], filter=ds.field("date") == snap).to_pandas()
    d = t.assign(cus_id=norm_cus_id(t["cus_id"])).drop_duplicates("cus_id").set_index("cus_id")["duration"].astype("float32")
    target = 59.26 + (pd.Timestamp(str(snap)) - pd.Timestamp("20260801")).days / 30.44
    return d.mask((d - target).abs() < 0.12)


def build_cohort(cfg: dict, snap: int, orders: pd.DataFrame, horizon_avail: int) -> pd.DataFrame:
    """Prediction set + nhãn của một cohort. horizon_avail = số snapshot tương lai đang có (<= H)."""
    rc = cfg["reliable"]
    H = rc["horizon"]
    base_snaps = [shift(snap, -k) for k in (2, 1, 0)]
    fut_snaps = [shift(snap, k) for k in range(1, min(H, horizon_avail) + 1)]
    o = orders[base_snaps]
    o = o[(o > 0).all(axis=1)]
    df = pd.DataFrame({"cus_id": o.index, "date": snap, "orders_mean_3m": o.mean(axis=1).to_numpy(),
                       "baseline": o.sum(axis=1).to_numpy()})
    df["tenure"] = tenure(cfg, snap).reindex(df["cus_id"]).to_numpy()
    df = df[(df["orders_mean_3m"] >= rc["min_orders"]) & (df["tenure"] >= rc["min_tenure"])].reset_index(drop=True)
    s0 = (df["orders_mean_3m"] >= rc["s0_min_orders"]) & (df["tenure"] >= rc["s0_min_tenure"])
    df["in_s0"] = s0.to_numpy()

    k = len(fut_snaps)
    df["label_months"] = k
    if k == 0:
        df["future"], df["label"], df["label_status"] = np.nan, np.nan, "none"
        return df
    df["future"] = orders.reindex(df["cus_id"])[fut_snaps].sum(axis=1).to_numpy()
    fut_m, base_m = df["future"] / k, df["baseline"] / 3          # quy về trung bình tháng
    m = fut_m[s0].sum() / base_m[s0].sum()
    ratio = (fut_m / base_m) / m
    df["market_factor"] = m
    df["label"] = ((df["future"] == 0) | (ratio < rc["bad_threshold"])).astype(int)
    df["label_status"] = "full" if k == H else f"partial_{k}m"
    return df


def make_windows(cohorts: dict[int, str], train_months: int, max_partial_tests: int) -> list[dict]:
    """Cửa sổ trượt: train X cohort liên tiếp có nhãn đủ, test cohort ngay sau (đủ nhãn, hoặc nhãn tạm
    cho tối đa `max_partial_tests` cửa sổ cuối)."""
    snaps = sorted(cohorts)
    windows, partial_used = [], 0
    for i in range(train_months, len(snaps)):
        train = snaps[i - train_months:i]
        test = snaps[i]
        consecutive = all(shift(train[0], j) == s for j, s in enumerate(train + [test]))
        if not consecutive or any(cohorts[s] != "full" for s in train):
            continue
        status = cohorts[test]
        if status == "none":
            continue
        if status != "full":
            if partial_used >= max_partial_tests:
                continue
            partial_used += 1
        windows.append({"train": train, "test": test, "test_label": status})
    return windows
