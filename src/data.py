"""Đọc nhãn từ Excel và feature từ feature store trên MinIO (S3)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import yaml

KEYS = ["cus_id", "date"]


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def norm_cus_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def load_labels(cfg: dict) -> pd.DataFrame:
    """Một dòng mỗi (cus_id, date); label = 1 nếu shop nộp hồ sơ trong tháng snapshot."""
    lc = cfg["label"]
    df = pd.read_excel(lc["file"], sheet_name=lc["sheet"],
                       usecols=[lc["key"], lc["snapshot_col"], lc["positive_col"]])
    df = df.dropna(subset=[lc["key"]])
    out = pd.DataFrame({
        "cus_id": norm_cus_id(df[lc["key"]]),
        "date": pd.to_datetime(df[lc["snapshot_col"]]).dt.strftime("%Y%m01").astype(int),
        "label": df[lc["positive_col"]].notna().astype(int),
    })
    out = out[out["date"].isin(lc["snapshots"])]
    out = out.groupby(KEYS, as_index=False)["label"].max()
    return out


def minio_fs(cfg: dict) -> tuple[pafs.S3FileSystem, str]:
    mc = cfg["minio"]
    conn = load_config(mc["connect_yaml"])["minio"]
    bucket = mc.get("bucket") or conn.get("bucket")
    # Endpoint của host: bỏ proxy cho 127.0.0.1
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    fs = pafs.S3FileSystem(endpoint_override=mc["endpoint"], access_key=conn["access_key"],
                           secret_key=conn["secret_key"], scheme="http")
    return fs, bucket


def group_name(source: str) -> str:
    return source.removeprefix("features/")


def _read_source(fs, bucket: str, source: str, dates: list[int], ids: list[str]) -> pd.DataFrame:
    """Lọc theo date và cus_id ngay khi quét (cus_id lưu dạng chuỗi, có nơi kèm '.0')."""
    dset = ds.dataset(f"{bucket}/{source}", filesystem=fs, format="parquet", partitioning="hive")
    id_arr = pa.array(ids + [i + ".0" for i in ids], pa.string())
    flt = ds.field("date").isin(pa.array(dates, pa.int64())) & ds.field("cus_id").isin(id_arr)
    table = dset.to_table(filter=flt)
    df = table.to_pandas()
    df["cus_id"] = norm_cus_id(df["cus_id"])
    df["date"] = df["date"].astype(int)
    return df


def load_features(cfg: dict, keys: pd.DataFrame, use_cache: bool = True
                  ) -> tuple[pd.DataFrame, dict[str, str]]:
    """Merge feature của mọi group vào `keys` (cus_id, date).

    Trả về (df, feature_group): cột trùng tên giữa các group được thêm hậu tố __<group>.
    """
    dates = sorted(keys["date"].unique().tolist())
    sig = hashlib.md5(json.dumps([cfg["feature_sources"], cfg["exclude_columns"], dates,
                                  len(keys), int(keys["cus_id"].str.len().sum())]).encode()).hexdigest()[:10]
    cache_dir = Path(cfg["paths"]["cache_dir"])
    cache = cache_dir / f"features_{sig}.parquet"
    meta = cache_dir / f"features_{sig}.json"
    if use_cache and cache.exists() and meta.exists():
        return pd.read_parquet(cache), json.loads(meta.read_text())

    fs, bucket = minio_fs(cfg)
    wanted = set(map(tuple, keys[KEYS].to_numpy()))
    ids = sorted(keys["cus_id"].unique().tolist())
    out = keys[KEYS].copy()
    feature_group: dict[str, str] = {}
    exclude = set(cfg["exclude_columns"]) | set(KEYS)
    for source in cfg["feature_sources"]:
        grp = group_name(source)
        df = _read_source(fs, bucket, source, dates, ids)
        df = df[[c for c in df.columns if c not in exclude or c in KEYS]]
        df = df[pd.MultiIndex.from_frame(df[KEYS]).isin(wanted)]
        df = df.drop_duplicates(KEYS)
        ren = {}
        for c in df.columns:
            if c in KEYS:
                continue
            ren[c] = f"{c}__{grp.replace('/', '_')}" if (c in feature_group or c in out.columns) else c
        df = df.rename(columns=ren)
        for c in ren.values():
            feature_group[c] = grp
        out = out.merge(df, on=KEYS, how="left", validate="one_to_one")
        print(f"[data] {grp:24s} cols={len(ren):3d} matched={len(df):,}", flush=True)

    # Cột ngày/thời gian không dùng làm feature
    for c in list(feature_group):
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out = out.drop(columns=c)
            feature_group.pop(c)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    meta.write_text(json.dumps(feature_group, ensure_ascii=False, indent=1))
    return out, feature_group


def prepare_matrix(df: pd.DataFrame, features: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """Ma trận cho CatBoost: biến categorical thành chuỗi, số giữ NaN, inf -> NaN."""
    X = df[features].copy()
    for c in features:
        if c in cat_cols or X[c].dtype == object:
            X[c] = X[c].astype(str).fillna("NA")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").replace([float("inf"), float("-inf")], float("nan"))
    return X
