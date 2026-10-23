"""Chấm điểm leadgen cho một snapshot bằng model đã train.

    # Toàn bộ whitelist (leadgen/leads, rule_tnex_upl = True) của snapshot
    python predict.py --date 20260801

    # Hoặc một danh sách cus_id (file CSV có cột cus_id)
    python predict.py --date 20260801 --cus-file my_list.csv

Đầu ra: outputs/predictions/leadgen_<date>.parquet (+ .csv) với
cus_id, date, proba, score (0–100, xếp hạng trong lô), bin (1 = 10% điểm cao nhất theo mốc của tập test).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from catboost import CatBoostClassifier

from src.data import load_config, load_features, minio_fs, norm_cus_id, prepare_matrix


def whitelist_ids(cfg: dict, date: int) -> pd.DataFrame:
    fs, bucket = minio_fs(cfg)
    dset = ds.dataset(f"{bucket}/leadgen/leads", filesystem=fs, format="parquet", partitioning="hive")
    t = dset.to_table(columns=["cus_id", "rule_tnex_upl"], filter=ds.field("date") == date).to_pandas()
    t = t[t["rule_tnex_upl"].astype(bool)]
    return pd.DataFrame({"cus_id": norm_cus_id(t["cus_id"]).unique()})


def main(cfg_path: str, date: int, cus_file: str | None, model_dir: str | None):
    cfg = load_config(cfg_path)
    mdir = Path(model_dir or cfg["paths"]["model_dir"])
    meta = json.loads((mdir / "model_meta.json").read_text())
    model = CatBoostClassifier()
    model.load_model(str(mdir / "model.cbm"))

    ids = pd.read_csv(cus_file, dtype={"cus_id": str})[["cus_id"]] if cus_file else whitelist_ids(cfg, date)
    keys = pd.DataFrame({"cus_id": norm_cus_id(ids["cus_id"]).unique(), "date": date})
    # Đọc đúng danh sách nguồn lúc train để tên cột trùng (hậu tố __group) giống hệt
    cfg = {**cfg, "feature_sources": meta["feature_sources"], "exclude_columns": meta["exclude_columns"]}
    df, _ = load_features(cfg, keys, use_cache=False)
    missing = [f for f in meta["features"] if f not in df.columns]
    if missing:
        raise ValueError(f"Thiếu feature so với lúc train: {missing}")
    X = prepare_matrix(df, meta["features"], meta["cat_features"])

    proba = model.predict_proba(X)[:, 1]
    cut = np.array(meta["decile_cutpoints_test"])
    out = pd.DataFrame({
        "cus_id": df["cus_id"], "date": date, "proba": proba,
        "score": (pd.Series(proba).rank(pct=True) * 100).round(2).to_numpy(),
        "bin": len(cut) + 1 - np.searchsorted(cut, proba, side="right"),
    }).sort_values("proba", ascending=False)

    od = Path("outputs/predictions"); od.mkdir(parents=True, exist_ok=True)
    out.to_parquet(od / f"leadgen_{date}.parquet", index=False)
    out.to_csv(od / f"leadgen_{date}.csv", index=False)
    print(f"[predict] {len(out):,} shop, model {meta['trained_at']}, test AUC {meta['test_auc']:.4f} -> {od}/leadgen_{date}.*")
    print(out["bin"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/leadgen.yaml")
    ap.add_argument("--date", type=int, required=True, help="snapshot YYYYMM01, ví dụ 20260801")
    ap.add_argument("--cus-file", default=None)
    ap.add_argument("--model-dir", default=None)
    a = ap.parse_args()
    main(a.config, a.date, a.cus_file, a.model_dir)
