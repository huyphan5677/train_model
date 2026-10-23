"""Chấm reliable score cho một snapshot bằng model cuối.

    python predict_reliable.py                 # snapshot mới nhất có đủ nguồn feature
    python predict_reliable.py --date 20260801

Đầu ra outputs/predictions/reliable_<date>.parquet (+ .csv): cus_id, date, risk (xác suất bad của model,
chỉ dùng nội bộ), score 0-100 (xếp hạng trong tháng, 100 = ít rủi ro nhất), band A-E (5 khúc 20 điểm).
Chỉ chấm shop thuộc prediction set như lúc train.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.data import KEYS, load_config, load_features, prepare_matrix
from src.reliable_data import available_snapshots, build_cohort, monthly_orders, shift


def main(cfg_path: str, date: int | None, model_dir: str | None):
    cfg = load_config(cfg_path)
    mdir = Path(model_dir or cfg["paths"]["model_dir"])
    meta = json.loads((mdir / "model_meta.json").read_text())
    model = CatBoostClassifier()
    model.load_model(str(mdir / "model.cbm"))
    cfg = {**cfg, "reliable": meta["reliable"], "feature_sources": meta["feature_sources"],
           "exclude_columns": meta["exclude_columns"]}

    feat_snaps, order_snaps = available_snapshots(cfg)
    date = date or max(feat_snaps)
    if date not in feat_snaps:
        raise ValueError(f"Snapshot {date} chưa đủ mọi nguồn feature")
    orders = monthly_orders(cfg, [shift(date, -2), shift(date, -1), date])
    ps = build_cohort(cfg, date, orders, horizon_avail=0)          # chỉ prediction set, không nhãn
    df, _ = load_features(cfg, ps[KEYS], use_cache=False)
    X = prepare_matrix(df, meta["features"], meta["cat_features"])
    risk = model.predict_proba(X)[:, 1]

    score = 100 * (1 - pd.Series(risk).rank(pct=True).to_numpy()) + 50 / len(risk)
    band = pd.cut(score, [-1, 20, 40, 60, 80, 101], labels=["E", "D", "C", "B", "A"])
    out = pd.DataFrame({"cus_id": df["cus_id"], "date": date, "risk": risk, "score": score.round(2), "band": band})
    od = Path("outputs/predictions"); od.mkdir(parents=True, exist_ok=True)
    out.to_parquet(od / f"reliable_{date}.parquet", index=False)
    out.to_csv(od / f"reliable_{date}.csv", index=False)
    print(f"[predict] {date}: {len(out):,} shop trong prediction set, model {meta['trained_at']} "
          f"(train {meta['train_cohorts']}) -> {od}/reliable_{date}.*")
    print(out["band"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reliable.yaml")
    ap.add_argument("--date", type=int, default=None)
    ap.add_argument("--model-dir", default=None)
    a = ap.parse_args()
    main(a.config, a.date, a.model_dir)
