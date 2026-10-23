"""Train leadgen model một lần, lưu model và toàn bộ kết quả cho báo cáo.

    python train.py --config configs/leadgen.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src import evaluate as ev
from src.data import load_config, load_features, load_labels, prepare_matrix
from src.features import cat_features_of, select_features
from src.tuning import tune


def optuna_plot(trials: pd.DataFrame, out: Path):
    t = trials.dropna(subset=["value"]).sort_values("number")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.scatter(t["number"], t["value"], color=ev.C_TRAIN, s=18, label="AUC CV từng trial")
    ax.plot(t["number"], t["value"].cummax(), color=ev.INK, lw=2, label="Tốt nhất tới trial")
    ax.set_xlabel("Trial")
    ax.set_ylabel("AUC CV (train)")
    ax.set_title("Optuna — lịch sử tối ưu")
    ax.legend(frameon=False, loc="lower right")
    ev._save(fig, out / "optuna_history.png")


def main(cfg_path: str):
    t0 = time.time()
    cfg = load_config(cfg_path)
    out = Path(cfg["paths"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    mdir = Path(cfg["paths"]["model_dir"]); mdir.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]

    # 1. Dữ liệu
    labels = load_labels(cfg)
    df, feature_group = load_features(cfg, labels)
    df = df.merge(labels, on=["cus_id", "date"], how="left", validate="one_to_one")
    df["split"] = np.where(df["date"].isin(cfg["split"]["train"]), "train",
                           np.where(df["date"].isin(cfg["split"]["test"]), "test", "unused"))
    df = df[df["split"] != "unused"].reset_index(drop=True)
    features_all = list(feature_group)
    X = prepare_matrix(df, features_all, cfg["categorical_columns"])
    cats_all = cat_features_of(X, features_all)
    tr = (df["split"] == "train").to_numpy()
    te = ~tr
    y = df["label"]
    print(f"[train] rows train={tr.sum():,} test={te.sum():,} features={len(features_all)}", flush=True)
    ev.label_rate(df, out)

    # 2. Chọn biến (chỉ trên train)
    Xtr, ytr, gtr = X[tr].reset_index(drop=True), y[tr].reset_index(drop=True), df.loc[tr, "cus_id"].reset_index(drop=True)
    sel = select_features(Xtr, ytr, gtr, features_all, feature_group, cfg)
    features = sel["final"]
    cats = [c for c in features if c in cats_all]
    sel["stat_log"].to_csv(out / "selection_1_stat_filter.csv", index=False)
    sel["group_log"].to_csv(out / "selection_2_by_group.csv", index=False)
    sel["combined_log"].to_csv(out / "selection_3_combined.csv", index=False)

    # 3. Optuna
    tun = tune(Xtr, ytr, gtr, features, cfg)
    tun["trials"].to_csv(out / "optuna_trials.csv", index=False)
    optuna_plot(tun["trials"], out)
    params = tun["params"]
    (out / "best_params.json").write_text(json.dumps({**params, "cv_auc": tun["cv_auc"]}, indent=2))

    # 4. Model cuối trên toàn bộ train
    model = CatBoostClassifier(**params, eval_metric="AUC", random_seed=seed, verbose=False,
                               thread_count=cfg["thread_count"], allow_writing_files=False)
    model.fit(Pool(Xtr[features], ytr, cat_features=cats))
    model.save_model(str(mdir / "model.cbm"))

    # 5. Đánh giá: train dùng OOF CV, test dùng model cuối
    df["score"] = np.nan
    df.loc[tr, "score"] = tun["oof"]
    df.loc[te, "score"] = model.predict_proba(X.loc[te, features])[:, 1]
    df["score_insample"] = model.predict_proba(X[features])[:, 1]
    aucs = ev.auc_by_month(df[["date", "split", "label", "score"]], out)
    test_df = df[te][["date", "label", "score"]].reset_index(drop=True)
    prec = ev.precision_tables(test_df, cfg["eval"]["n_bins"], out)

    psi_train = X[tr][features].assign(score=df.loc[tr, "score_insample"].to_numpy())
    psi_test = X[te][features].assign(score=df.loc[te, "score_insample"].to_numpy(), date=df.loc[te, "date"].to_numpy())
    psi_s, psi_f = ev.psi_report(psi_train, psi_test, features, cats, cfg["eval"]["psi_bins"], out)

    fi, gi = ev.importance_report(model, features, feature_group, out)
    shap_t = ev.shap_report(model, X[te].reset_index(drop=True), features, cats, fi,
                            cfg["eval"]["shap_top"], cfg["eval"]["shap_sample"], seed, out)

    # 6. Lưu meta để predict dùng lại
    cut = np.quantile(df.loc[te, "score"], np.linspace(0, 1, cfg["eval"]["n_bins"] + 1))[1:-1].tolist()
    meta = {
        "run_name": cfg["run_name"], "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "features": features, "cat_features": cats,
        "feature_group": {f: feature_group[f] for f in features},
        "feature_sources": cfg["feature_sources"], "exclude_columns": cfg["exclude_columns"],
        "categorical_columns": cfg["categorical_columns"],
        "train_snapshots": cfg["split"]["train"], "test_snapshots": cfg["split"]["test"],
        "params": params, "cv_auc": tun["cv_auc"],
        "test_auc": float(aucs.loc[aucs["date"] == "all_test", "auc"].iloc[0]),
        "decile_cutpoints_test": cut,
    }
    (mdir / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    # 7. Gom bảng vào một file Excel + tóm tắt
    with pd.ExcelWriter(out / "report_tables.xlsx") as w:
        pd.read_csv(out / "label_rate_by_month.csv").to_excel(w, sheet_name="label_rate", index=False)
        aucs.to_excel(w, sheet_name="auc_gini_by_month", index=False)
        prec.to_excel(w, sheet_name="precision_decile_test", index=False)
        psi_s.to_excel(w, sheet_name="psi_score", index=False)
        psi_f.to_excel(w, sheet_name="psi_features", index=False)
        fi.to_excel(w, sheet_name="importance_feature", index=False)
        gi.to_excel(w, sheet_name="importance_group", index=False)
        shap_t.to_excel(w, sheet_name="shap_top10", index=False)
        sel["stat_log"].to_excel(w, sheet_name="sel_1_stat", index=False)
        sel["group_log"].to_excel(w, sheet_name="sel_2_group", index=False)
        sel["combined_log"].to_excel(w, sheet_name="sel_3_combined", index=False)
        tun["trials"].to_excel(w, sheet_name="optuna_trials", index=False)

    n_stat = int((sel["stat_log"]["drop_reason"] == "").sum())
    n_grp = int(sel["group_log"]["selected"].sum())
    top_bin = prec[(prec["date"] == "all_test") & (prec["bin"] == 1)].iloc[0]
    lines = [
        f"# Leadgen model — {meta['trained_at']}", "",
        f"- Nhãn: shop trong WL nộp hồ sơ trong tháng snapshot. Train {cfg['split']['train']}, test {cfg['split']['test']} (out-of-time).",
        f"- Chọn biến: {len(features_all)} → lọc thống kê {n_stat} → theo group {n_grp} → gộp, cộng dồn {cfg['selection']['cum_importance']:.0%} importance: **{len(features)} biến**.",
        f"- Optuna {len(tun['trials'])} trial, AUC CV train tốt nhất {tun['cv_auc']:.4f}. Tham số: `{json.dumps(params)}`",
        "", "| Tháng | Tập | n | Nộp hồ sơ | AUC | Gini |", "|---|---|---|---|---|---|",
    ]
    for r in aucs.itertuples():
        lines.append(f"| {r.date} | {r.split} | {r.n:,} | {r.n_pos:,} | {r.auc:.4f} | {r.gini:.4f} |")
    lines += ["", f"- Bin 1 (10% điểm cao nhất) trên test: precision {top_bin.precision:.2%}, lift ×{top_bin.lift:.2f}, bắt {top_bin.cum_recall:.1%} số shop nộp hồ sơ.",
              f"- PSI score: " + ", ".join(f"{r.date} {r.psi_score:.4f}" for r in psi_s.itertuples()),
              f"- Thời gian chạy: {(time.time() - t0) / 60:.1f} phút."]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/leadgen.yaml")
    main(ap.parse_args().config)
