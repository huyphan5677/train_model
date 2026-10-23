"""Reliable score: kiểm định cửa sổ trượt + model cuối.

Mỗi cửa sổ: train X cohort liên tiếp (nhãn đủ) -> test cohort ngay sau. Trượt 1 tháng, lặp tới
cohort mới nhất; cửa sổ cuối được phép có nhãn tạm (chưa đủ H tháng tương lai) để xem trước.
Trong mỗi cửa sổ: chọn biến lại từ đầu (lọc thống kê -> theo group -> gộp), train CatBoost với
tham số Optuna (tune một lần trên train của cửa sổ đầu tiên), đánh giá tháng test.
Model cuối = train trên X cohort nhãn đủ mới nhất, lưu để chấm tháng mới.

    python train_reliable.py --config configs/reliable.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src import evaluate as ev
from src import reliable_report as rr
from src.data import KEYS, load_config, load_features, prepare_matrix
from src.features import cat_features_of, select_features
from src.reliable_data import available_snapshots, build_cohort, make_windows, monthly_orders, shift
from src.tuning import tune


def sample_by_shop(df: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    """Chỉ số dòng của mẫu ~n dòng, lấy trọn mọi cohort của các shop được chọn."""
    if len(df) <= n:
        return np.arange(len(df))
    shops = df["cus_id"].drop_duplicates().sample(frac=n / len(df), random_state=seed)
    return np.flatnonzero(df["cus_id"].isin(set(shops)).to_numpy())


def fit(X, y, features, cats, params, cfg) -> CatBoostClassifier:
    m = CatBoostClassifier(**params, eval_metric="AUC", random_seed=cfg["seed"], verbose=False,
                           thread_count=cfg["thread_count"], allow_writing_files=False)
    m.fit(Pool(X[features], y, cat_features=cats))
    return m


def main(cfg_path: str):
    t0 = time.time()
    cfg = load_config(cfg_path)
    rc = cfg["reliable"]
    out = Path(cfg["paths"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    mdir = Path(cfg["paths"]["model_dir"]); mdir.mkdir(parents=True, exist_ok=True)
    seed, H, X_months = cfg["seed"], rc["horizon"], rc["train_months"]

    # 1. Cohort và nhãn
    feat_snaps, order_snaps = available_snapshots(cfg)
    cand = [s for s in feat_snaps if s >= rc["first_cohort"]
            and all(shift(s, -k) in order_snaps for k in (0, 1, 2))]
    orders = monthly_orders(cfg, sorted({shift(s, k) for s in cand for k in range(-2, H + 1)} & set(order_snaps)))
    parts, status = [], {}
    for s in cand:
        avail = 0
        while avail < H and shift(s, avail + 1) in order_snaps:
            avail += 1
        c = build_cohort(cfg, s, orders, avail)
        status[s] = c["label_status"].iloc[0]
        parts.append(c)
        print(f"[cohort] {s} n={len(c):,} label={status[s]} bad={c['label'].mean():.3%}", flush=True)
    cohorts = pd.concat(parts, ignore_index=True)
    coh_tab = (cohorts.groupby(["date", "label_status"])
               .agg(n=("cus_id", "size"), n_bad=("label", "sum"), bad_rate=("label", "mean"),
                    n_s0=("in_s0", "sum"), market_factor=("market_factor", "first"))
               .reset_index())
    coh_tab["bad_rate_s0"] = cohorts[cohorts["in_s0"]].groupby("date")["label"].mean().reindex(coh_tab["date"]).to_numpy()
    coh_tab.to_csv(out / "label_rate_by_cohort.csv", index=False)
    rr.label_rate_chart(coh_tab, out)

    windows = make_windows(status, X_months, rc["max_partial_test"])
    if not windows:
        raise ValueError("Không có cửa sổ nào đủ điều kiện; kiểm tra train_months và dữ liệu")
    print("[windows] " + "; ".join(f"{w['train'][0]}-{w['train'][-1]} -> {w['test']} ({w['test_label']})" for w in windows), flush=True)
    full = [s for s in sorted(status) if status[s] == "full"]
    final_train = full[-X_months:]

    # 2. Feature cho mọi cohort dùng tới
    used = sorted({s for w in windows for s in w["train"] + [w["test"]]} | set(final_train))
    data = cohorts[cohorts["date"].isin(used)].reset_index(drop=True)
    feats_df, feature_group = load_features(cfg, data[KEYS])
    data = data.merge(feats_df, on=KEYS, how="left", validate="one_to_one")
    features_all = list(feature_group)
    Xall = prepare_matrix(data, features_all, cfg["categorical_columns"])
    num = [c for c in features_all if Xall[c].dtype != object]
    Xall[num] = Xall[num].astype("float32")
    cats_all = cat_features_of(Xall, features_all)
    yall = data["label"]
    print(f"[data] rows={len(data):,} features={len(features_all)}", flush=True)

    # 3. Optuna một lần trên train của cửa sổ đầu tiên (đứng trước mọi tháng test -> không rò rỉ)
    w0 = np.flatnonzero(data["date"].isin(windows[0]["train"]).to_numpy())
    s0 = w0[sample_by_shop(data.iloc[w0], cfg["optuna"]["sample_rows"], seed)]
    sel0 = select_features(Xall.iloc[s0].reset_index(drop=True), yall.iloc[s0].reset_index(drop=True).astype(int),
                           data["cus_id"].iloc[s0].reset_index(drop=True), features_all, feature_group, cfg)
    tun = tune(Xall.iloc[s0].reset_index(drop=True), yall.iloc[s0].reset_index(drop=True).astype(int),
               data["cus_id"].iloc[s0].reset_index(drop=True), sel0["final"], cfg)
    params = tun["params"]
    tun["trials"].to_csv(out / "optuna_trials.csv", index=False)
    (out / "best_params.json").write_text(json.dumps({**params, "cv_auc": tun["cv_auc"],
                                                       "tuned_on": windows[0]["train"]}, indent=2))

    # 4. Từng cửa sổ
    win_rows, dec_parts, gi_parts, sel_rows, models = [], [], [], [], {}
    for k, w in enumerate(windows):
        tr = np.flatnonzero(data["date"].isin(w["train"]).to_numpy())
        te = np.flatnonzero((data["date"] == w["test"]).to_numpy())
        if k == 0:
            sel = sel0
        else:
            ss = tr[sample_by_shop(data.iloc[tr], cfg["selection"]["sample_rows"], seed)]
            sel = select_features(Xall.iloc[ss].reset_index(drop=True), yall.iloc[ss].reset_index(drop=True).astype(int),
                                  data["cus_id"].iloc[ss].reset_index(drop=True), features_all, feature_group, cfg)
        feats = sel["final"]
        cats = [c for c in feats if c in cats_all]
        model = fit(Xall.iloc[tr], yall.iloc[tr].astype(int), feats, cats, params, cfg)
        models[w["test"]] = (model, feats, cats)
        p_te = pd.Series(model.predict_proba(Xall.iloc[te][feats])[:, 1])
        p_tr = model.predict_proba(Xall.iloc[tr][feats])[:, 1]
        y_te = yall.iloc[te].reset_index(drop=True).astype(int)
        in_s0 = data["in_s0"].iloc[te].reset_index(drop=True)

        m_all, m_s0 = rr.metrics(y_te, p_te), rr.metrics(y_te[in_s0], p_te[in_s0])
        row = {"window": k + 1, "train_from": w["train"][0], "train_to": w["train"][-1], "test": w["test"],
               "test_label": w["test_label"], "n_features": len(feats),
               "train_auc_insample": rr.metrics(yall.iloc[tr].astype(int).reset_index(drop=True), pd.Series(p_tr))["auc"],
               **m_all, **{f"{key}_s0": v for key, v in m_s0.items()},
               "psi_score": ev.psi_numeric(pd.Series(p_tr), p_te, cfg["eval"]["psi_bins"])}
        win_rows.append(row)
        dec = ev.decile_table(y_te, p_te, cfg["eval"]["n_bins"]).assign(test=w["test"], test_label=w["test_label"])
        dec_parts.append(dec)
        fi = pd.DataFrame({"feature": feats, "importance": model.get_feature_importance()})
        fi["group"] = fi["feature"].map(feature_group)
        g = fi.groupby("group")["importance"].sum()
        gi_parts.append(pd.DataFrame({"test": w["test"], "group": g.index, "share": (g / g.sum()).to_numpy()}))
        sel_rows += [{"test": w["test"], "feature": f, "group": feature_group[f], "importance": imp}
                     for f, imp in zip(fi["feature"], fi["importance"])]
        print(f"[window {k + 1}] train {w['train'][0]}-{w['train'][-1]} test {w['test']} ({w['test_label']}): "
              f"AUC {row['auc']:.4f} / S0 {row['auc_s0']:.4f}, {len(feats)} biến", flush=True)

    wm = pd.DataFrame(win_rows)
    deciles = pd.concat(dec_parts, ignore_index=True)
    gi = pd.concat(gi_parts, ignore_index=True)
    sel_df = pd.DataFrame(sel_rows)
    # Độ ổn định tập biến giữa hai cửa sổ liên tiếp
    sets = {t: set(g["feature"]) for t, g in sel_df.groupby("test")}
    ts = sorted(sets)
    jac = pd.DataFrame([{"test_prev": a, "test": b, "jaccard": len(sets[a] & sets[b]) / len(sets[a] | sets[b])}
                        for a, b in zip(ts, ts[1:])])
    fullw = wm[wm["test_label"] == "full"]
    stab = pd.DataFrame([{"metric": m, "set": s, "mean": fullw[c].mean(), "std": fullw[c].std(ddof=0),
                          "min": fullw[c].min(), "max": fullw[c].max(), "n_windows": len(fullw)}
                         for m in ("auc", "gini", "ks") for s, c in (("prediction_set", m), ("s0", m + "_s0"))])
    wm.to_csv(out / "window_metrics.csv", index=False)
    stab.to_csv(out / "stability_summary.csv", index=False)
    deciles.to_csv(out / "decile_bad_rate_by_window.csv", index=False)
    gi.to_csv(out / "importance_by_group_by_window.csv", index=False)
    sel_df.to_csv(out / "selected_features_by_window.csv", index=False)
    jac.to_csv(out / "feature_set_jaccard.csv", index=False)
    rr.window_metrics_chart(wm, out)
    rr.decile_chart(deciles, out)
    rr.psi_chart(wm, out)
    rr.group_importance_heatmap(gi, out)

    # 5. Model cuối: X cohort nhãn đủ mới nhất (dùng lại model cửa sổ có đúng khung train đó nếu có)
    reuse = [w for w in windows if w["train"] == final_train]
    if reuse:
        model, feats, cats = models[reuse[0]["test"]]
    else:
        tr = np.flatnonzero(data["date"].isin(final_train).to_numpy())
        ss = tr[sample_by_shop(data.iloc[tr], cfg["selection"]["sample_rows"], seed)]
        sel = select_features(Xall.iloc[ss].reset_index(drop=True), yall.iloc[ss].reset_index(drop=True).astype(int),
                              data["cus_id"].iloc[ss].reset_index(drop=True), features_all, feature_group, cfg)
        feats = sel["final"]; cats = [c for c in feats if c in cats_all]
        model = fit(Xall.iloc[tr], yall.iloc[tr].astype(int), feats, cats, params, cfg)
    model.save_model(str(mdir / "model.cbm"))
    last_test = np.flatnonzero((data["date"] == windows[-1]["test"]).to_numpy())
    fi_final, gi_final = ev.importance_report(model, feats, feature_group, out)
    ev.shap_report(model, Xall.iloc[last_test].reset_index(drop=True), feats, cats, fi_final,
                   cfg["eval"]["shap_top"], cfg["eval"]["shap_sample"], seed, out)
    meta = {
        "run_name": cfg["run_name"], "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "target": "bad = shop ngừng giao hoặc giảm đơn sâu trong H tháng tới; score 0-100: 100 = ít rủi ro nhất",
        "features": feats, "cat_features": cats, "feature_group": {f: feature_group[f] for f in feats},
        "feature_sources": cfg["feature_sources"], "exclude_columns": cfg["exclude_columns"],
        "categorical_columns": cfg["categorical_columns"], "reliable": rc,
        "train_cohorts": final_train, "params": params,
        "window_auc_mean": float(fullw["auc"].mean()), "window_auc_std": float(fullw["auc"].std(ddof=0)),
    }
    (mdir / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    with pd.ExcelWriter(out / "report_tables.xlsx") as w:
        coh_tab.to_excel(w, sheet_name="label_rate_by_cohort", index=False)
        wm.to_excel(w, sheet_name="window_metrics", index=False)
        stab.to_excel(w, sheet_name="stability_summary", index=False)
        deciles.to_excel(w, sheet_name="decile_bad_rate", index=False)
        gi.to_excel(w, sheet_name="importance_group_window", index=False)
        jac.to_excel(w, sheet_name="feature_set_jaccard", index=False)
        sel_df.to_excel(w, sheet_name="features_by_window", index=False)
        fi_final.to_excel(w, sheet_name="final_importance", index=False)
        gi_final.to_excel(w, sheet_name="final_importance_group", index=False)
        tun["trials"].to_excel(w, sheet_name="optuna_trials", index=False)

    lines = [f"# Reliable score — {meta['trained_at']}", "",
             f"- Nhãn: bad = ngừng giao hoặc đơn {H} tháng tới < {rc['bad_threshold']} × baseline (đã chia hệ số thị trường). "
             f"Prediction set: đơn đủ 3 tháng, TB ≥ {rc['min_orders']} đơn/tháng, thâm niên ≥ {rc['min_tenure']} tháng.",
             f"- Cửa sổ trượt: train {X_months} cohort → test cohort kế tiếp; chọn biến lại trong mỗi cửa sổ; "
             f"tham số Optuna tune một lần trên train cửa sổ đầu ({tun['cv_auc']:.4f} AUC CV).",
             "", "| Cửa sổ | Train | Test | Nhãn test | n test | Tỷ lệ bad | AUC | Gini | KS | AUC S0 | PSI score | Số biến |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in wm.itertuples():
        lines.append(f"| {r.window} | {r.train_from}–{r.train_to} | {r.test} | {r.test_label} | {r.n:,} | {r.bad_rate:.2%} | "
                     f"{r.auc:.4f} | {r.gini:.4f} | {r.ks:.4f} | {r.auc_s0:.4f} | {r.psi_score:.4f} | {r.n_features} |")
    a = stab[(stab["metric"] == "auc") & (stab["set"] == "prediction_set")].iloc[0]
    g_ = stab[(stab["metric"] == "gini") & (stab["set"] == "prediction_set")].iloc[0]
    lines += ["", f"- Độ ổn định ({int(a.n_windows)} cửa sổ nhãn đủ): AUC {a['mean']:.4f} ± {a['std']:.4f} "
                  f"(min {a['min']:.4f}, max {a['max']:.4f}); Gini {g_['mean']:.4f} ± {g_['std']:.4f}.",
              f"- Model cuối train trên {final_train}, {len(feats)} biến → `{mdir}/model.cbm`.",
              f"- Thời gian chạy: {(time.time() - t0) / 60:.1f} phút."]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reliable.yaml")
    main(ap.parse_args().config)
