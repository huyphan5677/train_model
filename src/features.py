"""Chọn biến: lọc thống kê -> chọn theo group (thư mục S3) -> gộp lại và chọn top theo importance cộng dồn."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


def cat_features_of(X: pd.DataFrame, features: list[str]) -> list[str]:
    return [c for c in features if X[c].dtype == object]


def univariate_auc(x: pd.Series, y: pd.Series) -> float:
    """AUC đơn biến, đảo chiều thành >= 0,5; thiếu -> trung vị; categorical -> tỷ lệ nhãn theo nhóm."""
    if x.dtype == object:
        x = x.map(y.groupby(x).mean())
    x = x.fillna(x.median())
    if x.nunique() < 2:
        return 0.5
    auc = roc_auc_score(y, x)
    return max(auc, 1 - auc)


def stat_filter(X: pd.DataFrame, y: pd.Series, features: list[str], cfg: dict, seed: int
                ) -> tuple[list[str], pd.DataFrame]:
    """Bỏ biến thiếu nhiều, gần hằng, và một biến trong mỗi cặp tương quan Spearman cao."""
    sc = cfg["stat_filter"]
    rows = []
    for c in features:
        s = X[c]
        miss = float(s.isna().mean()) if s.dtype != object else float((s == "nan").mean())
        vc = s.value_counts(dropna=False, normalize=True)
        dom = float(vc.iloc[0]) if len(vc) else 1.0
        rows.append({"feature": c, "missing": miss, "dominant": dom, "n_unique": int(s.nunique()),
                     "uni_auc": univariate_auc(s, y)})
    log = pd.DataFrame(rows).set_index("feature")
    log["drop_reason"] = ""
    log.loc[log["missing"] > sc["max_missing"], "drop_reason"] = "missing"
    log.loc[(log["drop_reason"] == "") & ((log["dominant"] > sc["max_dominant"]) | (log["n_unique"] <= 1)),
            "drop_reason"] = "constant"

    num = [c for c in log.index[log["drop_reason"] == ""] if X[c].dtype != object]
    sample = X[num].sample(min(len(X), sc["corr_sample"]), random_state=seed)
    corr = sample.rank().corr().abs()
    order = log.loc[num, "uni_auc"].sort_values(ascending=False).index.tolist()
    kept: list[str] = []
    for c in order:  # giữ biến AUC đơn biến cao hơn trong mỗi cặp tương quan
        hit = [k for k in kept if corr.at[c, k] > sc["max_spearman"]]
        if hit:
            log.at[c, "drop_reason"] = f"spearman>{sc['max_spearman']} with {hit[0]}"
        else:
            kept.append(c)
    keep = [c for c in features if log.at[c, "drop_reason"] == ""]
    return keep, log.reset_index()


def make_folds(y: pd.Series, groups: pd.Series, n_folds: int, seed: int):
    return list(StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed).split(
        np.zeros(len(y)), y, groups))


def cv_catboost(X: pd.DataFrame, y: pd.Series, folds, features: list[str], params: dict,
                early_stopping: int, thread_count: int, seed: int) -> dict:
    """CV CatBoost: trả về AUC từng fold, OOF, importance trung bình, best_iteration từng fold."""
    cats = cat_features_of(X, features)
    oof = np.zeros(len(y))
    imps, aucs, iters = [], [], []
    for tr, va in folds:
        m = CatBoostClassifier(**params, eval_metric="AUC", random_seed=seed, verbose=False,
                               thread_count=thread_count, allow_writing_files=False)
        m.fit(Pool(X.iloc[tr][features], y.iloc[tr], cat_features=cats),
              eval_set=Pool(X.iloc[va][features], y.iloc[va], cat_features=cats),
              early_stopping_rounds=early_stopping, use_best_model=True)
        p = m.predict_proba(X.iloc[va][features])[:, 1]
        oof[va] = p
        aucs.append(roc_auc_score(y.iloc[va], p))
        iters.append(m.get_best_iteration() + 1)
        imps.append(pd.Series(m.get_feature_importance(), index=features))
    return {"auc_folds": aucs, "auc": float(np.mean(aucs)), "oof": oof,
            "importance": pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False),
            "best_iterations": iters}


def cum_select(imp: pd.Series, threshold: float, min_n: int) -> list[str]:
    """Giữ feature theo thứ tự importance giảm dần tới khi cộng dồn >= threshold."""
    imp = imp.sort_values(ascending=False).clip(lower=0)
    if imp.sum() <= 0:
        return imp.index[:min_n].tolist()
    share = imp.cumsum() / imp.sum()
    n = int((share < threshold).sum()) + 1
    return imp.index[:max(n, min_n)].tolist()


def select_features(X: pd.DataFrame, y: pd.Series, groups: pd.Series, features: list[str],
                    feature_group: dict[str, str], cfg: dict) -> dict:
    seed, threads = cfg["seed"], cfg["thread_count"]
    sel = cfg["selection"]
    cb = {k: v for k, v in sel["catboost"].items() if k != "early_stopping_rounds"}
    es = sel["catboost"]["early_stopping_rounds"]
    folds = make_folds(y, groups, sel["cv_folds"], seed)

    stat_keep, stat_log = stat_filter(X, y, features, cfg, seed)
    stat_log["group"] = stat_log["feature"].map(feature_group)
    print(f"[select] stat filter: {len(features)} -> {len(stat_keep)}", flush=True)

    group_rows, group_keep = [], []
    for grp in sorted({feature_group[f] for f in stat_keep}):
        feats = [f for f in stat_keep if feature_group[f] == grp]
        res = cv_catboost(X, y, folds, feats, cb, es, threads, seed)
        keep = cum_select(res["importance"], sel["cum_importance"], sel["min_features_group"])
        group_keep += keep
        imp = res["importance"]
        for f in feats:
            group_rows.append({"group": grp, "feature": f, "importance": float(imp[f]),
                               "selected": f in keep, "group_cv_auc": res["auc"]})
        print(f"[select] group {grp:24s} {len(feats):3d} -> {len(keep):3d} (cv auc {res['auc']:.3f})", flush=True)
    group_log = pd.DataFrame(group_rows)

    res = cv_catboost(X, y, folds, group_keep, cb, es, threads, seed)
    final = cum_select(res["importance"], sel["cum_importance"], sel["min_features_final"])
    combined_log = pd.DataFrame({"feature": res["importance"].index, "importance": res["importance"].values})
    combined_log["group"] = combined_log["feature"].map(feature_group)
    combined_log["cum_share"] = combined_log["importance"].clip(lower=0).cumsum() / combined_log["importance"].clip(lower=0).sum()
    combined_log["selected"] = combined_log["feature"].isin(final)
    print(f"[select] combined: {len(group_keep)} -> {len(final)} (cv auc {res['auc']:.3f})", flush=True)
    return {"final": final, "stat_log": stat_log, "group_log": group_log,
            "combined_log": combined_log, "combined_cv_auc": res["auc"]}
