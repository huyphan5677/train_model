"""Chọn biến theo các bước khai báo ở cfg['selection']['steps'] (xem METHODS). Mặc định: lọc thống kê ->
chọn theo group (thư mục S3) -> gộp lại và chọn top theo importance cộng dồn."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data import MISSING


def cat_features_of(X: pd.DataFrame, features: list[str]) -> list[str]:
    return [c for c in features if X[c].dtype == object]


def univariate_auc(x: pd.Series, y: pd.Series) -> float:
    """AUC đơn biến, đảo chiều thành >= 0,5; thiếu -> trung vị; categorical -> tỷ lệ nhãn theo nhóm."""
    if x.dtype == object:
        x = x.map(y.groupby(x).mean()) # label rate
    x = x.fillna(x.median())
    if x.nunique() < 2:
        return 0.5
    auc = roc_auc_score(y, x)
    return max(auc, 1 - auc) # auc = 0.3 va auc = 0.7 strong nhu nhau (chu yeu la nguoc trend voi label)


def scaled_variances(X: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Phương sai không phụ thuộc đơn vị của biến số: cắt p1–p99, MinMaxScaler về [0, 1].
    Thiếu đặt dưới min một khoảng bằng biên độ (như nan_mode=Min của CatBoost): {1, NaN} không bị coi là phương sai thấp.
    Cột không được toàn thiếu (gọi sau bước lọc missing)."""
    A = X[cols].astype("float64")
    mn, mx = A.min(), A.max()
    A = A.fillna(mn - (mx - mn).where(mx > mn, 1.0))
    A = A.clip(A.quantile(0.01), A.quantile(0.99), axis=1)
    return pd.Series(MinMaxScaler().fit_transform(A).var(axis=0), index=cols)


def equal_share(A: np.ndarray) -> np.ndarray:
    """Ma trận tỷ lệ dòng mà hai cột có giá trị giống hệt (cùng thiếu cũng tính là giống)."""
    p = A.shape[1]
    out = np.eye(p)
    na = pd.isna(A)
    for i in range(p - 1):
        eq = (A[:, [i]] == A[:, i + 1:]) | (na[:, [i]] & na[:, i + 1:])
        out[i, i + 1:] = out[i + 1:, i] = eq.mean(axis=0)
    return out


def stat_filter(X: pd.DataFrame, y: pd.Series, features: list[str], cfg: dict, seed: int
                ) -> tuple[list[str], pd.DataFrame]:
    """Bỏ biến thiếu nhiều, gần hằng, phương sai thấp, cột trùng hệt, và một biến trong mỗi cặp
    tương quan Spearman cao."""
    sc = cfg["stat_filter"]
    rows = []
    for c in features:
        s = X[c]
        miss = float(s.isna().mean()) if s.dtype != object else float((s == MISSING).mean())
        vc = s.value_counts(dropna=False, normalize=True)
        dom = float(vc.iloc[0]) if len(vc) else 1.0
        # thiếu tính là một giá trị cho cả số lẫn categorical: {1, NaN} không phải hằng
        rows.append({"feature": c, "missing": miss, "dominant": dom, "n_unique": int(s.nunique(dropna=False)),
                     "uni_auc": univariate_auc(s, y)})
    log = pd.DataFrame(rows).set_index("feature")
    log["drop_reason"] = ""
    log.loc[log["missing"] > sc["max_missing"], "drop_reason"] = "missing"
    log.loc[(log["drop_reason"] == "") & ((log["dominant"] > sc["max_dominant"]) | (log["n_unique"] <= 1)),
            "drop_reason"] = "constant"

    # phương sai thấp: chỉ biến số, categorical bỏ qua
    num = [c for c in log.index[log["drop_reason"] == ""] if X[c].dtype != object]
    log["scaled_var"] = np.nan
    if num:
        log.loc[num, "scaled_var"] = scaled_variances(X, num)
    log.loc[(log["drop_reason"] == "") & (log["scaled_var"] < sc["min_scaled_var"]), "drop_reason"] = "low_variance"

    # cột gần trùng: >= max_equal_share số dòng giống hệt; chỉ so trong cùng loại (số / categorical).
    # Giữ biến AUC đơn biến cao hơn, bằng nhau (trùng hệt) thì giữ cột đứng trước theo thứ tự feature_sources.
    alive = log.index[log["drop_reason"] == ""].tolist()
    idx = X.sample(min(len(X), sc["corr_sample"]), random_state=seed).index
    for cols, dt in (([c for c in alive if X[c].dtype != object], float), ([c for c in alive if X[c].dtype == object], object)):
        if len(cols) < 2:
            continue
        share = pd.DataFrame(equal_share(X.loc[idx, cols].to_numpy(dtype=dt)), index=cols, columns=cols)
        uniq: list[str] = []
        for c in log.loc[cols, "uni_auc"].sort_values(ascending=False, kind="stable").index:
            hit = [k for k in uniq if share.at[c, k] >= sc["max_equal_share"]]
            if hit:
                log.at[c, "drop_reason"] = f"duplicate>={sc['max_equal_share']} with {hit[0]}"
            else:
                uniq.append(c)

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


# ---------------------------------------------------------------------------------------------------------
# Phương pháp chọn biến. Mỗi hàm nhận (X, y, groups, features, params, ctx) và trả về DataFrame có ít nhất
# [feature, score, selected]. Biến mà phương pháp không đánh giá được (vd. categorical với phương pháp tuyến
# tính) có score NaN và selected=True (cho qua) để không bị loại oan. Mọi phép chia train/validation đều theo
# nhóm cus_id (StratifiedGroupKFold) để một shop không nằm ở cả hai phía.
# ---------------------------------------------------------------------------------------------------------

def _pick(score: pd.Series, params: dict, defaults: dict) -> pd.Series:
    """Chọn theo score (cao = tốt), các điều kiện kết hợp AND; đặt null trong config để tắt một điều kiện.
    min_score: score >= ngưỡng; min_score_rel_mean: score >= hệ số × trung bình; top_k / top_pct: k biến / tỷ lệ đầu."""
    keys = ("min_score", "min_score_rel_mean", "top_k", "top_pct")
    p = {**defaults, **{k: params[k] for k in keys if k in params}}
    s = score.dropna()
    ok = pd.Series(True, index=s.index)
    if p.get("min_score") is not None:
        ok &= s >= p["min_score"]
    if p.get("min_score_rel_mean") is not None:
        ok &= s >= p["min_score_rel_mean"] * s.mean()
    rank = s.rank(ascending=False, method="first")
    if p.get("top_k") is not None:
        ok &= rank <= p["top_k"]
    if p.get("top_pct") is not None:
        ok &= rank <= np.ceil(p["top_pct"] * len(s))
    return ok.reindex(score.index, fill_value=True).astype(bool)


def _codes(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Ma trận số cho model sklearn: categorical -> mã thứ tự (MISSING là một mã riêng), số giữ NaN."""
    return pd.DataFrame({c: pd.factorize(X[c])[0] if X[c].dtype == object else X[c].astype("float64")
                         for c in features}, index=X.index)


def _imputed(X: pd.DataFrame, features: list[str], scale: bool) -> pd.DataFrame:
    """Biến số điền trung vị (+ StandardScaler nếu scale) cho phương pháp không nhận NaN."""
    A = X[features].astype("float64")
    A = A.fillna(A.median()).fillna(0.0)
    if scale:
        A = pd.DataFrame(StandardScaler().fit_transform(A), columns=features, index=X.index)
    return A


def _rf(params: dict, ctx: dict) -> RandomForestClassifier:
    return RandomForestClassifier(max_depth=params.get("max_depth", 6), n_estimators=params.get("n_estimators", 1000),
                                  random_state=ctx["cfg"]["seed"], n_jobs=ctx["cfg"]["thread_count"])


def _catboost_params(params: dict, ctx: dict) -> tuple[dict, int]:
    cb = {**ctx["cfg"]["selection"]["catboost"], **params.get("catboost", {})}
    es = cb.pop("early_stopping_rounds")
    return cb, es


def m_stat_filter(X, y, groups, features, params, ctx) -> pd.DataFrame:
    """Lọc thống kê (missing, constant, low_variance, duplicate, Spearman); params ghi đè cfg['stat_filter']."""
    cfg = {**ctx["cfg"], "stat_filter": {**ctx["cfg"]["stat_filter"], **params}}
    keep, log = stat_filter(X, y, features, cfg, cfg["seed"])
    log["score"] = log["uni_auc"]
    log["selected"] = log["feature"].isin(keep)
    return log


def m_group_importance(X, y, groups, features, params, ctx) -> pd.DataFrame:
    """CatBoost CV trên biến của từng group, giữ tới cum_importance importance cộng dồn."""
    cfg, sel = ctx["cfg"], ctx["cfg"]["selection"]
    cb, es = _catboost_params(params, ctx)
    folds = make_folds(y, groups, params.get("cv_folds", sel["cv_folds"]), cfg["seed"])
    cum, min_n = params.get("cum_importance", sel["cum_importance"]), params.get("min_features", sel["min_features_group"])
    rows = []
    for grp in sorted({ctx["feature_group"][f] for f in features}):
        feats = [f for f in features if ctx["feature_group"][f] == grp]
        res = cv_catboost(X, y, folds, feats, cb, es, cfg["thread_count"], cfg["seed"])
        keep = cum_select(res["importance"], cum, min_n)
        rows += [{"feature": f, "score": float(res["importance"][f]), "selected": f in keep, "group_cv_auc": res["auc"]}
                 for f in feats]
        print(f"[select]   group {grp:24s} {len(feats):3d} -> {len(keep):3d} (cv auc {res['auc']:.3f})", flush=True)
    return pd.DataFrame(rows)


def m_cum_importance(X, y, groups, features, params, ctx) -> pd.DataFrame:
    """CatBoost CV trên toàn bộ biến, giữ tới cum_importance importance cộng dồn."""
    cfg, sel = ctx["cfg"], ctx["cfg"]["selection"]
    cb, es = _catboost_params(params, ctx)
    folds = make_folds(y, groups, params.get("cv_folds", sel["cv_folds"]), cfg["seed"])
    res = cv_catboost(X, y, folds, features, cb, es, cfg["thread_count"], cfg["seed"])
    keep = cum_select(res["importance"], params.get("cum_importance", sel["cum_importance"]),
                      params.get("min_features", sel["min_features_final"]))
    log = pd.DataFrame({"feature": res["importance"].index, "score": res["importance"].values})
    log["cum_share"] = log["score"].clip(lower=0).cumsum() / log["score"].clip(lower=0).sum()
    log["selected"] = log["feature"].isin(keep)
    log["cv_auc"] = res["auc"]
    return log


def m_permutation(X, y, groups, features, params, ctx) -> pd.DataFrame:
    """Permutation importance: AUC validation giảm bao nhiêu khi xáo trộn từng biến (RandomForest).
    params: repeat (1), test_size (0,25), max_depth, n_estimators; mặc định giữ biến có score > 0."""
    seed, repeat = ctx["cfg"]["seed"], params.get("repeat", 1)
    A = _codes(X, features)
    deltas = {f: [] for f in features}
    for i in range(repeat):
        tr, va = make_folds(y, groups, round(1 / params.get("test_size", 0.25)), seed + i)[0]
        model = _rf(params, ctx).fit(A.iloc[tr], y.iloc[tr])
        Xv, yv = A.iloc[va].copy(), y.iloc[va]
        base = roc_auc_score(yv, model.predict_proba(Xv)[:, 1])
        rng = np.random.RandomState(seed + i)
        for f in features:
            orig = Xv[f].to_numpy().copy()
            Xv[f] = rng.permutation(orig)
            deltas[f].append(base - roc_auc_score(yv, model.predict_proba(Xv)[:, 1]))
            Xv[f] = orig
    score = pd.Series({f: float(np.mean(d)) for f, d in deltas.items()})
    return pd.DataFrame({"feature": features, "score": score[features].values,
                         "selected": _pick(score, params, {"min_score": 1e-12})[features].values})


def m_single_auc(X, y, groups, features, params, ctx) -> pd.DataFrame:
    """AUC CV của model chỉ dùng một biến (DecisionTree nông, bắt được quan hệ không đơn điệu).
    params: cv (5), max_depth (4), min_samples_leaf (100); mặc định giữ biến có score >= 0,52."""
    folds = make_folds(y, groups, params.get("cv", 5), ctx["cfg"]["seed"])
    A = _codes(X, features)
    scores = {}
    for f in features:
        aucs = []
        for tr, va in folds:
            m = DecisionTreeClassifier(max_depth=params.get("max_depth", 4), min_samples_leaf=params.get("min_samples_leaf", 100),
                                       random_state=ctx["cfg"]["seed"]).fit(A.iloc[tr][[f]], y.iloc[tr])
            aucs.append(roc_auc_score(y.iloc[va], m.predict_proba(A.iloc[va][[f]])[:, 1]))
        scores[f] = float(np.mean(aucs))
    score = pd.Series(scores)
    return pd.DataFrame({"feature": features, "score": score[features].values,
                         "selected": _pick(score, params, {"min_score": 0.52})[features].values})


def m_beat_random(X, y, groups, features, params, ctx) -> pd.DataFrame:
    """Importance RandomForest khi thêm một cột ngẫu nhiên (nhãn xáo trộn); giữ biến quan trọng hơn cột ngẫu nhiên."""
    A = _codes(X, features)
    A["RANDOM_COLUMN"] = np.random.RandomState(101).permutation(y.to_numpy())
    model = _rf(params, ctx).fit(A, y)
    imp = pd.Series(model.feature_importances_, index=A.columns)
    rand = float(imp["RANDOM_COLUMN"])
    score = imp[features]
    return pd.DataFrame({"feature": features, "score": score.values, "random_importance": rand,
                         "selected": _pick(score, params, {"min_score": np.nextafter(rand, np.inf)})[features].values})


def _linear(X, y, features, params, ctx, penalty: str) -> pd.DataFrame:
    num = [f for f in features if X[f].dtype != object]
    model = LogisticRegression(penalty=penalty, solver="liblinear", C=params.get("C", 1.0), random_state=ctx["cfg"]["seed"])
    model.fit(_imputed(X, num, scale=True), y)
    score = pd.Series(np.abs(model.coef_[0]), index=num).reindex(features)
    return pd.DataFrame({"feature": features, "score": score.values,
                         "selected": _pick(score, params, {"min_score_rel_mean": 0.1}).values})


def m_l1(X, y, groups, features, params, ctx) -> pd.DataFrame:
    """|hệ số| LogisticRegression L1 (điền trung vị + StandardScaler); mặc định giữ |hệ số| >= 0,1 × trung bình.
    Categorical không đánh giá (cho qua)."""
    return _linear(X, y, features, params, ctx, "l1")


def m_l2(X, y, groups, features, params, ctx) -> pd.DataFrame:
    """Như l1 nhưng phạt L2."""
    return _linear(X, y, features, params, ctx, "l2")


def m_mutual_info(X, y, groups, features, params, ctx) -> pd.DataFrame:
    """Mutual information với nhãn (số điền trung vị, categorical là biến rời rạc); mặc định giữ 75% biến đầu."""
    num = [f for f in features if X[f].dtype != object]
    A = pd.concat([_imputed(X, num, scale=False), _codes(X, [f for f in features if f not in num])], axis=1)[features]
    mi = mutual_info_classif(A, y, discrete_features=[f not in num for f in features], random_state=ctx["cfg"]["seed"])
    score = pd.Series(mi, index=features)
    return pd.DataFrame({"feature": features, "score": score.values,
                         "selected": _pick(score, params, {"top_pct": 0.75}).values})


def m_target_corr(X, y, groups, features, params, ctx) -> pd.DataFrame:
    """|Tương quan Pearson| với nhãn (số điền trung vị); mặc định |corr| >= 0,1 và top 10. Categorical cho qua.
    Lưu ý: nhãn hiếm thì |corr| rất nhỏ, nên hạ min_score."""
    num = [f for f in features if X[f].dtype != object]
    # cột hằng sau khi điền thiếu cho corr NaN -> 0 (đã đánh giá, không tương quan), khác categorical (không đánh giá)
    score = _imputed(X, num, scale=False).corrwith(y.astype("float64")).abs().fillna(0.0).reindex(features)
    return pd.DataFrame({"feature": features, "score": score.values,
                         "selected": _pick(score, params, {"min_score": 0.1, "top_k": 10}).values})


def m_rfe(X, y, groups, features, params, ctx) -> pd.DataFrame:
    """RFE với DecisionTree: loại dần step biến kém nhất tới khi còn k (số nguyên, hoặc tỷ lệ < 1; mặc định 0,5).
    score = -hạng (hạng 1 = được chọn)."""
    k = params.get("k", 0.5)
    k = max(1, int(round(k * len(features)))) if k < 1 else int(k)
    rfe = RFE(DecisionTreeClassifier(max_depth=params.get("max_depth"), random_state=ctx["cfg"]["seed"]),
              n_features_to_select=min(k, len(features)), step=params.get("step", 30))
    rfe.fit(_codes(X, features), y)
    return pd.DataFrame({"feature": features, "score": -rfe.ranking_, "selected": rfe.ranking_ == 1})


def m_vote(X, y, groups, features, params, ctx) -> pd.DataFrame:
    """Chạy nhiều phương pháp trên cùng tập biến, giữ biến được >= min_share (0,5) số phương pháp đánh giá được nó chọn.
    params: methods (danh sách bước như ở selection.steps), min_share."""
    log = pd.DataFrame(index=pd.Index(features, name="feature"))
    votes = pd.Series(0, index=features)
    n_eval = pd.Series(0, index=features)
    for j, step in enumerate(params["methods"], 1):
        r = run_method(step, X, y, groups, features, ctx).set_index("feature").reindex(features)
        name = f"{j}_{step['method']}"
        ev = r["score"].notna()
        log[f"{name}_score"], log[f"{name}_selected"] = r["score"], r["selected"]
        votes += (r["selected"].astype(bool) & ev).astype(int)
        n_eval += ev.astype(int)
        print(f"[select]   vote {name}: chọn {int((r['selected'].astype(bool) & ev).sum())}/{int(ev.sum())}", flush=True)
    log["votes"], log["n_methods"] = votes, n_eval
    log["score"] = votes / n_eval.replace(0, np.nan)
    log["selected"] = log["score"].fillna(1.0) >= params.get("min_share", 0.5)
    return log.reset_index()


METHODS = {
    "stat_filter": m_stat_filter,
    "group_importance": m_group_importance,
    "cum_importance": m_cum_importance,
    "permutation": m_permutation,
    "single_auc": m_single_auc,
    "beat_random": m_beat_random,
    "l1": m_l1,
    "l2": m_l2,
    "mutual_info": m_mutual_info,
    "target_corr": m_target_corr,
    "rfe": m_rfe,
    "vote": m_vote,
}
DEFAULT_STEPS = [{"method": "stat_filter"}, {"method": "group_importance"}, {"method": "cum_importance"}]


def _sample_groups(X, y, groups, n: int, seed: int):
    """Mẫu ~n dòng, lấy trọn mọi dòng của các cus_id được chọn."""
    if len(X) <= n:
        return X, y, groups
    g = groups.drop_duplicates().sample(frac=n / len(X), random_state=seed)
    m = groups.isin(set(g)).to_numpy()
    return X[m].reset_index(drop=True), y[m].reset_index(drop=True), groups[m].reset_index(drop=True)


def run_method(step: dict, X, y, groups, features: list[str], ctx: dict) -> pd.DataFrame:
    """Chạy một bước {method: ..., <params>}; sample_rows (tuỳ chọn) chạy bước đó trên mẫu theo cus_id."""
    if step["method"] not in METHODS:
        raise ValueError(f"Phương pháp không có: {step['method']}. Có: {sorted(METHODS)}")
    params = {k: v for k, v in step.items() if k != "method"}
    if params.get("sample_rows"):
        X, y, groups = _sample_groups(X, y, groups, params["sample_rows"], ctx["cfg"]["seed"])
    log = METHODS[step["method"]](X, y, groups, features, params, ctx)
    log["group"] = log["feature"].map(ctx["feature_group"])
    return log


def select_features(X: pd.DataFrame, y: pd.Series, groups: pd.Series, features: list[str],
                    feature_group: dict[str, str], cfg: dict) -> dict:
    """Chạy lần lượt các bước trong cfg['selection']['steps'] (mặc định DEFAULT_STEPS); mỗi bước nhận tập biến
    còn lại của bước trước. Trả về final (thứ tự theo log bước cuối), logs [(tên, DataFrame)], counts [(tên, số biến)]."""
    ctx = {"cfg": cfg, "feature_group": feature_group}
    feats = list(features)
    logs, counts = [], [("input", len(feats))]
    for i, step in enumerate(cfg["selection"].get("steps") or DEFAULT_STEPS, 1):
        name = f"{i}_{step['method']}"
        log = run_method(step, X, y, groups, feats, ctx)
        alive = set(feats)
        feats = [f for f in log.loc[log["selected"].astype(bool), "feature"] if f in alive]
        if not feats:
            raise ValueError(f"Bước {name} loại hết biến; nới ngưỡng trong config")
        logs.append((name, log))
        counts.append((name, len(feats)))
        print(f"[select] {name}: {counts[-2][1]} -> {len(feats)}", flush=True)
    return {"final": feats, "logs": logs, "counts": counts}
