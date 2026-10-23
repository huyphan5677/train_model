"""Bảng và biểu đồ đánh giá cho báo cáo."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, roc_curve

# Bảng màu tham chiếu (dataviz skill): train = slot 1, test = slot 2; chữ và lưới trung tính
C_TRAIN, C_TEST = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight",
})


def month_label(d: int) -> str:
    s = str(d)
    return f"{s[4:6]}/{s[:4]}"


def _save(fig, path: Path):
    fig.savefig(path)
    plt.close(fig)


def gini(auc: float) -> float:
    return 2 * auc - 1


# ---------------------------------------------------------------- nhãn theo thời gian
def label_rate(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    t = (df.groupby(["date", "split"]).agg(n=("label", "size"), n_pos=("label", "sum"))
         .reset_index().sort_values("date"))
    t["label_rate"] = t["n_pos"] / t["n"]
    t.to_csv(out / "label_rate_by_month.csv", index=False)

    x = np.arange(len(t))
    colors = [C_TRAIN if s == "train" else C_TEST for s in t["split"]]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    a1.bar(x, t["label_rate"] * 100, color=colors, width=0.55)
    for i, r in enumerate(t.itertuples()):
        a1.text(i, r.label_rate * 100, f"{r.label_rate:.2%}", ha="center", va="bottom", color=INK, fontsize=9)
    a1.set_ylabel("Tỷ lệ nộp hồ sơ (%)")
    a1.set_ylim(0, t["label_rate"].max() * 100 * 1.2)
    a1.set_title("Tỷ lệ nhãn theo tháng snapshot")
    a2.bar(x, t["n_pos"], color=colors, width=0.55)
    for i, r in enumerate(t.itertuples()):
        a2.text(i, r.n_pos, f"{r.n_pos:,} / {r.n:,}", ha="center", va="bottom", color=INK, fontsize=9)
    a2.set_ylabel("Số shop nộp hồ sơ")
    a2.set_ylim(0, t["n_pos"].max() * 1.2)
    a2.set_xticks(x, [month_label(d) for d in t["date"]])
    a2.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_TRAIN), plt.Rectangle((0, 0), 1, 1, color=C_TEST)],
              labels=["train", "test"], frameon=False, loc="upper left", ncols=2)
    _save(fig, out / "label_rate_by_month.png")
    return t


# ---------------------------------------------------------------- AUC / Gini
def auc_by_month(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    """df: date, split, label, score (train = OOF CV, test = model cuối)."""
    rows = []
    for (d, s), g in df.groupby(["date", "split"]):
        auc = roc_auc_score(g["label"], g["score"]) if g["label"].nunique() == 2 else np.nan
        rows.append({"date": d, "split": s, "n": len(g), "n_pos": int(g["label"].sum()), "auc": auc, "gini": gini(auc)})
    for s, g in df.groupby("split"):
        auc = roc_auc_score(g["label"], g["score"])
        rows.append({"date": f"all_{s}", "split": s, "n": len(g), "n_pos": int(g["label"].sum()), "auc": auc, "gini": gini(auc)})
    t = pd.DataFrame(rows)
    t.to_csv(out / "auc_gini_by_month.csv", index=False)

    m = t[~t["date"].astype(str).str.startswith("all_")].sort_values("date")
    x = np.arange(len(m))
    colors = [C_TRAIN if s == "train" else C_TEST for s in m["split"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col, name in [(axes[0], "auc", "AUC"), (axes[1], "gini", "Gini")]:
        ax.bar(x, m[col], color=colors, width=0.55)
        for i, v in enumerate(m[col]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_xticks(x, [month_label(d) for d in m["date"]])
        ax.set_title(f"{name} theo tháng (train = OOF CV, test = out-of-time)")
        ax.set_ylim(0.5 if col == "auc" else 0, 1 if col == "auc" else max(0.6, m[col].max() * 1.15))
    axes[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_TRAIN), plt.Rectangle((0, 0), 1, 1, color=C_TEST)],
                   labels=["train", "test"], frameon=False)
    _save(fig, out / "auc_gini_by_month.png")

    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    for s, c in [("train", C_TRAIN), ("test", C_TEST)]:
        g = df[df["split"] == s]
        fpr, tpr, _ = roc_curve(g["label"], g["score"])
        auc = roc_auc_score(g["label"], g["score"])
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{s}: AUC {auc:.3f}, Gini {gini(auc):.3f}")
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, out / "roc_train_test.png")
    return t


# ---------------------------------------------------------------- precision theo decile
def decile_table(label: pd.Series, score: pd.Series, n_bins: int) -> pd.DataFrame:
    """Bin 1 = 10% điểm cao nhất. Chia theo thứ hạng để tránh trùng điểm."""
    rank = score.rank(ascending=False, method="first")
    bins = pd.qcut(rank, n_bins, labels=range(1, n_bins + 1))
    d = pd.DataFrame({"bin": bins.astype(int), "label": label.values, "score": score.values})
    t = d.groupby("bin").agg(n=("label", "size"), n_pos=("label", "sum"),
                             score_min=("score", "min"), score_max=("score", "max")).reset_index()
    base = label.mean()
    t["precision"] = t["n_pos"] / t["n"]
    t["cum_n"] = t["n"].cumsum()
    t["cum_pos"] = t["n_pos"].cumsum()
    t["cum_precision"] = t["cum_pos"] / t["cum_n"]
    t["cum_recall"] = t["cum_pos"] / label.sum()
    t["lift"] = t["precision"] / base
    t["cum_lift"] = t["cum_precision"] / base
    return t


def precision_tables(df_test: pd.DataFrame, n_bins: int, out: Path) -> pd.DataFrame:
    parts = [decile_table(df_test["label"], df_test["score"], n_bins).assign(date="all_test")]
    for d, g in df_test.groupby("date"):
        parts.append(decile_table(g["label"].reset_index(drop=True), g["score"].reset_index(drop=True), n_bins).assign(date=d))
    t = pd.concat(parts, ignore_index=True)
    t = t[["date"] + [c for c in t.columns if c != "date"]]
    t.to_csv(out / "precision_by_decile_test.csv", index=False)

    plot_precision(t, out)
    return t


def plot_precision(t: pd.DataFrame, out: Path):
    """Vẽ từ bảng precision (vẽ lại được từ precision_by_decile_test.csv)."""
    a = t[t["date"].astype(str) == "all_test"]
    base = a["n_pos"].sum() / a["n"].sum()
    n_bins = len(a)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(a["bin"], a["precision"] * 100, color=C_TEST, width=0.6, label="Precision trong bin")
    ax.plot(a["bin"], a["cum_precision"] * 100, color=INK, lw=2, marker="o", ms=5, label="Precision cộng dồn từ bin 1")
    ax.axhline(base * 100, color=INK2, ls="--", lw=1, label=f"Tỷ lệ chung {base:.2%}")
    for r in a.itertuples():
        ax.text(r.bin, r.precision * 100, f"{r.precision:.2%}\n×{r.lift:.1f}", ha="center", va="bottom", color=INK, fontsize=8)
    ax.set_ylim(0, a["precision"].max() * 100 * 1.18)
    ax.set_xticks(a["bin"], [f"{b}" for b in a["bin"]])
    ax.set_xlim(0.4, n_bins + 0.6)
    ax.set_xlabel("Bin điểm (1 = 10% điểm cao nhất)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("Precision theo decile điểm — tập test")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, out / "precision_by_decile_test.png")


# ---------------------------------------------------------------- PSI
def _psi(exp_share: np.ndarray, act_share: np.ndarray, eps: float = 1e-4) -> float:
    e = np.clip(exp_share, eps, None)
    a = np.clip(act_share, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(expected: pd.Series, actual: pd.Series, n_bins: int) -> float:
    e = pd.to_numeric(expected, errors="coerce")
    a = pd.to_numeric(actual, errors="coerce")
    edges = np.unique(np.nanquantile(e.dropna(), np.linspace(0, 1, n_bins + 1))) if e.notna().any() else np.array([])
    if len(edges) < 2:
        cats = lambda s: s.isna().map({True: "NA", False: "val"})
        return psi_categorical(cats(e), cats(a))
    edges[0], edges[-1] = -np.inf, np.inf
    def shares(s):
        b = pd.cut(s, edges, include_lowest=True).cat.codes.replace(-1, len(edges) - 1)  # NaN -> bin riêng
        return np.bincount(b, minlength=len(edges)) / len(s)
    return _psi(shares(e), shares(a))


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    cats = sorted(set(expected.astype(str)) | set(actual.astype(str)))
    e = expected.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    a = actual.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(e, a)


def psi_report(train: pd.DataFrame, test: pd.DataFrame, features: list[str], cats: list[str],
               n_bins: int, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows, feat_rows = [], []
    for d, g in test.groupby("date"):
        score_rows.append({"date": d, "psi_score": psi_numeric(train["score"], g["score"], n_bins)})
        for f in features:
            v = psi_categorical(train[f], g[f]) if f in cats else psi_numeric(train[f], g[f], n_bins)
            feat_rows.append({"date": d, "feature": f, "psi": v})
    score_rows.append({"date": "all_test", "psi_score": psi_numeric(train["score"], test["score"], n_bins)})
    ps = pd.DataFrame(score_rows)
    pf = pd.DataFrame(feat_rows).pivot(index="feature", columns="date", values="psi")
    pf.columns = [f"psi_{c}" for c in pf.columns]
    pf["psi_max"] = pf.max(axis=1)
    pf = pf.sort_values("psi_max", ascending=False).reset_index()
    ps.to_csv(out / "psi_score_test.csv", index=False)
    pf.to_csv(out / "psi_features_test.csv", index=False)

    top = pf.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(top) + 1)))
    ax.barh(top["feature"], top["psi_max"], color=C_TEST, height=0.6)
    for th, lab in [(0.1, "0,1"), (0.25, "0,25")]:
        ax.axvline(th, color=INK2, ls="--", lw=1)
        ax.text(th, len(top) - 0.3, lab, color=INK2, fontsize=8, ha="left")
    ax.set_xlabel("PSI lớn nhất giữa các tháng test (so với train)")
    ax.set_title(f"PSI feature — 20 biến dịch chuyển nhiều nhất\nPSI score: " +
                 ", ".join(f"{month_label(r.date) if str(r.date).isdigit() else 'cả test'} {r.psi_score:.3f}" for r in ps.itertuples()))
    ax.grid(axis="y", visible=False)
    _save(fig, out / "psi_features_test.png")
    return ps, pf


# ---------------------------------------------------------------- importance và SHAP
def importance_report(model: CatBoostClassifier, features: list[str], feature_group: dict[str, str],
                      out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fi = pd.DataFrame({"feature": features, "importance": model.get_feature_importance()})
    fi["group"] = fi["feature"].map(feature_group)
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
    fi["share"] = fi["importance"] / fi["importance"].sum()
    fi.to_csv(out / "feature_importance.csv", index=False)
    gi = (fi.groupby("group").agg(n_features=("feature", "size"), importance=("importance", "sum"))
          .sort_values("importance", ascending=False).reset_index())
    gi["share"] = gi["importance"] / gi["importance"].sum()
    gi.to_csv(out / "feature_importance_by_group.csv", index=False)

    g = gi.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(g) + 1)))
    ax.barh(g["group"], g["share"] * 100, color=C_TRAIN, height=0.6)
    for i, r in enumerate(g.itertuples()):
        ax.text(r.share * 100, i, f" {r.share:.1%} ({r.n_features} biến)", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Tỷ trọng importance (%)")
    ax.set_title("Feature importance theo group (thư mục S3) — CatBoost")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, g["share"].max() * 130)
    _save(fig, out / "feature_importance_by_group.png")
    return fi, gi


def shap_report(model: CatBoostClassifier, X: pd.DataFrame, features: list[str], cats: list[str],
                fi: pd.DataFrame, top_n: int, sample: int, seed: int, out: Path) -> pd.DataFrame:
    Xs = X[features].sample(min(len(X), sample), random_state=seed)
    sv = model.get_feature_importance(Pool(Xs, cat_features=cats), type="ShapValues")[:, :-1]
    top = fi["feature"].head(top_n).tolist()
    idx = [features.index(f) for f in top]
    Xplot = Xs[top].copy()
    for c in top:  # categorical -> mã số để tô màu
        if c in cats:
            Xplot[c] = Xplot[c].astype("category").cat.codes
    t = pd.DataFrame({"feature": top, "mean_abs_shap": np.abs(sv[:, idx]).mean(axis=0)})
    t.to_csv(out / "shap_top10.csv", index=False)
    plt.figure()
    shap.summary_plot(sv[:, idx], Xplot, feature_names=top, show=False, max_display=top_n, plot_size=(8, 5))
    plt.title(f"SHAP — top {top_n} feature theo importance CatBoost (mẫu {len(Xs):,} shop test)", color=INK)
    plt.savefig(out / "shap_top10.png", bbox_inches="tight")
    plt.close("all")
    return t
