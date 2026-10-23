"""Biểu đồ/bảng độ ổn định qua các cửa sổ trượt cho reliable score."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, roc_curve

from src.evaluate import C_TEST, C_TRAIN, GRID, INK, INK2, _save, month_label

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
C_PARTIAL = "#f6b49a"   # cam nhạt: tháng test nhãn tạm


def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def metrics(y: pd.Series, p: pd.Series) -> dict:
    if y.nunique() < 2:
        return {"n": len(y), "n_bad": int(y.sum()), "bad_rate": float(y.mean()), "auc": np.nan, "gini": np.nan, "ks": np.nan}
    auc = roc_auc_score(y, p)
    return {"n": len(y), "n_bad": int(y.sum()), "bad_rate": float(y.mean()), "auc": auc, "gini": 2 * auc - 1, "ks": ks(y, p)}


def label_rate_chart(cohorts: pd.DataFrame, out: Path):
    """cohorts: date, n, n_bad, bad_rate, label_status, n_s0, bad_rate_s0."""
    c = cohorts[cohorts["label_status"] != "none"].sort_values("date")
    x = np.arange(len(c))
    colors = [C_TRAIN if s == "full" else C_PARTIAL for s in c["label_status"]]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    a1.bar(x, c["bad_rate"] * 100, color=colors, width=0.55)
    for i, r in enumerate(c.itertuples()):
        a1.text(i, r.bad_rate * 100, f"{r.bad_rate:.1%}", ha="center", va="bottom", fontsize=9, color=INK)
    a1.set_ylim(0, c["bad_rate"].max() * 100 * 1.2)
    a1.set_ylabel("Tỷ lệ bad (%)")
    a1.set_title("Tỷ lệ bad theo cohort — prediction set")
    a2.bar(x, c["n"] / 1000, color=colors, width=0.55)
    a2.set_ylabel("Số shop (nghìn)")
    a2.set_xticks(x, [month_label(d) for d in c["date"]])
    a1.legend(handles=[Patch(color=C_TRAIN), Patch(color=C_PARTIAL)], labels=["nhãn đủ", "nhãn tạm"],
              frameon=False, loc="upper left", ncols=2)
    _save(fig, out / "label_rate_by_cohort.png")


def window_metrics_chart(wm: pd.DataFrame, out: Path):
    """wm: một dòng/cửa sổ: test, test_label, auc, gini, ks, auc_s0, gini_s0."""
    w = wm.sort_values("test")
    x = np.arange(len(w))
    full = w["test_label"] == "full"
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for ax, col, name in [(axes[0], "auc", "AUC"), (axes[1], "gini", "Gini"), (axes[2], "ks", "KS")]:
        for j, (sub, color, lab) in enumerate([("", SERIES[0], "prediction set"), ("_s0", SERIES[1], "tập khó S0")]):
            vals = w[col + sub]
            xs = x + (j - 0.5) * 0.36
            bars = ax.bar(xs, vals, width=0.34, color=color, label=lab)
            for b, is_full in zip(bars, full):
                if not is_full:
                    b.set_alpha(0.45)
                    b.set_hatch("///")
            for xi, v in zip(xs, vals):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK)
        m = w.loc[full, col]
        if len(m):
            ax.axhline(m.mean(), color=INK2, ls="--", lw=1)
            ax.text(len(w) - 0.5, m.mean(), f"TB {m.mean():.3f} ± {m.std(ddof=0):.3f}", ha="right", va="bottom", fontsize=8, color=INK2)
        ax.set_xticks(x, [month_label(d) + ("\n(nhãn tạm)" if not f else "") for d, f in zip(w["test"], full)])
        ax.set_title(f"{name} tháng test theo cửa sổ")
        lo = 0.5 if col == "auc" else 0
        ax.set_ylim(lo, min(1, max(w[col].max(), w[col + "_s0"].max()) * 1.12))
    axes[0].legend(frameon=False, loc="lower left")
    _save(fig, out / "stability_auc_gini_ks.png")


def decile_chart(deciles: pd.DataFrame, out: Path):
    """deciles: test, bin, bad_rate (bin 1 = 10% rủi ro cao nhất)."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, (t, g) in enumerate(deciles.groupby("test")):
        partial = g["test_label"].iloc[0] != "full"
        ax.plot(g["bin"], g["precision"] * 100, color=SERIES[i % len(SERIES)], lw=2, marker="o", ms=4,
                ls="--" if partial else "-", label=month_label(t) + (" (nhãn tạm)" if partial else ""))
    ax.set_xticks(range(1, deciles["bin"].max() + 1))
    ax.set_xlabel("Bin rủi ro (1 = 10% điểm rủi ro cao nhất)")
    ax.set_ylabel("Tỷ lệ bad (%)")
    ax.set_title("Tỷ lệ bad theo decile — mỗi đường là một tháng test")
    ax.legend(frameon=False, title="Tháng test")
    _save(fig, out / "stability_decile_bad_rate.png")


def psi_chart(wm: pd.DataFrame, out: Path):
    w = wm.sort_values("test")
    x = np.arange(len(w))
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(x, w["psi_score"], color=[SERIES[0] if s == "full" else C_PARTIAL for s in w["test_label"]], width=0.5)
    for i, v in enumerate(w["psi_score"]):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.axhline(0.1, color=INK2, ls="--", lw=1)
    ax.text(len(w) - 0.5, 0.1, "0,1", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_ylim(0, max(0.12, w["psi_score"].max() * 1.25))
    ax.set_xticks(x, [month_label(d) for d in w["test"]])
    ax.set_title("PSI score: khung train vs tháng test, theo cửa sổ")
    _save(fig, out / "stability_psi_score.png")


def group_importance_heatmap(gi: pd.DataFrame, out: Path):
    """gi: test, group, share."""
    p = gi.pivot_table(index="group", columns="test", values="share", fill_value=0)
    p = p.loc[p.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(1.6 * p.shape[1] + 3, 0.42 * p.shape[0] + 1.4))
    ax.imshow(p.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=max(0.01, p.to_numpy().max()))
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.iat[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if v > p.to_numpy().max() * 0.6 else INK)
    ax.set_xticks(range(p.shape[1]), [f"test {month_label(c)}" for c in p.columns])
    ax.set_yticks(range(p.shape[0]), p.index)
    ax.grid(False)
    ax.set_title("Tỷ trọng importance theo group, từng cửa sổ")
    _save(fig, out / "stability_importance_by_group.png")
