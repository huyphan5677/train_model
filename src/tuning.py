"""Tune CatBoost bằng Optuna, tối ưu AUC trung bình trên CV theo nhóm cus_id trong tập train."""
from __future__ import annotations

import optuna
import pandas as pd

from src.features import cv_catboost, make_folds


def tune(X: pd.DataFrame, y: pd.Series, groups: pd.Series, features: list[str], cfg: dict) -> dict:
    oc, seed, threads = cfg["optuna"], cfg["seed"], cfg["thread_count"]
    folds = make_folds(y, groups, oc["cv_folds"], seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": oc["max_iterations"],
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "border_count": trial.suggest_categorical("border_count", [64, 128, 254]),
        }
        res = cv_catboost(X, y, folds, features, params, oc["early_stopping_rounds"], threads, seed)
        trial.set_user_attr("auc_folds", res["auc_folds"])
        trial.set_user_attr("best_iterations", res["best_iterations"])
        return res["auc"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=oc["n_trials"], timeout=oc["timeout_sec"],
                   callbacks=[lambda s, t: print(f"[optuna] trial {t.number:3d} auc={t.value:.4f} best={s.best_value:.4f}", flush=True)])

    best = study.best_trial
    iters = best.user_attrs["best_iterations"]
    params = {**best.params, "iterations": int(round(sum(iters) / len(iters) * 1.1))}
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration"))

    # OOF của tham số tốt nhất (đánh giá train theo tháng mà không dùng dự báo in-sample)
    res = cv_catboost(X, y, folds, features, {**best.params, "iterations": oc["max_iterations"]},
                      oc["early_stopping_rounds"], threads, seed)
    return {"params": params, "cv_auc": best.value, "trials": trials, "study": study, "oof": res["oof"]}
