"""
train_models.py - trains the forecasting models and measures their accuracy.

    python src/train_models.py

Method in one line: gradient-boosted decision trees (LightGBM, a close cousin
of XGBoost) trained with QUANTILE loss, so each model gives a low (10th
percentile), likely (median) and high (90th percentile) estimate.

Honest testing: we train on older films and test on newer ones the model has
never seen (a "time-based split"), exactly like forecasting real releases.
Final models are then retrained on all the data.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from features import POST_FEATURES, PRE_FEATURES, pre_model_frame

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS = ROOT / "data", ROOT / "models"
MODELS.mkdir(exist_ok=True)
QUANTILES = {"low": 0.1, "mid": 0.5, "high": 0.9}

PARAMS = dict(objective="quantile", learning_rate=0.03, n_estimators=500,
              num_leaves=15, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def fit_quantiles(X, y):
    return {name: lgb.LGBMRegressor(alpha=a, **PARAMS).fit(X, y)
            for name, a in QUANTILES.items()}


def predict_quantiles(models, X, widen=0.0):
    preds = np.column_stack([models[n].predict(X) for n in ("low", "mid", "high")])
    preds = np.sort(preds, axis=1)          # keep low <= mid <= high
    preds[:, 0] -= widen                    # calibrated widening (see below)
    preds[:, 2] += widen
    return preds


def calibration_widen(X_fit, y_fit, X_cal, y_cal, target=0.8):
    """Conformal calibration: raw quantile models are usually over-confident.
    We hold back the most recent slice of the training data, see how far
    actual results fell outside the predicted range, and widen the range just
    enough that ~80% of real outcomes land inside it."""
    m = fit_quantiles(X_fit, y_fit)
    p = predict_quantiles(m, X_cal)
    miss = np.maximum(p[:, 0] - y_cal, y_cal - p[:, 2])
    n = len(miss)
    return float(np.quantile(miss, min(1.0, np.ceil((n + 1) * target) / n)))


def score(actual, low, mid, high):
    """Accuracy numbers, all in plain money terms."""
    ape = np.abs(mid - actual) / actual
    return {
        "films_tested": int(len(actual)),
        "median_abs_pct_error": round(float(np.median(ape)) * 100, 1),
        "within_25pct": round(float(np.mean(ape <= 0.25)) * 100, 1),
        "within_50pct": round(float(np.mean(ape <= 0.50)) * 100, 1),
        "range_coverage": round(float(np.mean((actual >= low) & (actual <= high))) * 100, 1),
    }


def train_pre():
    df = pd.read_csv(DATA / "pre_release.csv", parse_dates=["release_date"])
    train, test = df[df["year"] <= 2015], df[df["year"] >= 2016]
    fit_, cal_ = train[train["year"] <= 2011], train[train["year"] >= 2012]
    widen = calibration_widen(pre_model_frame(fit_), fit_["log_gross"],
                              pre_model_frame(cal_), cal_["log_gross"])

    models = fit_quantiles(pre_model_frame(train), train["log_gross"])
    q = np.exp(predict_quantiles(models, pre_model_frame(test), widen))
    metrics = score(test["gross"].values, *q.T)

    # Baseline to beat: a straight line on budget alone
    tb, sb = train.dropna(subset=["log_budget"]), test.dropna(subset=["log_budget"])
    lr = LinearRegression().fit(tb[["log_budget"]], tb["log_gross"])
    base = np.exp(lr.predict(sb[["log_budget"]]))
    ours = q[test["log_budget"].notna().values, 1]
    metrics["baseline_budget_only_median_abs_pct_error"] = round(
        float(np.median(np.abs(base - sb["gross"]) / sb["gross"])) * 100, 1)
    metrics["ours_same_films_median_abs_pct_error"] = round(
        float(np.median(np.abs(ours - sb["gross"]) / sb["gross"])) * 100, 1)

    out = test[["name", "year", "genre", "budget", "gross"]].copy()
    out["pred_low"], out["pred_mid"], out["pred_high"] = q.T
    out.to_csv(DATA / "test_predictions_pre.csv", index=False)

    importance = pd.Series(models["mid"].booster_.feature_importance("gain"),
                           index=PRE_FEATURES).sort_values(ascending=False)

    final = fit_quantiles(pre_model_frame(df), df["log_gross"])
    for name, m in final.items():
        m.booster_.save_model(str(MODELS / f"pre_{name}.txt"))
    metrics["range_widen_log"] = round(widen, 4)
    return metrics, (importance / importance.sum()).round(4).to_dict()


def train_pre_us():
    """Pre-release forecasts of US opening weekend and US (domestic) total."""
    df = pd.read_csv(DATA / "pre_us.csv", parse_dates=["release_date"])
    out = {}
    for tag, target, money in (("open", "log_w1", "w1"), ("dom", "log_dom", "lifetime")):
        train, test = df[df["year"] <= 2015], df[df["year"] >= 2016]
        fit_, cal_ = train[train["year"] <= 2011], train[train["year"] >= 2012]
        widen = calibration_widen(pre_model_frame(fit_), fit_[target],
                                  pre_model_frame(cal_), cal_[target])
        models = fit_quantiles(pre_model_frame(train), train[target])
        q = np.exp(predict_quantiles(models, pre_model_frame(test), widen))
        m = score(test[money].values, *q.T)
        m["range_widen_log"] = round(widen, 4)
        out[tag] = m
        final = fit_quantiles(pre_model_frame(df), df[target])
        for name, mdl in final.items():
            mdl.booster_.save_model(str(MODELS / f"pre_{tag}_{name}.txt"))
    return out


def train_post():
    df = pd.read_csv(DATA / "post_release.csv", parse_dates=["release_date"])
    df = df[df["multiplier"] <= 20]                    # drop extreme oddities
    df["y"] = np.log(df["multiplier"])
    split = pd.Timestamp("2022-07-01")
    results, importances, test_frames = {}, {}, []

    for k, feats in POST_FEATURES.items():
        d = df.copy()
        if k >= 2:
            d = d[d["drop2"].notna()]
        if k >= 3:
            d = d[d["drop3"].notna()]
        train, test = d[d["release_date"] < split], d[d["release_date"] >= split]
        cal_split = pd.Timestamp("2018-07-01")
        fit_, cal_ = train[train["release_date"] < cal_split], train[train["release_date"] >= cal_split]
        widen = calibration_widen(fit_[feats], fit_["y"], cal_[feats], cal_["y"])

        models = fit_quantiles(train[feats], train["y"])
        q = np.exp(predict_quantiles(models, test[feats], widen)) * test["w1"].values[:, None]
        m = score(test["lifetime"].values, *q.T)
        m["range_widen_log"] = round(widen, 4)
        base = np.exp(train["y"].median()) * test["w1"]
        m["baseline_typical_multiplier_median_abs_pct_error"] = round(
            float(np.median(np.abs(base - test["lifetime"]) / test["lifetime"])) * 100, 1)
        results[f"after_weekend_{k}"] = m

        t = test[["film_id", "title", "release_date", "w1", "w2", "w3", "lifetime"]].copy()
        t["k"] = k
        t["pred_low"], t["pred_mid"], t["pred_high"] = q.T
        test_frames.append(t)

        imp = pd.Series(models["mid"].booster_.feature_importance("gain"), index=feats)
        importances[f"after_weekend_{k}"] = (imp / imp.sum()).round(4).sort_values(ascending=False).to_dict()

        final = fit_quantiles(d[feats], d["y"])
        for name, mdl in final.items():
            mdl.booster_.save_model(str(MODELS / f"post_k{k}_{name}.txt"))

    pd.concat(test_frames).to_csv(DATA / "test_predictions_post.csv", index=False)
    return results, importances


if __name__ == "__main__":
    pre_metrics, pre_imp = train_pre()
    post_metrics, post_imp = train_post()
    us_metrics = train_pre_us()
    report = {
        "pre_release": {"test_period": "films released 2016-2019 (trained on 1990-2015)",
                        "target": "worldwide lifetime gross", **pre_metrics},
        "pre_release_us": {"test_period": "films released 2016-2019 (trained on 2000-2015)",
                           "opening_weekend": us_metrics["open"], "domestic_total": us_metrics["dom"]},
        "post_release": {"test_period": "films released Jul 2022 - Aug 2024 (trained on 2000-Jun 2022)",
                         "target": "domestic (US+Canada) lifetime gross", **post_metrics},
        "importance": {"pre_release": pre_imp, "post_release": post_imp},
    }
    (MODELS / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "importance"}, indent=2))
