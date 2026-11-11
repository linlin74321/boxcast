"""
train_models.py - trains the forecasting models and measures their accuracy.

    python src/train_models.py

Method in one line: gradient-boosted decision trees (LightGBM, a close cousin
of XGBoost) trained with QUANTILE loss, so each model gives a low (10th
percentile), likely (median) and high (90th percentile) estimate. Ranges are
then conformally calibrated so they contain the real result ~80% of the time.

Honest testing: we train on older films and test on newer ones the model has
never seen (a "time-based split"). When recent films are available (after the
Retrain workflow has fetched them from TMDB) the test is on 2022+ releases -
today's post-COVID market. Final models are then retrained on all the data.

Safety check: new pre-release models only replace the current ones if they
are at least as accurate on the same unseen films.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from buzz import HORIZONS
from features import (POST_FEATURES, PRE_FEATURES, PRE_FEATURES_FRANCHISE,
                      PRE_FEATURES_SIGNALS, pre_model_frame)

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS = ROOT / "data", ROOT / "models"
MODELS.mkdir(exist_ok=True)
QUANTILES = {"low": 0.1, "mid": 0.5, "high": 0.9}

PARAMS = dict(objective="quantile", learning_rate=0.03, n_estimators=500,
              num_leaves=15, min_child_samples=20, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def fit_quantiles(X, y, weights=None):
    return {name: lgb.LGBMRegressor(alpha=a, **PARAMS).fit(X, y, sample_weight=weights)
            for name, a in QUANTILES.items()}


def with_buzz(df, horizon=None):
    """Attach the Wikipedia-buzz inputs.
    horizon=None: films with buzz data get one training row per distance from
    release (7/30/60/90 days, each weighted 1/4), so the model learns how buzz
    builds up. horizon=30: one row per film, buzz measured 30 days out (used
    for testing and calibration)."""
    cols = [f"wiki_{h}" for h in HORIZONS]
    has = df[cols].notna().any(axis=1) if set(cols) <= set(df.columns) else pd.Series(False, index=df.index)
    def at(sub, h, weight):
        sub = sub.copy()
        sub["log_wiki_views"] = np.log1p(sub[f"wiki_{h}"]) if f"wiki_{h}" in sub else np.nan
        sub["wiki_days_before"] = np.where(sub["log_wiki_views"].notna(), h, np.nan)
        sub["w"] = weight
        return sub
    if horizon is not None:
        return at(df, horizon, 1.0)
    parts = [df[~has].assign(log_wiki_views=np.nan, wiki_days_before=np.nan, w=1.0)]
    parts += [at(df[has], h, 1 / len(HORIZONS)) for h in HORIZONS]
    return pd.concat(parts, ignore_index=True)


def predict_quantiles(models, X, widen=0.0):
    preds = np.column_stack([models[n].predict(X) for n in ("low", "mid", "high")])
    preds = np.sort(preds, axis=1)          # keep low <= mid <= high
    preds[:, 0] -= widen                    # calibrated widening (see below)
    preds[:, 2] += widen
    return preds


def calibration_widen(X_fit, y_fit, X_cal, y_cal, target=0.8, weights=None):
    """Conformal calibration: raw quantile models are usually over-confident.
    We hold back the most recent slice of the training data, see how far
    actual results fell outside the predicted range, and widen the range just
    enough that ~80% of real outcomes land inside it."""
    m = fit_quantiles(X_fit, y_fit, weights)
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


def pre_split(df):
    """Train / calibration / test split for the pre-release models."""
    if (df["year"] >= 2022).sum() >= 100:
        test = df[df["year"] >= 2022]
        train = df[df["year"] <= 2021]
        period = "films released 2022 onwards (trained on 1990-2021)"
    else:
        test = df[df["year"] >= 2016]
        train = df[df["year"] <= 2015]
        period = "films released 2016-2019 (trained on 1990-2015)"
    cal_start = int(train["year"].max()) - 3 if train["year"].max() < 2020 else 2016
    fit_, cal_ = train[train["year"] < cal_start], train[train["year"] >= cal_start]
    return train, test, fit_, cal_, period


def old_model_error(prefix, test, target_money):
    """How the CURRENTLY saved model does on the same unseen films (for the
    safety check). None if it can't be compared fairly."""
    try:
        old = json.loads((MODELS / "metrics.json").read_text())
        trained_through = old.get("trained_through_year", 2019)
        if trained_through >= test["year"].min():
            return None                         # it has already seen these films
        b = lgb.Booster(model_file=str(MODELS / f"{prefix}_mid.txt"))
        pred = np.exp(b.predict(pre_model_frame(test, b.feature_name())))
        return float(np.median(np.abs(pred - test[target_money]) / test[target_money]) * 100)
    except Exception:
        return None


def prepared(df, features):
    """(training rows, single-row-per-film evaluation rows) for a feature set."""
    if "log_wiki_views" in features:
        return with_buzz(df), with_buzz(df, 30)
    return df.assign(w=1.0), df.assign(w=1.0)


def fit_eval(train, test, fit_, cal_, features, target, money):
    """Train on `train`, calibrate on fit_/cal_, score on `test`."""
    tr, _ = prepared(train, features)
    fi, _ = prepared(fit_, features)
    _, ca = prepared(cal_, features)
    _, te = prepared(test, features)
    widen = calibration_widen(pre_model_frame(fi, features), fi[target],
                              pre_model_frame(ca, features), ca[target], weights=fi["w"])
    models = fit_quantiles(pre_model_frame(tr, features), tr[target], tr["w"])
    q = np.exp(predict_quantiles(models, pre_model_frame(te, features), widen))
    m = score(te[money].values, *q.T)
    m["range_widen_log"] = round(widen, 4)
    return m, models, q, te


def train_pre(features):
    df = pd.read_csv(DATA / "pre_release.csv", parse_dates=["release_date"])
    train, test, fit_, cal_, period = pre_split(df)
    metrics, models, q, te = fit_eval(train, test, fit_, cal_, features, "log_gross", "gross")
    metrics["test_period"] = period

    # Baseline to beat: a straight line on budget alone
    tb, sb = train.dropna(subset=["log_budget"]), test.dropna(subset=["log_budget"])
    lr = LinearRegression().fit(tb[["log_budget"]], tb["log_gross"])
    base = np.exp(lr.predict(sb[["log_budget"]]))
    metrics["baseline_budget_only_median_abs_pct_error"] = round(
        float(np.median(np.abs(base - sb["gross"]) / sb["gross"])) * 100, 1)
    seq = te["log_prev_gross"].notna().values if "log_prev_gross" in te else np.zeros(len(te), bool)
    if seq.sum() >= 10:
        metrics["sequels_median_abs_pct_error"] = round(float(np.median(
            np.abs(q[seq, 1] - te["gross"].values[seq]) / te["gross"].values[seq])) * 100, 1)
    if "log_wiki_views" in features:
        # How accuracy changes with distance from release, for films with buzz data
        by_h = {}
        for h in HORIZONS:
            th = with_buzz(test, h)
            th = th[th["log_wiki_views"].notna()]
            if len(th) >= 20:
                p = np.exp(models["mid"].predict(pre_model_frame(th, features)))
                by_h[f"{h}_days_before"] = round(float(np.median(np.abs(p - th["gross"]) / th["gross"])) * 100, 1)
        metrics["buzz_films_error_by_days_before_release"] = by_h

    out = te[["name", "year", "genre", "budget", "gross"]].copy()
    out["pred_low"], out["pred_mid"], out["pred_high"] = q.T
    importance = pd.Series(models["mid"].booster_.feature_importance("gain"), index=features)
    importance = (importance / importance.sum()).round(4).sort_values(ascending=False).to_dict()

    old_err = old_model_error("pre", test, "gross")
    full, _ = prepared(df, features)
    final = fit_quantiles(pre_model_frame(full, features), full["log_gross"], full["w"])
    return {"metrics": metrics, "importance": importance, "test_predictions": out,
            "final": final, "old_error": old_err, "trained_through": int(df["year"].max())}


def choose_features():
    """Fair comparison on the same unseen films: are the new market signals
    (China release, competition, Wikipedia buzz) worth adding?"""
    df = pd.read_csv(DATA / "pre_release.csv", parse_dates=["release_date"])
    has_franchise = df.get("log_prev_gross", pd.Series(dtype=float)).notna().sum() >= 100
    basic = PRE_FEATURES_FRANCHISE if has_franchise else PRE_FEATURES
    has_signals = (df.get("china_release", pd.Series(dtype=float)).notna().sum() >= 100
                   or df.get("wiki_30", pd.Series(dtype=float)).notna().sum() >= 100)
    signals = PRE_FEATURES_SIGNALS if has_signals else basic + ["competition_big"]
    train, test, fit_, cal_, _ = pre_split(df)
    without, *_ = fit_eval(train, test, fit_, cal_, basic, "log_gross", "gross")
    with_, *_ = fit_eval(train, test, fit_, cal_, signals, "log_gross", "gross")
    a, b = without["median_abs_pct_error"], with_["median_abs_pct_error"]
    chosen = signals if b <= a - 0.3 else basic
    comparison = {"without_new_signals": a, "with_new_signals": b,
                  "signals_tested": [f for f in signals if f not in basic],
                  "chosen": "with new signals" if chosen is signals else "without new signals"}
    print(f"Feature comparison on the same unseen films: without {a}% vs with {b}% "
          f"-> using {comparison['chosen']}")
    return chosen, comparison


def train_pre_us(features):
    """Pre-release forecasts of US opening weekend and US (domestic) total."""
    df = pd.read_csv(DATA / "pre_us.csv", parse_dates=["release_date"])
    out, finals = {}, {}
    train, test, fit_, cal_, period = pre_split(df)
    full, _ = prepared(df, features)
    for tag, target, money in (("open", "log_w1", "w1"), ("dom", "log_dom", "lifetime")):
        m, *_ = fit_eval(train, test, fit_, cal_, features, target, money)
        out[tag] = m
        finals[tag] = fit_quantiles(pre_model_frame(full, features), full[target], full["w"])
    out["test_period"] = period
    return out, finals


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


def eval_in_release():
    """Accuracy of projecting a film's final total from its earnings so far,
    using the typical earning curve of older films (tested on Jul 2022 - Aug 2024)."""
    c = pd.read_csv(DATA / "curves.csv")
    p = pd.read_csv(DATA / "post_release.csv", parse_dates=["release_date"])
    d = p.merge(c, on="film_id")
    d["open_share"] = d["w1"] / d["lifetime"]
    split = pd.Timestamp("2022-07-01")
    train, test = d[d["release_date"] < split], d[d["release_date"] >= split]
    errs = {}
    for label, col in (("after_opening_weekend", "open_share"), ("after_week_1", "wk1"),
                       ("after_week_2", "wk2"), ("after_week_3", "wk3")):
        errs[label] = round(float(np.median(np.abs(test[col] / train[col].median() - 1))) * 100, 1)
    return {"test_period": "Jul 2022 - Aug 2024", "films_tested": int(len(test)),
            "median_abs_pct_error": errs}


if __name__ == "__main__":
    pre_df = pd.read_csv(DATA / "pre_release.csv")
    print(f"Training data: {len(pre_df):,} films up to {pre_df['year'].max()}")
    features, comparison = choose_features()

    pre = train_pre(features)
    us_metrics, us_finals = train_pre_us(features)
    post_metrics, post_imp = train_post()

    new_err, old_err = pre["metrics"]["median_abs_pct_error"], pre["old_error"]
    keep_old = old_err is not None and new_err > old_err + 0.5
    previous = json.loads((MODELS / "metrics.json").read_text()) if (MODELS / "metrics.json").exists() else {}

    if keep_old:
        print(f"SAFETY CHECK: new pre-release model error {new_err}% is worse than the current "
              f"model's {old_err:.1f}% on the same films. Keeping the current pre-release models.")
        report = previous
    else:
        if old_err is not None:
            print(f"Safety check passed: new {new_err}% vs current {old_err:.1f}% on the same films.")
        for name, m in pre["final"].items():
            m.booster_.save_model(str(MODELS / f"pre_{name}.txt"))
        for tag, models in us_finals.items():
            for name, m in models.items():
                m.booster_.save_model(str(MODELS / f"pre_{tag}_{name}.txt"))
        pre["test_predictions"].to_csv(DATA / "test_predictions_pre.csv", index=False)
        report = {
            "trained_through_year": pre["trained_through"],
            "pre_release": {"target": "worldwide lifetime gross", **pre["metrics"],
                            "feature_comparison": comparison},
            "pre_release_us": {"test_period": us_metrics["test_period"],
                               "opening_weekend": us_metrics["open"],
                               "domestic_total": us_metrics["dom"]},
            "importance": {"pre_release": pre["importance"]},
        }
    report["post_release"] = {"test_period": "films released Jul 2022 - Aug 2024 (trained on 2000-Jun 2022)",
                              "target": "domestic (US+Canada) lifetime gross", **post_metrics}
    report["in_release_projection"] = eval_in_release()
    report.setdefault("importance", {})["post_release"] = post_imp
    (MODELS / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("importance",)}, indent=2))
