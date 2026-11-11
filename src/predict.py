"""
predict.py - loads the trained models and turns inputs into forecasts.
Used by both the app and the daily TMDB update.
"""
import json
import math
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import POST_FEATURES, PRE_FEATURES, post_feature_row, pre_model_frame

ROOT = Path(__file__).resolve().parents[1]
MODELS, DATA = ROOT / "models", ROOT / "data"
BREAK_EVEN_MULTIPLE = 2.5   # rule of thumb: worldwide gross ~2.5x budget to break even


@lru_cache(maxsize=None)
def metrics():
    return json.loads((MODELS / "metrics.json").read_text())


@lru_cache(maxsize=None)
def _booster(name):
    return lgb.Booster(model_file=str(MODELS / f"{name}.txt"))


@lru_cache(maxsize=None)
def people_power():
    p = pd.read_csv(DATA / "people_power.csv")
    return {(r.role, r.name): (r.mean, r.count) for r in p.itertuples()}


def person_power(name, role):
    """Average log-gross and number of past films for a star or director."""
    return people_power().get((role, name), (np.nan, 0))


def _quantiles(prefix, X, widen):
    q = np.column_stack([_booster(f"{prefix}_{n}").predict(X)
                         for n in ("low", "mid", "high")])
    q = np.sort(q, axis=1)
    q[:, 0] -= widen
    q[:, 2] += widen
    return q


def predict_pre(df):
    """Worldwide lifetime gross (USD) for each row of PRE_FEATURES."""
    widen = metrics()["pre_release"]["range_widen_log"]
    q = np.exp(_quantiles("pre", pre_model_frame(df), widen))
    return pd.DataFrame(q, columns=["pred_low", "pred_mid", "pred_high"], index=df.index)


def predict_pre_all(df):
    """Worldwide total, US opening weekend and US total for each row."""
    out = predict_pre(df)
    us = metrics()["pre_release_us"]
    X = pre_model_frame(df)
    for tag, key in (("open", "opening_weekend"), ("dom", "domestic_total")):
        q = np.exp(_quantiles(f"pre_{tag}", X, us[key]["range_widen_log"]))
        for i, n in enumerate(("low", "mid", "high")):
            out[f"{tag}_{n}"] = q[:, i]
    # US numbers can't logically exceed worldwide, nor opening exceed US total
    out["dom_mid"] = np.minimum(out["dom_mid"], out["pred_mid"])
    out["open_mid"] = np.minimum(out["open_mid"], out["dom_mid"])
    return out


def break_even_chance(pred_low, pred_mid, pred_high, budget):
    """Probability worldwide gross reaches 2.5x budget, assuming the forecast
    range is a log-normal 80% interval."""
    if not budget or pd.isna(budget) or budget <= 0:
        return None
    sigma = (math.log(pred_high) - math.log(pred_low)) / (2 * 1.2816)
    if sigma <= 0:
        return None
    z = (math.log(BREAK_EVEN_MULTIPLE * budget) - math.log(pred_mid)) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2))


DRIVER_GROUPS = {
    "Budget": ["log_budget"],
    "Lead star's track record": ["star_power", "star_films"],
    "Director's track record": ["director_power", "director_films"],
    "Franchise / sequel": ["is_sequel"],
    "Genre": ["genre"],
    "Release timing": ["month"],
    "Release year": ["year"],
    "Runtime": ["runtime"],
    "Age rating": ["rating"],
    "Studio": ["studio"],
}


def explain(df_row):
    """Why the worldwide forecast is what it is. Returns the typical film's gross
    and a list of (driver, % effect) sorted by size of effect."""
    contrib = _booster("pre_mid").predict(pre_model_frame(df_row), pred_contrib=True)[0]
    by_feature = dict(zip(PRE_FEATURES, contrib[:-1]))
    baseline = math.exp(contrib[-1])
    drivers = [(name, sum(by_feature[f] for f in feats))
               for name, feats in DRIVER_GROUPS.items()]
    drivers = [(n, math.exp(v) - 1, v) for n, v in drivers if abs(v) > 0.03]
    drivers.sort(key=lambda d: abs(d[2]), reverse=True)
    return baseline, drivers


def predict_post(w1, theaters, release_date, w2=None, w3=None, major=1):
    """Domestic lifetime forecast from opening (and later) weekend grosses."""
    k = 3 if (w2 and w3) else 2 if w2 else 1
    row = post_feature_row(w1, theaters, release_date, w2, w3, major)
    widen = metrics()["post_release"][f"after_weekend_{k}"]["range_widen_log"]
    q = np.exp(_quantiles(f"post_k{k}", row[POST_FEATURES[k]], widen))[0] * w1
    floor = w1 + (w2 or 0) + (w3 or 0)   # can't finish below what it has earned
    low, mid, high = np.maximum(q, floor)
    return {"k": k, "low": low, "mid": mid, "high": high, "multiplier": mid / w1}
