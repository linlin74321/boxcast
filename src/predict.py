"""
predict.py - loads the trained models and turns inputs into forecasts.
Used by both the app and the daily TMDB update.
"""
import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from features import POST_FEATURES, post_feature_row, pre_model_frame

ROOT = Path(__file__).resolve().parents[1]
MODELS, DATA = ROOT / "models", ROOT / "data"


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
    """df needs the PRE_FEATURES columns. Returns worldwide gross in USD."""
    widen = metrics()["pre_release"]["range_widen_log"]
    q = np.exp(_quantiles("pre", pre_model_frame(df), widen))
    return pd.DataFrame(q, columns=["pred_low", "pred_mid", "pred_high"], index=df.index)


def predict_post(w1, theaters, release_date, w2=None, w3=None, major=1):
    """Domestic lifetime forecast from opening (and later) weekend grosses."""
    k = 3 if (w2 and w3) else 2 if w2 else 1
    row = post_feature_row(w1, theaters, release_date, w2, w3, major)
    widen = metrics()["post_release"][f"after_weekend_{k}"]["range_widen_log"]
    q = np.exp(_quantiles(f"post_k{k}", row[POST_FEATURES[k]], widen))[0] * w1
    # Lifetime can never be less than what the film has already earned
    floor = w1 + (w2 or 0) + (w3 or 0)
    low, mid, high = np.maximum(q, floor)
    return {"k": k, "low": low, "mid": mid, "high": high,
            "multiplier": mid / w1}
