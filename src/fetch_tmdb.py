"""
fetch_tmdb.py - the DAILY AUTOMATIC UPDATE (run by GitHub Actions).

  1. asks TMDB for films now playing and coming soon in US cinemas,
  2. forecasts each one,
  3. saves data/live_forecasts.csv (the "In cinemas & coming soon" strip in the
     app) and appends to data/forecast_history.csv (your track record).
"""
import math
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import apply_sequel_anchor, predict_pre_all, project_in_release  # noqa: E402
from tmdb import TMDB, film_features  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def run(key):
    api = TMDB(key)
    films = {}
    for status, which in (("In cinemas", "now_playing"), ("Coming soon", "upcoming")):
        for f in api.listing(which, pages=4):
            films.setdefault(f["id"], status)

    rows, prevs = [], {}
    for tmdb_id, status in films.items():
        try:
            d = api.details(tmdb_id)
            rows.append(film_features(d, status))
        except requests.HTTPError as e:
            print(f"skipped {tmdb_id}: {e}")
            continue
        try:                    # franchise lookup is a bonus; never let it stop the run
            prev = api.previous_installment(d)
            if prev:
                prevs[tmdb_id] = prev["revenue"]
                rows[-1].update(log_prev_gross=math.log(prev["revenue"]),
                                franchise_n=prev.get("count", 1), is_sequel=1)
        except Exception as e:
            print(f"no franchise info for {tmdb_id}: {e}")
        time.sleep(0.05)

    live = pd.DataFrame(rows).dropna(subset=["release_date"]).reset_index(drop=True)
    preds = predict_pre_all(live)
    # Same logic as the film page: lean sequels towards their franchise, and
    # for films already earning, project the final total from earnings so far.
    today = pd.Timestamp(date.today())
    labels = []
    for i, r in live.iterrows():
        if r["tmdb_id"] in prevs:
            preds.loc[i] = apply_sequel_anchor(preds.loc[i], prevs[r["tmdb_id"]])
        days_out = (today - pd.Timestamp(r["release_date"])).days
        if pd.notna(r["revenue"]) and r["revenue"] > 0 and days_out >= 0:
            proj = project_in_release(r["revenue"], days_out)
            preds.loc[i, ["pred_low", "pred_mid", "pred_high"]] = [proj["low"], proj["mid"], proj["high"]]
            labels.append("projected")
        else:
            labels.append("forecast")
    live = live.join(preds)
    live["fig_label"] = labels
    live["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    live = live.sort_values("popularity", ascending=False)
    live.to_csv(DATA / "live_forecasts.csv", index=False)

    hist_path = DATA / "forecast_history.csv"
    today = live[["tmdb_id", "title", "release_date", "pred_low", "pred_mid", "pred_high"]].copy()
    today.insert(0, "forecast_date", date.today().isoformat())
    if hist_path.exists():
        hist = pd.concat([pd.read_csv(hist_path), today])
        hist = hist.drop_duplicates(["forecast_date", "tmdb_id"], keep="last")
    else:
        hist = today
    hist.to_csv(hist_path, index=False)
    print(f"Forecast {len(live)} films.")


if __name__ == "__main__":
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key:
        sys.exit("Set the TMDB_API_KEY environment variable first.")
    run(key)
