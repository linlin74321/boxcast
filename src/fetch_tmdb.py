"""
fetch_tmdb.py - the DAILY AUTOMATIC UPDATE (run by GitHub Actions).

  1. asks TMDB for films now playing and coming soon in US cinemas,
  2. forecasts each one,
  3. saves data/live_forecasts.csv (the "In cinemas & coming soon" strip in the
     app) and appends to data/forecast_history.csv (your track record).
"""
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import predict_pre_all  # noqa: E402
from tmdb import TMDB, film_features  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"


def run(key):
    api = TMDB(key)
    films = {}
    for status, which in (("In cinemas", "now_playing"), ("Coming soon", "upcoming")):
        for f in api.listing(which, pages=4):
            films.setdefault(f["id"], status)

    rows = []
    for tmdb_id, status in films.items():
        try:
            rows.append(film_features(api.details(tmdb_id), status))
        except requests.HTTPError as e:
            print(f"skipped {tmdb_id}: {e}")
        time.sleep(0.05)

    live = pd.DataFrame(rows).dropna(subset=["release_date"])
    live = live.join(predict_pre_all(live))
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
