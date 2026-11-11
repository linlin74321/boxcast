"""
fetch_tmdb.py - the DAILY AUTOMATIC UPDATE.

GitHub Actions runs this every morning. It:
  1. asks TMDB for films now playing and coming soon in US cinemas,
  2. turns each film into model features (budget, genre, star, director...),
  3. forecasts its worldwide lifetime gross,
  4. saves data/live_forecasts.csv (what the app shows) and appends to
     data/forecast_history.csv (your public track record over time).

Needs a TMDB key in the environment variable TMDB_API_KEY.
Either the short "API Key" or the long "API Read Access Token" works.
"""
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import imdb_primary_genre, is_sequel, studio_group, clean_rating  # noqa: E402
from predict import person_power, predict_pre  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BASE = "https://api.themoviedb.org/3"


def make_getter(key):
    session = requests.Session()
    if key.startswith("eyJ"):                       # long read-access token
        session.headers["Authorization"] = f"Bearer {key}"
        params = {}
    else:                                           # short API key
        params = {"api_key": key}

    def get(path, **extra):
        for attempt in range(3):
            r = session.get(f"{BASE}{path}", params={**params, **extra}, timeout=20)
            if r.status_code == 429:                # too many requests - wait
                time.sleep(2 + attempt * 2)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
    return get


def list_films(get, pages=4):
    films = {}
    for status, path in (("Now playing", "/movie/now_playing"),
                         ("Coming soon", "/movie/upcoming")):
        for page in range(1, pages + 1):
            res = get(path, region="US", page=page)
            for f in res.get("results", []):
                films.setdefault(f["id"], status)
            if page >= res.get("total_pages", 1):
                break
    return films


def us_release(details):
    """US theatrical release date and age certificate (e.g. PG-13)."""
    rel_date, cert = details.get("release_date"), None
    for country in details.get("release_dates", {}).get("results", []):
        if country["iso_3166_1"] != "US":
            continue
        entries = country["release_dates"]
        # Type 3 = wide theatrical release, type 2 = limited release
        for wanted in (3, 2):
            dates = sorted(r["release_date"] for r in entries if r["type"] == wanted)
            if dates:
                rel_date = dates[0][:10]
                break
        certs = [r["certification"] for r in entries if r.get("certification")]
        cert = certs[0] if certs else None
    return rel_date, cert


def film_features(details, status):
    rel, cert = us_release(details)
    rel = pd.to_datetime(rel, errors="coerce")
    cast = details.get("credits", {}).get("cast", [])
    crew = details.get("credits", {}).get("crew", [])
    star = cast[0]["name"] if cast else None
    director = next((c["name"] for c in crew if c.get("job") == "Director"), None)
    studios = [studio_group(c["name"]) for c in details.get("production_companies", [])]
    studio = next((s for s in studios if s != "Other"), "Other")
    s_pow, s_n = person_power(star, "star")
    d_pow, d_n = person_power(director, "director")
    budget = details.get("budget") or np.nan
    return {
        "tmdb_id": details["id"],
        "title": details.get("title"),
        "status": status,
        "release_date": rel.date() if pd.notna(rel) else None,
        "genre": imdb_primary_genre([g["name"] for g in details.get("genres", [])]),
        "genres_all": ", ".join(g["name"] for g in details.get("genres", [])),
        "rating": clean_rating(cert),
        "studio": studio,
        "star": star,
        "director": director,
        "budget": budget,
        "log_budget": np.log(budget) if budget and budget > 0 else np.nan,
        "runtime": details.get("runtime") or np.nan,
        "month": rel.month if pd.notna(rel) else np.nan,
        "year": rel.year if pd.notna(rel) else np.nan,
        "is_sequel": is_sequel(details.get("title"), details.get("belongs_to_collection")),
        "star_power": s_pow, "star_films": s_n,
        "director_power": d_pow, "director_films": d_n,
        "popularity": details.get("popularity"),
        "poster_path": details.get("poster_path"),
        "overview": details.get("overview"),
    }


def run(key):
    get = make_getter(key)
    films = list_films(get)
    rows = []
    for tmdb_id, status in films.items():
        try:
            d = get(f"/movie/{tmdb_id}", append_to_response="credits,release_dates")
            rows.append(film_features(d, status))
        except requests.HTTPError as e:
            print(f"skipped {tmdb_id}: {e}")
        time.sleep(0.05)

    live = pd.DataFrame(rows).dropna(subset=["release_date"])
    live = live.join(predict_pre(live))
    live["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    live = live.sort_values("pred_mid", ascending=False)
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
    print(f"Forecast {len(live)} films. Saved live_forecasts.csv")


if __name__ == "__main__":
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key:
        sys.exit("Set the TMDB_API_KEY environment variable first.")
    run(key)
