"""
tmdb.py - everything that talks to TMDB (The Movie Database).

Used by the app (live title search) and by the daily update.
The key comes from the TMDB_API_KEY setting: Streamlit "Secrets" for the app,
GitHub "Secrets" for the daily update. Either the short API key or the long
read-access token works.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import requests

from features import clean_rating, imdb_primary_genre, is_sequel, studio_group
from predict import person_power

BASE = "https://api.themoviedb.org/3"
IMG = "https://image.tmdb.org/t/p/"


class TMDB:
    def __init__(self, key):
        self.key = (key or "").strip()
        self.session = requests.Session()
        # Test mode: answers come from a local JSON file instead of TMDB
        self.fake = None
        if os.environ.get("BOXCAST_FAKE_TMDB"):
            with open(os.environ["BOXCAST_FAKE_TMDB"]) as f:
                self.fake = json.load(f)
        if self.key.startswith("eyJ"):
            self.session.headers["Authorization"] = f"Bearer {self.key}"
            self.params = {}
        else:
            self.params = {"api_key": self.key}

    def get(self, path, **extra):
        if self.fake is not None:
            return self._fake_get(path, **extra)
        for attempt in range(3):
            r = self.session.get(f"{BASE}{path}", params={**self.params, **extra}, timeout=20)
            if r.status_code == 429:                    # rate limited - wait a moment
                time.sleep(2 + attempt * 2)
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()

    def _fake_get(self, path, **extra):
        f = self.fake
        if path == "/search/movie":
            q = extra.get("query", "").lower()
            return {"results": [m for m in f["movies"].values() if q in m["title"].lower()]}
        if path == "/discover/movie":
            return {"results": [f["movies"][i] for i in f.get("discover", [])], "total_pages": 1}
        if path.startswith("/collection/"):
            return f["collections"][path.split("/")[-1]]
        if path in ("/movie/now_playing", "/movie/upcoming"):
            return {"results": [f["movies"][i] for i in f["lists"][path]], "total_pages": 1}
        return f["movies"][path.split("/")[-1]]

    # -- endpoints ---------------------------------------------------------
    def search(self, query):
        res = self.get("/search/movie", query=query, include_adult="false")
        results = res.get("results", [])
        # Most-searched films first, so the one people mean is usually on top
        return sorted(results, key=lambda m: m.get("popularity", 0), reverse=True)[:12]

    def details(self, movie_id):
        return self.get(f"/movie/{movie_id}",
                        append_to_response="credits,release_dates,external_ids")

    def previous_installment(self, details):
        """The most recent earlier film in the same TMDB collection (franchise),
        with its worldwide revenue. None if it isn't part of a series."""
        coll = details.get("belongs_to_collection")
        if not coll or not details.get("release_date"):
            return None
        parts = self.get(f"/collection/{coll['id']}").get("parts", [])
        earlier = [p for p in parts if p.get("id") != details["id"] and p.get("release_date")
                   and p["release_date"] < details["release_date"]]
        for part in sorted(earlier, key=lambda p: p["release_date"], reverse=True):
            d = self.get(f"/movie/{part['id']}")
            if (d.get("revenue") or 0) > 1_000_000:
                return {"title": d.get("title"), "year": d["release_date"][:4],
                        "revenue": d["revenue"], "count": len(earlier)}
        return None

    def big_releases_near(self, release_date, exclude_id=None, days=7, big_budget=50e6):
        """How many OTHER $50M+ films open in the US within a week either side."""
        rel = pd.Timestamp(release_date)
        res = self.get("/discover/movie", region="US", with_release_type="2|3",
                       sort_by="popularity.desc", page=1,
                       **{"primary_release_date.gte": (rel - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
                          "primary_release_date.lte": (rel + pd.Timedelta(days=days)).strftime("%Y-%m-%d")})
        count = 0
        for m in res.get("results", [])[:12]:
            if m["id"] == exclude_id:
                continue
            d = self.get(f"/movie/{m['id']}")
            if (d.get("budget") or 0) >= big_budget:
                count += 1
        return count

    def listing(self, which, pages=1):
        out = []
        for page in range(1, pages + 1):
            res = self.get(f"/movie/{which}", region="US", page=page)
            out += res.get("results", [])
            if page >= res.get("total_pages", 1):
                break
        return out


def poster_url(path, size="w342"):
    return f"{IMG}{size}{path}" if path else None


def us_release(details):
    """US theatrical release date and age certificate (e.g. PG-13)."""
    rel_date, cert = details.get("release_date"), None
    for country in details.get("release_dates", {}).get("results", []):
        if country["iso_3166_1"] != "US":
            continue
        entries = country["release_dates"]
        for wanted in (3, 2):                      # 3 = wide release, 2 = limited
            dates = sorted(r["release_date"] for r in entries if r["type"] == wanted)
            if dates:
                rel_date = dates[0][:10]
                break
        certs = [r["certification"] for r in entries if r.get("certification")]
        cert = certs[0] if certs else None
    return rel_date, cert


def film_features(details, status=None):
    """Turn a TMDB movie record into everything the models (and the app) need."""
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
    china = any(c["iso_3166_1"] == "CN" and any(r["type"] in (2, 3) for r in c["release_dates"])
                for c in details.get("release_dates", {}).get("results", []))
    return {
        "tmdb_id": details["id"],
        "title": details.get("title"),
        "status": status,
        "release_date": rel.date() if pd.notna(rel) else None,
        "genre": imdb_primary_genre([g["name"] for g in details.get("genres", [])]),
        "genres_all": ", ".join(g["name"] for g in details.get("genres", [])),
        "rating": clean_rating(cert),
        "certification": cert,
        "studio": studio,
        "star": star,
        "cast_top": ", ".join(c["name"] for c in cast[:4]),
        "director": director,
        "budget": budget,
        "revenue": details.get("revenue") or np.nan,
        "log_budget": np.log(budget) if budget and budget > 0 else np.nan,
        "runtime": details.get("runtime") or np.nan,
        "month": rel.month if pd.notna(rel) else np.nan,
        "year": rel.year if pd.notna(rel) else np.nan,
        "is_sequel": is_sequel(details.get("title"), details.get("belongs_to_collection")),
        "star_power": s_pow, "star_films": s_n,
        "director_power": d_pow, "director_films": d_n,
        "popularity": details.get("popularity"),
        "poster_path": details.get("poster_path"),
        "backdrop_path": details.get("backdrop_path"),
        "overview": details.get("overview"),
        "tagline": details.get("tagline"),
        "log_prev_gross": np.nan,     # filled in when the previous film is known
        "franchise_n": 0,
        "china_release": int(china),
        "wikidata_id": details.get("external_ids", {}).get("wikidata_id"),
        "competition_big": np.nan,    # filled in by the app / daily update
        "log_wiki_views": np.nan,
        "wiki_days_before": np.nan,
    }
