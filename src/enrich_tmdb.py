"""
enrich_tmdb.py - adds TMDB knowledge to the training data (run by the
"Retrain models" GitHub workflow; needs TMDB_API_KEY).

  1. Franchise history: for every historical film, find it on TMDB, check
     whether it belongs to a series, and record what the PREVIOUS film in that
     series earned worldwide.            -> data/tmdb_history.csv
  2. Recent films: US releases from 2020 to last year, with budget, cast,
     director, genre, rating, worldwide gross and franchise history.
                                         -> data/tmdb_recent.csv

Results are cached, so re-running only fetches what's missing.
"""
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tmdb import TMDB, film_features  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
HISTORY, RECENT = DATA / "tmdb_history.csv", DATA / "tmdb_recent.csv"


class Enricher:
    def __init__(self, key):
        self.api = TMDB(key)
        self._details, self._collections = {}, {}

    def details(self, movie_id):
        if movie_id not in self._details:
            self._details[movie_id] = self.api.get(
                f"/movie/{movie_id}", append_to_response="credits,release_dates")
            time.sleep(0.02)
        return self._details[movie_id]

    def collection(self, coll_id):
        if coll_id not in self._collections:
            self._collections[coll_id] = self.api.get(f"/collection/{coll_id}").get("parts", [])
            time.sleep(0.02)
        return self._collections[coll_id]

    def previous(self, d):
        """(previous film's worldwide revenue, its title, how many earlier films)."""
        coll = d.get("belongs_to_collection")
        if not coll or not d.get("release_date"):
            return np.nan, None, 0
        parts = self.collection(coll["id"])
        earlier = sorted([p for p in parts if p.get("id") != d["id"] and p.get("release_date")
                          and p["release_date"] < d["release_date"]],
                         key=lambda p: p["release_date"], reverse=True)
        for p in earlier:
            pd_ = self.details(p["id"])
            if (pd_.get("revenue") or 0) > 1_000_000:
                return float(pd_["revenue"]), pd_.get("title"), len(earlier)
        return np.nan, None, len(earlier)

    def find(self, title, year):
        """TMDB id for a historical film, matching title and release year (±1)."""
        res = self.api.get("/search/movie", query=title, year=int(year)).get("results", [])
        if not res:
            res = self.api.get("/search/movie", query=title).get("results", [])
        for r in res:
            y = (r.get("release_date") or "0000")[:4]
            if y.isdigit() and abs(int(y) - int(year)) <= 1:
                return r["id"]
        return None


def enrich_history(e, pre):
    done = pd.read_csv(HISTORY) if HISTORY.exists() else pd.DataFrame(columns=["key"])
    have = set(done["key"])
    rows = []
    todo = [r for r in pre.itertuples() if f"{r.name}|{r.year}" not in have]
    print(f"Franchise history: {len(have)} cached, {len(todo)} to look up")
    for i, r in enumerate(todo, 1):
        key = f"{r.name}|{r.year}"
        try:
            mid = e.find(r.name, r.year)
            if mid is None:
                rows.append({"key": key, "tmdb_id": None, "prev_gross": np.nan,
                             "prev_title": None, "franchise_n": 0})
                continue
            prev, prev_title, n = e.previous(e.details(mid))
            rows.append({"key": key, "tmdb_id": mid, "prev_gross": prev,
                         "prev_title": prev_title, "franchise_n": n})
        except Exception as err:                  # keep going; try again next run
            print(f"  skipped {key}: {err}")
        if i % 250 == 0:
            print(f"  {i}/{len(todo)}")
            pd.concat([done, pd.DataFrame(rows)]).to_csv(HISTORY, index=False)
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(HISTORY, index=False)
    print(f"  saved {len(out)} films, {out['prev_gross'].notna().sum()} with a previous installment")


def enrich_recent(e, first_year=2020, pages_per_year=15):
    last_year = date.today().year - 1            # only films that have finished their run
    done = pd.read_csv(RECENT) if RECENT.exists() else pd.DataFrame(columns=["tmdb_id"])
    have = set(done["tmdb_id"].dropna().astype(int))
    rows = []
    for year in range(first_year, last_year + 1):
        ids = []
        for page in range(1, pages_per_year + 1):
            res = e.api.get("/discover/movie", region="US", with_release_type="2|3",
                            sort_by="revenue.desc", page=page,
                            **{"primary_release_date.gte": f"{year}-01-01",
                               "primary_release_date.lte": f"{year}-12-31"})
            ids += [m["id"] for m in res.get("results", [])]
            if page >= res.get("total_pages", 1):
                break
        new = [i for i in ids if i not in have]
        print(f"Recent films {year}: {len(ids)} found, {len(new)} new")
        cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=150)
        for mid in new:
            have.add(mid)                        # never save the same film twice
            try:
                d = e.details(mid)
                if (d.get("revenue") or 0) < 1_000_000 or not d.get("runtime"):
                    continue
                f = film_features(d)
                rel = pd.Timestamp(f["release_date"]) if f["release_date"] else None
                # Only films that have finished their cinema run count as "actual" results
                if rel is None or rel > cutoff or rel.year < first_year:
                    continue
                prev, prev_title, n = e.previous(d)
                rows.append({**{k: f[k] for k in ("tmdb_id", "title", "release_date", "genre",
                                                  "rating", "studio", "star", "director",
                                                  "budget", "runtime")},
                             "gross": d["revenue"], "in_collection": int(bool(d.get("belongs_to_collection"))),
                             "prev_gross": prev, "prev_title": prev_title, "franchise_n": n})
            except Exception as err:
                print(f"  skipped {mid}: {err}")
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(RECENT, index=False)
    print(f"  saved {len(out)} recent films")


if __name__ == "__main__":
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key:
        sys.exit("Set the TMDB_API_KEY environment variable first.")
    e = Enricher(key)
    pre = pd.read_csv(DATA / "pre_release_imdb.csv")
    enrich_history(e, pre)
    enrich_recent(e)
