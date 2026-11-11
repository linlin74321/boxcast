"""
enrich_tmdb.py - adds outside knowledge to the training data (run by the
"Retrain models" GitHub workflow; needs TMDB_API_KEY).

  1. Franchise history + China release: for every historical film, find it on
     TMDB, record what the PREVIOUS film in its series earned worldwide, and
     whether it had a cinema release in China.      -> data/tmdb_history.csv
  2. Recent films: US releases from 2020 to last year, with budget, cast,
     director, genre, rating, worldwide gross, franchise history and China
     release.                                       -> data/tmdb_recent.csv
  3. Online buzz: daily views of each film's English Wikipedia page, added up
     for the 4 weeks ending 7, 30, 60 and 90 days before release
     (films released from late 2015, when the data starts). -> data/wiki_views.csv

Everything is cached, so re-running only fetches what's missing.
"""
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from buzz import FIRST_DAY, horizon_views, wiki_title  # noqa: E402
from tmdb import TMDB, film_features  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
HISTORY, RECENT, WIKI = DATA / "tmdb_history.csv", DATA / "tmdb_recent.csv", DATA / "wiki_views.csv"


def china_release(d):
    return int(any(c["iso_3166_1"] == "CN" and any(r["type"] in (2, 3) for r in c["release_dates"])
                   for c in d.get("release_dates", {}).get("results", [])))


class Enricher:
    def __init__(self, key):
        self.api = TMDB(key)
        self._details, self._collections = {}, {}

    def details(self, movie_id):
        movie_id = int(movie_id)
        if movie_id not in self._details:
            self._details[movie_id] = self.api.get(
                f"/movie/{movie_id}", append_to_response="credits,release_dates,external_ids")
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

    def extras(self, d):
        return {"china_release": china_release(d),
                "wikidata_id": d.get("external_ids", {}).get("wikidata_id")}


def refresh_missing(e, df, label):
    """Fill China / Wikidata columns for rows cached before they existed."""
    for col in ("china_release", "wikidata_id"):
        if col not in df:
            df[col] = np.nan
    df["wikidata_id"] = df["wikidata_id"].astype(object)
    todo = df.index[df["china_release"].isna() & df["tmdb_id"].notna()]
    print(f"{label}: adding China/Wikipedia info to {len(todo)} cached films")
    for n, i in enumerate(todo, 1):
        try:
            for k, v in e.extras(e.details(df.at[i, "tmdb_id"])).items():
                df.at[i, k] = v
        except Exception as err:
            print(f"  skipped {df.at[i, 'tmdb_id']}: {err}")
        if n % 500 == 0:
            print(f"  {n}/{len(todo)}")
    return df


def enrich_history(e, pre):
    done = pd.read_csv(HISTORY) if HISTORY.exists() else pd.DataFrame(columns=["key", "tmdb_id"])
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
                             "prev_title": None, "franchise_n": 0, "china_release": 0})
                continue
            d = e.details(mid)
            prev, prev_title, n = e.previous(d)
            rows.append({"key": key, "tmdb_id": mid, "prev_gross": prev,
                         "prev_title": prev_title, "franchise_n": n, **e.extras(d)})
        except Exception as err:                  # keep going; try again next run
            print(f"  skipped {key}: {err}")
        if i % 250 == 0:
            print(f"  {i}/{len(todo)}")
            pd.concat([done, pd.DataFrame(rows)]).to_csv(HISTORY, index=False)
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out = refresh_missing(e, out, "Franchise history")
    out.to_csv(HISTORY, index=False)
    print(f"  saved {len(out)} films, {out['prev_gross'].notna().sum()} with a previous installment")


def enrich_recent(e, first_year=2020, pages_per_year=15):
    last_year = date.today().year - 1
    done = pd.read_csv(RECENT) if RECENT.exists() else pd.DataFrame(columns=["tmdb_id"])
    have = set(done["tmdb_id"].dropna().astype(int))
    rows = []
    cutoff = pd.Timestamp(date.today()) - pd.Timedelta(days=150)
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
                             "prev_gross": prev, "prev_title": prev_title, "franchise_n": n,
                             **e.extras(d)})
            except Exception as err:
                print(f"  skipped {mid}: {err}")
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out = refresh_missing(e, out, "Recent films")
    out.to_csv(RECENT, index=False)
    print(f"  saved {len(out)} recent films")


def enrich_wiki(pre):
    """Wikipedia views before release for every film released from late 2015."""
    films = []
    if HISTORY.exists():
        h = pd.read_csv(HISTORY)
        dates = pre.assign(key=pre["name"] + "|" + pre["year"].astype(str)).drop_duplicates("key")
        h["release_date"] = h["key"].map(dates.set_index("key")["release_date"])
        films.append(h[["key", "wikidata_id", "release_date"]])
    if RECENT.exists():
        r = pd.read_csv(RECENT)
        r["key"] = "tmdb:" + r["tmdb_id"].astype("Int64").astype(str)
        films.append(r[["key", "wikidata_id", "release_date"]])
    if not films:
        return
    films = pd.concat(films, ignore_index=True)
    films["release_date"] = pd.to_datetime(films["release_date"], errors="coerce")
    films = films[films["wikidata_id"].notna()
                  & (films["release_date"] >= FIRST_DAY + pd.Timedelta(days=35))]

    done = pd.read_csv(WIKI) if WIKI.exists() else pd.DataFrame(columns=["key"])
    have = set(done["key"])
    todo = films[~films["key"].isin(have)]
    print(f"Wikipedia buzz: {len(have)} cached, {len(todo)} to fetch")
    rows = []
    for n, r in enumerate(todo.itertuples(), 1):
        try:
            title = wiki_title(r.wikidata_id)
            views = horizon_views(title, r.release_date) if title else {}
            rows.append({"key": r.key, "wiki_title": title,
                         **{f"wiki_{h}": v for h, v in views.items()}})
        except Exception as err:
            print(f"  skipped {r.key}: {err}")
        time.sleep(0.02)
        if n % 250 == 0:
            print(f"  {n}/{len(todo)}")
            pd.concat([done, pd.DataFrame(rows)]).to_csv(WIKI, index=False)
    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
    out.to_csv(WIKI, index=False)
    n_buzz = out["wiki_30"].notna().sum() if "wiki_30" in out else 0
    print(f"  saved {len(out)} films, {n_buzz} with buzz data")


if __name__ == "__main__":
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key:
        sys.exit("Set the TMDB_API_KEY environment variable first.")
    e = Enricher(key)
    pre = pd.read_csv(DATA / "pre_release_imdb.csv")
    enrich_history(e, pre)
    enrich_recent(e)
    enrich_wiki(pre)
