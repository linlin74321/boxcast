"""
build_datasets.py - downloads the historical data and turns it into tidy
training tables. Run once (or whenever you want to rebuild):

    python src/build_datasets.py

Sources (both free, hosted on GitHub):
  1. IMDb movie stats 1980-2020 (budget, worldwide gross, genre, cast...)
     https://github.com/danielgrijalva/movie-stats
  2. Daily US box office 2000-Jan 2025 scraped from Box Office Mojo
     https://github.com/tjwaterman99/boxofficemojo-scraper
"""
from pathlib import Path

import numpy as np
import pandas as pd

from features import (clean_genre, clean_rating, is_major, is_sequel,
                      studio_group)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
OUT = ROOT / "data"
RAW.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

MOVIES_URL = "https://raw.githubusercontent.com/danielgrijalva/movie-stats/master/movies.csv"
DAILY_URL = ("https://github.com/tjwaterman99/boxofficemojo-scraper/releases/"
             "latest/download/revenues_per_day.csv.gz")


def load_raw():
    movies_path, daily_path = RAW / "movies.csv", RAW / "revenues_per_day.csv.gz"
    movies = pd.read_csv(movies_path if movies_path.exists() else MOVIES_URL)
    daily = pd.read_csv(daily_path if daily_path.exists() else DAILY_URL,
                        parse_dates=["date"])
    return movies, daily


# ---------------------------------------------------------------------------
# 1. Pre-release table
# ---------------------------------------------------------------------------
def prior_power(df, person_col):
    """For each film, the average log-gross of that person's EARLIER films.
    Using only earlier films means the model never 'peeks into the future'."""
    df = df.sort_values("release_date")
    grp = df.groupby(person_col)["log_gross"]
    power = grp.transform(lambda s: s.shift().expanding().mean())
    count = grp.cumcount()
    return power, count


def build_pre_release(movies):
    m = movies.copy()
    m["country_released"] = m["released"].str.extract(r"\((.*)\)")[0]
    m["release_date"] = pd.to_datetime(
        m["released"].str.replace(r"\s*\(.*\)", "", regex=True),
        format="mixed", errors="coerce")
    m = m[(m["country_released"] == "United States")
          & m["release_date"].notna() & m["gross"].notna()
          & (m["gross"] >= 100_000) & m["runtime"].notna()].copy()

    m["log_gross"] = np.log(m["gross"])
    m["log_budget"] = np.log(m["budget"])          # NaN when unknown - fine
    m["month"] = m["release_date"].dt.month
    m["year"] = m["release_date"].dt.year
    m["genre"] = m["genre"].map(clean_genre)
    m["rating"] = m["rating"].map(clean_rating)
    m["studio"] = m["company"].map(studio_group)
    m["is_sequel"] = m["name"].map(is_sequel)

    m["star_power"], m["star_films"] = prior_power(m, "star")
    m["director_power"], m["director_films"] = prior_power(m, "director")

    # A lookup of each person's full track record, used for NEW films.
    people = []
    for role in ("star", "director"):
        p = (m.groupby(role)["log_gross"].agg(["mean", "count"])
              .reset_index().rename(columns={role: "name"}))
        p["role"] = role
        people.append(p)
    people = pd.concat(people, ignore_index=True)

    keep = ["name", "release_date", "year", "month", "genre", "rating", "studio",
            "star", "director", "budget", "gross", "log_budget", "log_gross",
            "runtime", "is_sequel", "star_power", "star_films",
            "director_power", "director_films", "score"]
    m = m[(m["year"] >= 1990) & (m["year"] <= 2019)][keep]
    return m.sort_values("release_date"), people


# ---------------------------------------------------------------------------
# 2. Post-release table (from daily box office)
# ---------------------------------------------------------------------------
def split_runs(daily):
    """Box Office Mojo groups by title, so two different films with the same
    title (e.g. 'Halloween' 2007 and 2018) get mixed up. A gap of 60+ days
    starts a new 'run'."""
    daily = daily.sort_values(["title", "date"]).copy()
    gap = daily.groupby("title")["date"].diff().dt.days.fillna(0)
    daily["run"] = (gap > 60).astype(int).groupby(daily["title"]).cumsum()
    daily["film_id"] = daily["title"] + " #" + daily["run"].astype(str)
    return daily


def build_post_release(daily):
    daily = daily[~daily["title"].str.contains("re-release", case=False, na=False)]
    daily = split_runs(daily)

    rows, curves = [], []
    for fid, g in daily.groupby("film_id", sort=False):
        g = g.sort_values("date")
        first = g["date"].iloc[0]
        preview_days = (4 - first.dayofweek) % 7     # days before first Friday
        if preview_days > 3:                          # opened Sat/Sun/Mon - skip
            continue
        fri = first + pd.Timedelta(days=int(preview_days))
        rev = g.set_index("date")["revenue"]

        def weekend(n):
            start = fri + pd.Timedelta(days=7 * n)
            return rev.loc[start:start + pd.Timedelta(days=2)].sum()

        w1, w2, w3 = weekend(0), weekend(1), weekend(2)
        theaters = g.set_index("date")["theaters"].loc[fri:fri + pd.Timedelta(days=2)].max()
        lifetime = rev.sum()
        if not (w1 > 0 and theaters >= 1000 and lifetime >= w1):
            continue

        rows.append({
            "film_id": fid, "title": g["title"].iloc[0], "release_date": first,
            "distributor": g["distributor"].dropna().iloc[0] if g["distributor"].notna().any() else None,
            "w1": w1, "w2": w2, "w3": w3, "theaters": theaters,
            "preview_days": preview_days, "lifetime": lifetime,
            "days_in_release": (g["date"].iloc[-1] - first).days + 1,
        })
        # Cumulative share of lifetime earned by the end of each week (1-20)
        days = (rev.index - first).days
        cum = rev.groupby(np.minimum(days // 7 + 1, 21)).sum().cumsum()
        cum = cum.reindex(range(1, 21)).ffill().fillna(0) / lifetime
        curves.append([fid] + cum.clip(upper=1).round(4).tolist())

    p = pd.DataFrame(rows)
    p["month"] = p["release_date"].dt.month
    p["year"] = p["release_date"].dt.year
    p["holiday"] = p["month"].isin([11, 12]).astype(int)
    p["summer"] = p["month"].isin([5, 6, 7]).astype(int)
    p["major"] = p["distributor"].map(is_major)
    p["log_w1"] = np.log(p["w1"])
    p["log_theaters"] = np.log(p["theaters"])
    p["drop2"] = np.where(p["w2"] > 0, p["w2"] / p["w1"], np.nan)
    p["drop3"] = np.where((p["w2"] > 0) & (p["w3"] > 0), p["w3"] / p["w2"], np.nan)
    p["multiplier"] = p["lifetime"] / p["w1"]

    # Remove films whose run was cut short by COVID cinema closures, and films
    # still in cinemas when the data ends (Jan 2025).
    covid = (p["release_date"] >= "2020-01-15") & (p["release_date"] <= "2021-06-30")
    p = p[~covid & (p["release_date"] <= "2024-08-31")
          & (p["release_date"] >= "2000-01-07")]

    curves = pd.DataFrame(curves, columns=["film_id"] + [f"wk{i}" for i in range(1, 21)])
    curves = curves[curves["film_id"].isin(p["film_id"])]
    return p.sort_values("release_date"), curves


def build_pre_us(pre, post):
    """Match films across both datasets (same title + year) so the pre-release
    features can also predict US opening weekend and US total."""
    import re
    norm = lambda t: re.sub(r"[^a-z0-9]", "", str(t).lower())
    a = pre.assign(key=pre["name"].map(norm) + pre["year"].astype(str))
    b = post.assign(key=post["title"].map(norm) + post["year"].astype(str))
    b = b.drop_duplicates("key")[["key", "w1", "lifetime", "theaters"]]
    j = a.merge(b, on="key", how="inner").drop(columns="key")
    j["log_w1"] = np.log(j["w1"])
    j["log_dom"] = np.log(j["lifetime"])
    return j


if __name__ == "__main__":
    movies, daily = load_raw()
    pre, people = build_pre_release(movies)
    post, curves = build_post_release(daily)
    pre.to_csv(OUT / "pre_release.csv", index=False)
    people.to_csv(OUT / "people_power.csv", index=False)
    post.to_csv(OUT / "post_release.csv", index=False)
    curves.to_csv(OUT / "curves.csv", index=False)
    build_pre_us(pre, post).to_csv(OUT / "pre_us.csv", index=False)
    print(f"pre-release films: {len(pre):,}  people: {len(people):,}")
    print(f"post-release films: {len(post):,}")
