"""
buzz.py - online buzz from Wikipedia page views (free, no key needed).

Daily views of a film's English Wikipedia article are a well-studied early
signal of audience interest (Mestyan, Yasseri & Kertesz, PLOS ONE 2013).
Daily data is timestamped, so for past films we can use ONLY views from before
release - no peeking at how the film did.

We measure "views in the 28 days ending N days before release" for
N = 7, 30, 60 and 90, so the model learns how buzz builds up to release and an
upcoming film can be compared with past films at the same distance from release.
"""
import json
import os
from datetime import timedelta
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

HORIZONS = (7, 30, 60, 90)
WINDOW = 28
FIRST_DAY = pd.Timestamp("2015-07-01")       # Wikimedia's daily data starts here
HEADERS = {"User-Agent": "BoxCast/1.0 (student box office forecasting project; "
                         "https://github.com/linlin74321/boxcast)"}
PAGEVIEWS = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
             "en.wikipedia/all-access/user/{title}/daily/{start}/{end}")

_session = requests.Session()
_session.headers.update(HEADERS)
_FAKE = None
if os.environ.get("BOXCAST_FAKE_WIKI"):           # test mode: answers from a local file
    with open(os.environ["BOXCAST_FAKE_WIKI"]) as fh:
        _FAKE = json.load(fh)


def wiki_title(wikidata_id):
    """English Wikipedia article title for a Wikidata id (e.g. Q12345)."""
    if not wikidata_id:
        return None
    if _FAKE is not None:
        return _FAKE["titles"].get(wikidata_id)
    r = _session.get(f"https://www.wikidata.org/wiki/Special:EntityData/{wikidata_id}.json", timeout=20)
    if r.status_code != 200:
        return None
    ent = r.json().get("entities", {}).get(wikidata_id, {})
    return ent.get("sitelinks", {}).get("enwiki", {}).get("title")


def daily_views(title, start, end):
    """Daily page views between two dates (a pandas Series indexed by date)."""
    start, end = max(pd.Timestamp(start), FIRST_DAY), pd.Timestamp(end)
    if not title or end < start:
        return pd.Series(dtype=float)
    if _FAKE is not None:
        level = _FAKE["views"].get(title)
        if level is None:
            return pd.Series(dtype=float)
        days = pd.date_range(start, end)
        rel = pd.Timestamp(_FAKE.get("release", {}).get(title, end))
        ramp = np.clip(1 - (rel - days).days / 150, 0.15, 1.0)   # buzz grows towards release
        return pd.Series(level * ramp, index=days)
    url = PAGEVIEWS.format(title=quote(title.replace(" ", "_"), safe=""),
                           start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"))
    r = _session.get(url, timeout=20)
    if r.status_code == 404:
        return pd.Series(dtype=float)
    r.raise_for_status()
    items = r.json().get("items", [])
    return pd.Series({pd.Timestamp(i["timestamp"][:8]): i["views"] for i in items}, dtype=float)


def window_sum(series, end):
    """Views in the 28 days ending on `end` (None if that period isn't covered)."""
    end = pd.Timestamp(end)
    start = end - timedelta(days=WINDOW - 1)
    if start < FIRST_DAY:
        return None
    return float(series.loc[start:end].sum()) if len(series) else None


def horizon_views(title, release):
    """{7: views, 30: views, 60: views, 90: views} for a past film."""
    release = pd.Timestamp(release)
    s = daily_views(title, release - timedelta(days=max(HORIZONS) + WINDOW), release - timedelta(days=1))
    return {h: window_sum(s, release - timedelta(days=h)) for h in HORIZONS}


def live_buzz(title, release, today):
    """Buzz for a film being forecast now: (views in the last 28 days measured
    as close to release as possible, days before release at that point).
    Returns (None, None) when there's nothing comparable to past films."""
    release, today = pd.Timestamp(release), pd.Timestamp(today)
    end = min(today - timedelta(days=1), release - timedelta(days=min(HORIZONS)))
    days_before = (release - end).days
    if days_before > 120 or not title:
        return None, None
    s = daily_views(title, end - timedelta(days=WINDOW - 1), end)
    total = window_sum(s, end)
    if total is None:
        return None, None
    return total, int(np.clip(days_before, min(HORIZONS), max(HORIZONS)))
