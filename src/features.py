"""
features.py - shared helper code.

Everything that turns raw movie information into the numbers the model
understands ("features") lives here, so the training script, the daily TMDB
fetch and the app all use EXACTLY the same rules.
"""
import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# PRE-RELEASE model: predicts a film's WORLDWIDE lifetime gross before release
# ---------------------------------------------------------------------------

# Genres the model knows about. Anything else becomes "Other".
GENRES = ["Action", "Adventure", "Animation", "Biography", "Comedy",
          "Crime", "Drama", "Horror", "Other"]
RATINGS = ["G", "PG", "PG-13", "R", "Other"]
STUDIOS = ["Disney", "Warner", "Universal", "Sony", "Paramount", "Fox",
           "Lionsgate", "MGM", "Other"]

# Our training data (from IMDb) stores only the ALPHABETICALLY FIRST genre of
# each film (e.g. "Action, Adventure, Sci-Fi" -> "Action"). To stay
# consistent, live TMDB films are converted the same way.
TMDB_TO_IMDB_GENRE = {
    "Action": "Action", "Adventure": "Adventure", "Animation": "Animation",
    "Comedy": "Comedy", "Crime": "Crime", "Documentary": "Documentary",
    "Drama": "Drama", "Family": "Family", "Fantasy": "Fantasy",
    "History": "History", "Horror": "Horror", "Music": "Music",
    "Mystery": "Mystery", "Romance": "Romance", "Science Fiction": "Sci-Fi",
    "Thriller": "Thriller", "War": "War", "Western": "Western",
}

# Studio names vary a lot, so we group them by keywords.
STUDIO_KEYWORDS = [
    ("Disney", ["disney", "touchstone", "hollywood pictures", "pixar",
                "marvel", "lucasfilm", "buena vista"]),
    ("Warner", ["warner", "new line", "castle rock", "dc films", "dc studios"]),
    ("Universal", ["universal", "focus features", "illumination", "dreamworks",
                   "gramercy", "blumhouse"]),
    ("Sony", ["columbia", "tristar", "screen gems", "sony"]),
    ("Paramount", ["paramount"]),
    ("Fox", ["fox", "regency"]),
    ("Lionsgate", ["lionsgate", "lions gate", "summit"]),
    ("MGM", ["metro-goldwyn", "mgm", "united artists", "orion", "amazon mgm"]),
]

SEQUEL_RE = re.compile(r"(\b(2|3|4|5|6|7|8|9|II|III|IV|V|VI|Part)\b)|:")

PRE_FEATURES = ["log_budget", "runtime", "month", "year", "is_sequel",
                "star_power", "star_films", "director_power", "director_films",
                "genre", "rating", "studio"]
# Added once the Retrain workflow has fetched franchise history from TMDB
PRE_FEATURES_FRANCHISE = PRE_FEATURES + ["log_prev_gross", "franchise_n"]
# Market signals: China release, competition that weekend, Wikipedia buzz
PRE_FEATURES_SIGNALS = PRE_FEATURES_FRANCHISE + ["china_release", "competition_big",
                                                 "log_wiki_views", "wiki_days_before"]
PRE_CATEGORIES = {"genre": GENRES, "rating": RATINGS, "studio": STUDIOS}


def clean_genre(g):
    return g if g in GENRES else "Other"


def clean_rating(r):
    return r if r in RATINGS else "Other"


def studio_group(name):
    if not isinstance(name, str):
        return "Other"
    low = name.lower()
    for group, keys in STUDIO_KEYWORDS:
        if any(k in low for k in keys):
            return group
    return "Other"


def is_sequel(title, in_collection=False):
    """Rough sequel/franchise detector: numbers, 'Part', or a colon in the title."""
    return int(bool(in_collection) or bool(SEQUEL_RE.search(str(title))))


def imdb_primary_genre(tmdb_genre_names):
    names = sorted(TMDB_TO_IMDB_GENRE[g] for g in tmdb_genre_names
                   if g in TMDB_TO_IMDB_GENRE)
    return clean_genre(names[0]) if names else "Other"


def pre_model_frame(df, features=None):
    """Return only the model columns (in the model's order), with categories
    fixed in a set order. Missing columns are filled as unknown."""
    features = list(features or PRE_FEATURES)
    out = df.copy()
    for col in features:
        if col not in out.columns:
            out[col] = np.nan
    for col, cats in PRE_CATEGORIES.items():
        out[col] = pd.Categorical(out[col], categories=cats)
    for col in features:
        if col not in PRE_CATEGORIES:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[features]


# ---------------------------------------------------------------------------
# POST-RELEASE model: predicts DOMESTIC (US + Canada) lifetime gross after
# the opening weekend(s) are known.
# ---------------------------------------------------------------------------

MAJOR_DISTRIBUTORS = ["disney", "warner", "universal", "sony", "paramount",
                      "fox", "lionsgate"]

POST_BASE = ["log_w1", "log_theaters", "preview_days", "month", "year",
             "holiday", "summer", "major"]
POST_FEATURES = {
    1: POST_BASE,
    2: POST_BASE + ["drop2"],
    3: POST_BASE + ["drop2", "drop3"],
}


def is_major(distributor):
    if not isinstance(distributor, str):
        return 0
    return int(any(k in distributor.lower() for k in MAJOR_DISTRIBUTORS))


def post_feature_row(w1, theaters, release_date, w2=None, w3=None, major=1):
    """Build one row of post-release features from numbers a user can type in."""
    release_date = pd.Timestamp(release_date)
    # Days between release and the first Friday (0 = Friday release,
    # 1 = Thursday, 2 = Wednesday ...). Earlier days add extra money.
    preview_days = (4 - release_date.dayofweek) % 7
    row = {
        "log_w1": np.log(w1),
        "log_theaters": np.log(max(theaters, 1)),
        "preview_days": preview_days,
        "month": release_date.month,
        "year": release_date.year,
        "holiday": int(release_date.month in (11, 12)),
        "summer": int(release_date.month in (5, 6, 7)),
        "major": int(major),
        "drop2": (w2 / w1) if w2 else np.nan,
        "drop3": (w3 / w2) if (w2 and w3) else np.nan,
    }
    return pd.DataFrame([row])
