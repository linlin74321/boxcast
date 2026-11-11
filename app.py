"""
BoxCast - type a film title, get its box office forecast.

Run locally:  streamlit run app.py
Needs a TMDB key in Streamlit's Secrets as TMDB_API_KEY (see README).
"""
import html
import math
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from predict import (BREAK_EVEN_MULTIPLE, break_even_chance, explain,  # noqa: E402
                     metrics, predict_post, predict_pre_all)
from style import CSS  # noqa: E402
from tmdb import TMDB, film_features, poster_url  # noqa: E402

DATA = ROOT / "data"
st.set_page_config(page_title="BoxCast: box office forecasts", page_icon="🎬",
                   layout="wide", initial_sidebar_state="collapsed")
st.html(CSS)


# ------------------------------------------------------------------ helpers
def tmdb_key():
    """Find the TMDB key in Streamlit Secrets, forgiving common pasting slips:
    a different name (tmdb_key, TMDB API KEY...), a [section] header,
    extra spaces or quotes, or a leading 'Bearer '."""
    found = ""
    try:
        def walk(d, word):
            for k, v in d.items():
                if hasattr(v, "items"):
                    r = walk(v, word) or (walk(v, "") if word in str(k).lower() else "")
                    if r:
                        return r
                elif word in str(k).lower():
                    return str(v)
            return ""
        found = walk(st.secrets, "tmdb") or walk(st.secrets, "key")
    except Exception:
        found = ""
    key = found or os.environ.get("TMDB_API_KEY", "")
    key = key.strip().strip('"').strip("'").strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


def tmdb_problem(err):
    """Turn a TMDB failure into a specific, fixable message."""
    import requests
    if isinstance(err, requests.HTTPError) and err.response is not None:
        code = err.response.status_code
        if code == 401:
            return ("TMDB rejected the key (error 401). In Streamlit Secrets, paste the "
                    "<b>API Key</b> from themoviedb.org → Settings → API (a 32-character code), "
                    "save, and search again.")
        if code == 404:
            return "TMDB couldn't find that film (error 404). Go back and pick it again."
        return f"TMDB returned error {code}. Wait a minute and search again."
    if isinstance(err, requests.RequestException):
        return f"Couldn't reach TMDB ({type(err).__name__}). Wait a minute and search again."
    return f"Something went wrong talking to TMDB: {esc(type(err).__name__)}: {esc(str(err))[:200]}"


KEY = tmdb_key()
HAS_API = bool(KEY) or bool(os.environ.get("BOXCAST_FAKE_TMDB"))
esc = html.escape


def money(x):
    if x is None or pd.isna(x):
        return "—"
    if x >= 1e9:
        return f"${x / 1e9:.2f}B"
    if x >= 1e8:
        return f"${x / 1e6:.0f}M"
    if x >= 1e6:
        return f"${x / 1e6:.1f}M"
    return f"${x / 1e3:.0f}K"


def nice_date(d):
    d = pd.Timestamp(d)
    return f"{d.day} {d.strftime('%b %Y')}"


def img(path, size="w342"):
    if not path or (isinstance(path, float) and np.isnan(path)):
        return None
    return path if str(path).startswith("data:") else poster_url(path, size)


@st.cache_data(ttl=3600, show_spinner=False)
def search(query, key):
    return TMDB(key).search(query)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def details(movie_id, key):
    return TMDB(key).details(movie_id)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def listing(which, key):
    return TMDB(key).listing(which)


@st.cache_data(show_spinner=False)
def read_csv(name, modified):
    return pd.read_csv(DATA / name)


def load(name):
    path = DATA / name
    return read_csv(name, path.stat().st_mtime) if path.exists() else None


def open_film(movie_id):
    st.query_params.clear()
    st.query_params["film"] = str(movie_id)


def go_home():
    st.query_params.clear()


# ------------------------------------------------------------------ pieces
def top_bar(active):
    st.html(f"""
    <div class="bc-top">
      <a class="bc-mark" href="./" target="_self">Box<span>Cast</span></a>
      <nav class="bc-nav">
        <a href="./" target="_self" class="{'on' if active == 'home' else ''}">Forecast a film</a>
        <a href="?page=about" target="_self" class="{'on' if active == 'about' else ''}">How it works</a>
      </nav>
    </div>""")


def footer():
    st.html("""<div class="bc-foot">BoxCast forecasts are statistical estimates, not guarantees.
    Film details and images come from TMDB. This product uses the TMDB API but is not endorsed
    or certified by TMDB.</div>""")


def poster_grid(items, prefix, n_cols=6):
    """items: dicts with id, title, poster, sub, fig. The poster is the button."""
    rules = []
    for it in items:
        sel = f".st-key-pick_{prefix}_{it['id']} button"
        src = img(it.get("poster"))
        if src:
            rules.append(f'{sel} {{ background-image: url("{src}"); }}')
        else:
            rules.append(f"{sel} p {{ color: var(--muted) !important; font-size: .9rem !important; padding: 0 .6rem; }}")
    st.html("<style>" + "".join(rules) + "</style>")
    for start in range(0, len(items), n_cols):
        cols = st.columns(n_cols, gap="medium")
        for col, it in zip(cols, items[start:start + n_cols]):
            with col, st.container(gap="small"):
                st.button(it["title"], key=f"pick_{prefix}_{it['id']}", width="stretch",
                          on_click=open_film, args=(it["id"],))
                fig = f'<div class="bc-card-f">{it["fig"]}</div>' if it.get("fig") else ""
                st.html(f'<div class="bc-card-t">{esc(it["title"])}</div>'
                        f'<div class="bc-card-m">{esc(it.get("sub") or "")}</div>{fig}')


def note(text):
    st.html(f'<div class="bc-note">{text}</div>')


def no_key_note():
    note("Search isn't switched on yet. Add your TMDB key in Streamlit: "
         "<b>Manage app → Settings → Secrets</b>, paste <code>TMDB_API_KEY = \"your key\"</code> and save.")


# ------------------------------------------------------------------ home
def home():
    st.html("""<div class="bc-hero">
      <h1>Know the number before opening night.</h1>
      <p>Type any film. BoxCast forecasts its box office from three decades of
      releases, and shows you what's driving the number.</p></div>""")
    query = st.text_input("Search for a film", key="q", label_visibility="collapsed",
                          placeholder="Search a film title")
    query = query.strip()
    if query:
        if not HAS_API:
            no_key_note()
        else:
            try:
                results = search(query, KEY)
            except Exception as err:
                results = None
                note(tmdb_problem(err))
            if results == []:
                note(f"No films match “{esc(query)}”. Check the spelling or try the original title.")
            elif results:
                st.html(f'<div class="bc-section">Results for “{esc(query)}”</div>')
                poster_grid([{"id": r["id"], "title": r["title"], "poster": r.get("poster_path"),
                              "sub": (r.get("release_date") or "")[:4] or "Date not announced"}
                             for r in results[:6]], "res")

    live = load("live_forecasts.csv")
    items = []
    if live is not None and not live.empty:
        live = live.sort_values("popularity", ascending=False).head(12)
        items = [{"id": r.tmdb_id, "title": r.title, "poster": r.poster_path,
                  "sub": nice_date(r.release_date), "fig": f"{money(r.pred_mid)} forecast"}
                 for r in live.itertuples()]
        sub = "Forecasts refresh every morning. Figures are worldwide lifetime gross."
    elif HAS_API:
        try:
            films = listing("now_playing", KEY)[:6] + listing("upcoming", KEY)[:6]
            items = [{"id": f["id"], "title": f["title"], "poster": f.get("poster_path"),
                      "sub": nice_date(f["release_date"]) if f.get("release_date") else ""}
                     for f in films]
        except Exception:
            items = []
        sub = "Pick a film to see its forecast"
    if items:
        st.html('<div class="bc-section">In cinemas and coming soon</div>'
                f'<div class="bc-sub">{sub}</div>')
        poster_grid(items, "rail")


# ------------------------------------------------------------------ film page
def track_html(low, mid, high, be):
    lo = min(low, be or low) / 1.5
    hi = max(high, be or high) * 1.5
    pos = lambda v: (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) * 100
    be_html = ""
    if be:
        be_html = (f'<div class="be" style="left:{pos(be):.1f}%"></div>'
                   f'<div class="belab" style="left:{pos(be):.1f}%">Break-even ≈ {money(be)}</div>')
    return f"""<div class="bc-track" role="img"
      aria-label="Forecast range {money(low)} to {money(high)}, best estimate {money(mid)}">
      <div class="rail"></div>
      <div class="band" style="left:{pos(low):.1f}%; width:{pos(high) - pos(low):.1f}%"></div>
      <div class="mid" style="left:{pos(mid):.1f}%"></div>
      {be_html}
      <div class="lab" style="left:{pos(low):.1f}%">{money(low)}</div>
      <div class="lab" style="left:{pos(high):.1f}%">{money(high)}</div>
    </div>"""


def drivers_html(drivers):
    top = drivers[:7]
    if not top:
        return ""
    biggest = max(abs(d[2]) for d in top)
    rows = []
    for name, pct, logv in top:
        w = abs(logv) / biggest * 48
        left = 50 if logv > 0 else 50 - w
        color = "var(--up)" if logv > 0 else "var(--down)"
        sign = "+" if pct > 0 else "−"
        label = f"{1 + pct:.1f}×" if pct >= 1 else f"{sign}{abs(pct) * 100:.0f}%"
        rows.append(f"""<div class="bc-drv"><div class="n">{esc(name)}</div>
          <div class="b"><i style="left:{left:.1f}%; width:{w:.1f}%; background:{color}"></i></div>
          <div class="v">{label}</div></div>""")
    return "".join(rows)


def comparables(f, p):
    hist = load("pre_release.csv")
    pool = hist[hist["genre"] == f["genre"]] if f["genre"] != "Other" else hist
    if pd.notna(f["budget"]):
        pool = pool[pool["budget"].notna()].assign(gap=lambda d: (np.log(d["budget"]) - np.log(f["budget"])).abs())
    else:
        pool = pool.assign(gap=lambda d: (d["log_gross"] - np.log(p["pred_mid"])).abs())
    pool = pool.nsmallest(6, "gap").sort_values("gross", ascending=False)
    rows = "".join(
        f"<tr><td>{esc(r.name)}</td><td>{r.year}</td><td class='r'>{money(r.budget)}</td>"
        f"<td class='r'>{money(r.gross)}</td>"
        f"<td class='r'>{(r.gross / r.budget):.1f}×</td></tr>" if pd.notna(r.budget) else
        f"<tr><td>{esc(r.name)}</td><td>{r.year}</td><td class='r'>—</td>"
        f"<td class='r'>{money(r.gross)}</td><td class='r'>—</td></tr>"
        for r in pool.itertuples())
    return ("<table class='bc-tbl'><thead><tr><th>Film</th><th>Year</th><th class='r'>Budget</th>"
            f"<th class='r'>Worldwide</th><th class='r'>Return</th></tr></thead><tbody>{rows}</tbody></table>")


def film_page(movie_id):
    st.button("Back to search", key="back", on_click=go_home)
    if not HAS_API:
        no_key_note()
        return
    try:
        d = details(movie_id, KEY)
    except Exception as err:
        note(tmdb_problem(err))
        return
    f = film_features(d)
    if pd.isna(f["year"]):
        note(f"TMDB has no release date for <b>{esc(f['title'])}</b> yet, so it can't be forecast. "
             "Check back once a date is announced.")
        return
    row = pd.DataFrame([f])
    p = predict_pre_all(row).iloc[0]
    base, drivers = explain(row)
    be = BREAK_EVEN_MULTIPLE * f["budget"] if pd.notna(f["budget"]) else None
    chance = break_even_chance(p["pred_low"], p["pred_mid"], p["pred_high"], f["budget"])

    # ---- header with backdrop
    rel = pd.Timestamp(f["release_date"])
    chips = [str(rel.year)]
    if f.get("certification"):
        chips.append(f["certification"])
    if pd.notna(f["runtime"]):
        chips.append(f"{int(f['runtime']) // 60}h {int(f['runtime']) % 60:02d}m")
    chips += [g for g in (f["genres_all"] or "").split(", ") if g]
    backdrop = img(f.get("backdrop_path"), "w1280")
    poster = img(f.get("poster_path"), "w500")
    people = []
    if f["director"]:
        people.append(f"Directed by <b>{esc(f['director'])}</b>")
    if f["cast_top"]:
        people.append(f"Starring <b>{esc(f['cast_top'])}</b>")
    st.html(f"""
    <div class="bc-film" style="{f'background-image:url({backdrop});' if backdrop else ''}">
      <div class="bc-film-in">
        <div class="bc-poster" style="{f'background-image:url({poster});' if poster else ''}"
             role="img" aria-label="Poster for {esc(f['title'])}"></div>
        <div>
          <h1>{esc(f['title'])}</h1>
          <div class="bc-meta">{''.join(f'<span class="bc-chip">{esc(c)}</span>' for c in chips)}</div>
          <div class="bc-people">{'<br>'.join(people)}</div>
          <div class="bc-overview">{esc(f['overview'] or '')}</div>
        </div>
      </div>
    </div>""")

    # ---- the forecast
    st.html(f"""
    <div class="bc-fc">
      <div class="bc-fc-k">Forecast worldwide box office</div>
      <div class="bc-fc-n">{money(p['pred_mid'])}</div>
      <div class="bc-fc-r">Likely between {money(p['pred_low'])} and {money(p['pred_high'])}</div>
      {track_html(p['pred_low'], p['pred_mid'], p['pred_high'], be)}
    </div>""")

    if chance is None:
        verdict = ('<div class="v">—</div><div class="s">No budget listed on TMDB</div>')
    else:
        cls, word, icon = (("bc-good", "Likely profitable", "✓") if chance >= 0.6 else
                           ("bc-warn", "Could go either way", "≈") if chance >= 0.35 else
                           ("bc-risk", "Risky", "!"))
        verdict = (f'<div class="v">{chance * 100:.0f}%</div>'
                   f'<div class="s"><span class="{cls}">{icon} {word}</span> on a {money(f["budget"])} budget</div>')
    st.html(f"""
    <div class="bc-figs">
      <div class="bc-fig"><div class="k">US opening weekend</div>
        <div class="v">{money(p['open_mid'])}</div>
        <div class="s">{money(p['open_low'])} to {money(p['open_high'])}</div></div>
      <div class="bc-fig"><div class="k">US and Canada total</div>
        <div class="v">{money(p['dom_mid'])}</div>
        <div class="s">{money(p['dom_low'])} to {money(p['dom_high'])}</div></div>
      <div class="bc-fig"><div class="k">Chance to break even</div>{verdict}</div>
    </div>""")

    today = pd.Timestamp(date.today())
    if rel <= today and pd.notna(f["revenue"]) and f["revenue"] > 0:
        diff = (p["pred_mid"] - f["revenue"]) / f["revenue"] * 100
        extra = (" This film is part of the data the model learned from, so treat this as a sanity "
                 "check rather than a fair test." if rel.year <= 2019 else "")
        note(f"Released {nice_date(rel)}. TMDB reports <b>{money(f['revenue'])}</b> worldwide so far. "
             f"Our forecast is {abs(diff):.0f}% {'above' if diff > 0 else 'below'} that.{extra}")
    elif rel <= today:
        note(f"In cinemas since {nice_date(rel)}. Once the opening weekend is reported, use "
             "<b>Refine with real box office</b> below for a much sharper US forecast.")
    else:
        note(f"Opens in US cinemas on {nice_date(rel)}.")
    if pd.isna(f["budget"]):
        note("TMDB doesn't list a budget for this film, so the forecast leans on its genre, cast, "
             "director and timing, and the range is wider.")

    # ---- why
    st.html(f'<div class="bc-section">Why this number</div>'
            f'<div class="bc-sub">A typical film in our data earns {money(base)} worldwide. '
            f'Here\u2019s what moves this one up or down from there.</div>{drivers_html(drivers)}')

    # ---- comparables
    st.html('<div class="bc-section">Films like this</div>'
            '<div class="bc-sub">Same genre, closest budget. Worldwide lifetime gross.</div>'
            + comparables(f, p))

    # ---- refine with real numbers
    st.html('<div class="bc-section">Refine with real box office</div>'
            '<div class="bc-sub">Once the film opens, enter its weekend grosses. '
            'Typical error drops from about 45% to under 10% by the second weekend.</div>')
    with st.form(f"refine_{movie_id}", border=True):
        a, b, c, e = st.columns(4)
        w1 = a.number_input("Opening weekend ($M)", 0.0, 500.0, 0.0, 1.0)
        th = b.number_input("Opening theaters", 0, 5000, 3500, 50)
        w2 = c.number_input("2nd weekend ($M)", 0.0, 500.0, 0.0, 1.0)
        w3 = e.number_input("3rd weekend ($M)", 0.0, 500.0, 0.0, 1.0)
        submitted = st.form_submit_button("Refine forecast")
    if submitted and w1 > 0:
        w2v = w2 * 1e6 if w2 > 0 else None
        w3v = w3 * 1e6 if (w3 > 0 and w2v) else None
        r = predict_post(w1 * 1e6, max(th, 1000), rel, w2v, w3v, int(f["studio"] != "Other"))
        err = metrics()["post_release"][f"after_weekend_{r['k']}"]["median_abs_pct_error"]
        st.html(f"""<div class="bc-figs">
          <div class="bc-fig"><div class="k">US and Canada total, refined</div>
            <div class="v">{money(r['mid'])}</div>
            <div class="s">{money(r['low'])} to {money(r['high'])}</div></div>
          <div class="bc-fig"><div class="k">Multiplier</div>
            <div class="v">{r['multiplier']:.2f}×</div><div class="s">lifetime ÷ opening weekend</div></div>
          <div class="bc-fig"><div class="k">Typical error at this stage</div>
            <div class="v">±{err:.0f}%</div><div class="s">on films the model never saw</div></div>
        </div>""")
    elif submitted:
        note("Enter the opening weekend gross to refine the forecast.")


# ------------------------------------------------------------------ about
def about():
    M = metrics()
    pre, us, post = M["pre_release"], M["pre_release_us"], M["post_release"]
    stages = [("Before release", pre["median_abs_pct_error"], pre["baseline_budget_only_median_abs_pct_error"])]
    stages += [(f"After weekend {k}", post[f"after_weekend_{k}"]["median_abs_pct_error"],
                post[f"after_weekend_{k}"]["baseline_typical_multiplier_median_abs_pct_error"]) for k in (1, 2, 3)]
    top = max(max(s[1], s[2]) for s in stages)
    rows = "".join(f"""<div class="row"><div>{s}</div>
        <div><div class="bar" style="width:{v / top * 100:.0f}%" title="BoxCast: {v}%"></div>
        <div class="base" style="width:{b / top * 100:.0f}%" title="Rule of thumb: {b}%"></div></div>
        <div style="text-align:right;font-weight:650">{v:.0f}%</div></div>""" for s, v, b in stages)
    st.html(f"""
    <div class="bc-hero"><h1>How BoxCast works</h1>
    <p>Two kinds of forecast: one before a film opens, and a sharper one once its
    weekend grosses come in.</p></div>
    <div class="bc-section">Typical forecast error on films the model never saw</div>
    <div class="bc-sub">Thick bar: BoxCast. Thin bar: a simple rule of thumb (budget alone before
    release; a typical multiple of the opening weekend after). Lower is better.</div>
    <div class="bc-funnel">{rows}</div>
    <div class="bc-prose">
      <h3>Before release</h3>
      <p>The forecast learns from {len(load("pre_release.csv")):,} US releases from 1990 to 2019: budget,
      genre, age rating, runtime, release timing, studio, whether it's a sequel, and the past
      box office of the lead star and director. It predicts worldwide gross, plus the US
      opening weekend (typical error {us['opening_weekend']['median_abs_pct_error']:.0f}%) and
      US total ({us['domestic_total']['median_abs_pct_error']:.0f}%).</p>
      <p>Before a film opens nobody knows whether audiences will love it, so every forecast
      comes with a range. That range contained the real result about 80% of the time in
      testing.</p>
      <h3>After it opens</h3>
      <p>Using daily box office for 2,800 wide releases since 2000, BoxCast learns how many times
      its opening weekend a film goes on to earn. The drop from the first to the second weekend
      is the strongest signal of word of mouth. After two weekends, 93% of forecasts landed
      within 25% of the final total.</p>
      <h3>The technique</h3>
      <p>Gradient-boosted decision trees (LightGBM) trained to predict a low, likely and high
      outcome, with the range calibrated on recent films so it stays honest. "Why this number"
      breaks each forecast into the contribution of each factor.</p>
      <h3>Limits</h3>
      <p>Figures aren't adjusted for inflation, the pre-release training data ends in 2019,
      and marketing spend, trailer buzz and reviews aren't included yet.</p>
    </div>""")


# ------------------------------------------------------------------ router
film = st.query_params.get("film")
page = st.query_params.get("page", "home")
top_bar("about" if page == "about" else "home")
if film:
    film_page(int(film))
elif page == "about":
    about()
else:
    home()
footer()
