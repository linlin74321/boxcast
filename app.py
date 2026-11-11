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
from predict import (BREAK_EVEN_MULTIPLE, apply_sequel_anchor, break_even_chance,  # noqa: E402
                     explain, metrics, predict_pre_all, project_in_release)
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
                  "sub": nice_date(r.release_date),
                  "fig": f"{money(r.pred_mid)} {getattr(r, 'fig_label', 'forecast')}"}
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


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def previous_film(movie_id, key):
    api = TMDB(key)
    return api.previous_installment(api.details(movie_id))


def verdict_html(chance):
    if chance is None:
        return '<div class="v">—</div><div class="s">No budget listed on TMDB</div>'
    cls, word, icon = (("bc-good", "Likely profitable", "✓") if chance >= 0.6 else
                       ("bc-warn", "Could go either way", "≈") if chance >= 0.35 else
                       ("bc-risk", "Risky", "!"))
    shown = ">99%" if chance > 0.99 else "<1%" if chance < 0.01 else f"{chance * 100:.0f}%"
    return (f'<div class="v">{shown}</div>'
            f'<div class="s"><span class="{cls}">{icon} {word}</span></div>')


def fig(k, v, sub=""):
    return f'<div class="bc-fig"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{sub}</div></div>'


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
    try:
        prev = previous_film(movie_id, KEY)
    except Exception:
        prev = None
    if prev:
        f["log_prev_gross"] = math.log(prev["revenue"])
        f["franchise_n"] = prev.get("count", 1)
        f["is_sequel"] = 1
    row = pd.DataFrame([f])
    p = predict_pre_all(row).iloc[0]
    base, drivers = explain(row)
    if prev:
        p = apply_sequel_anchor(p, prev["revenue"])
    budget = f["budget"] if pd.notna(f["budget"]) else None
    be = BREAK_EVEN_MULTIPLE * budget if budget else None

    # ---- which situation is the film in?
    rel = pd.Timestamp(f["release_date"])
    days_out = (pd.Timestamp(date.today()) - rel).days
    revenue = f["revenue"] if pd.notna(f["revenue"]) and f["revenue"] > 0 else None
    if revenue and days_out >= 100:
        mode = "final"
    elif revenue and days_out >= 0:
        mode = "running"
    else:
        mode = "forecast"

    # ---- header with backdrop
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

    # ---- the headline number
    if mode == "final":
        ratio = revenue / budget if budget else None
        st.html(f"""<div class="bc-fc">
          <div class="bc-fc-k">Final worldwide box office</div>
          <div class="bc-fc-n">{money(revenue)}</div>
          <div class="bc-fc-r">Released {nice_date(rel)}. This film has finished its cinema run.</div>
        </div>""")
        if ratio:
            cls, word = (("bc-good", "✓ Profitable") if ratio >= BREAK_EVEN_MULTIPLE else
                         ("bc-warn", "≈ Roughly broke even") if ratio >= 1.8 else
                         ("bc-risk", "! Likely lost money"))
            verdict = fig("Result", f"{ratio:.1f}×", f'<span class="{cls}">{word}</span> on a {money(budget)} budget')
        else:
            verdict = fig("Result", "—", "No budget listed on TMDB")
        off = (p["pred_mid"] - revenue) / revenue * 100
        st.html(f"""<div class="bc-figs">
          {fig("Budget", money(budget), "Production budget, per TMDB")}
          {verdict}
          {fig("BoxCast's pre-release forecast", money(p['pred_mid']),
               f"{abs(off):.0f}% {'above' if off > 0 else 'below'} the final result")}
        </div>""")
        if rel.year <= 2019:
            note("This film is part of the data the model learned from, so its pre-release forecast "
                 "is a sanity check rather than a fair test.")
    elif mode == "running":
        proj = project_in_release(revenue, days_out)
        chance = break_even_chance(proj["low"], proj["mid"], proj["high"], budget)
        st.html(f"""<div class="bc-fc">
          <div class="bc-fc-k">Projected final worldwide box office</div>
          <div class="bc-fc-n">{money(proj['mid'])}</div>
          <div class="bc-fc-r">Likely between {money(proj['low'])} and {money(proj['high'])}</div>
          {track_html(proj['low'], proj['mid'], proj['high'], be)}
        </div>""")
        st.html(f"""<div class="bc-figs">
          {fig("Earned so far", money(revenue), f"Day {days_out + 1} in cinemas, per TMDB")}
          {fig("Pre-release forecast", money(p['pred_mid']), "Before any tickets were sold")}
          <div class="bc-fig"><div class="k">Chance to break even</div>{verdict_html(chance)}</div>
        </div>""")
        note("In cinemas now. The projection uses the latest worldwide total reported on TMDB and "
             "how films typically earn over their run. It sharpens every week.")
    else:
        chance = break_even_chance(p["pred_low"], p["pred_mid"], p["pred_high"], budget)
        st.html(f"""<div class="bc-fc">
          <div class="bc-fc-k">Forecast worldwide box office</div>
          <div class="bc-fc-n">{money(p['pred_mid'])}</div>
          <div class="bc-fc-r">Likely between {money(p['pred_low'])} and {money(p['pred_high'])}</div>
          {track_html(p['pred_low'], p['pred_mid'], p['pred_high'], be)}
        </div>""")
        st.html(f"""<div class="bc-figs">
          {fig("US opening weekend", money(p['open_mid']), f"{money(p['open_low'])} to {money(p['open_high'])}")}
          {fig("US and Canada total", money(p['dom_mid']), f"{money(p['dom_low'])} to {money(p['dom_high'])}")}
          <div class="bc-fig"><div class="k">Chance to break even</div>{verdict_html(chance)}</div>
        </div>""")
        if days_out >= 0:
            note(f"In cinemas since {nice_date(rel)}, but TMDB hasn't reported its box office yet. "
                 "Until it does, this is the pre-release forecast.")
        else:
            note(f"Opens in US cinemas on {nice_date(rel)}. Pre-release forecasts are the hardest kind; "
                 "the range shows how far the real result could reasonably land.")
        if not budget:
            note("TMDB doesn't list a budget for this film, so the forecast leans on its genre, cast, "
                 "director and timing, and the range is wider.")

    # ---- why
    prev_line = ""
    if prev and not any(d[0] == "Previous film in the series" for d in drivers):
        prev_line = (f'<div class="bc-drv"><div class="n">Previous film in the series</div>'
                     f'<div class="b"></div><div class="v">{money(prev["revenue"])}</div></div>'
                     f'<div class="bc-sub" style="margin:.4rem 0 0">{esc(prev["title"])} ({prev["year"]}). '
                     f'The forecast takes the series\u2019 track record into account.</div>')
    prev_note = ""
    if prev and not prev_line:
        prev_note = (f'<div class="bc-sub" style="margin:.6rem 0 0">Previous film in the series: '
                     f'{esc(prev["title"])} ({prev["year"]}), {money(prev["revenue"])} worldwide.</div>')
    heading = "Why this number" if mode == "forecast" else "What the pre-release model saw"
    st.html(f'<div class="bc-section">{heading}</div>'
            f'<div class="bc-sub">A typical film in our data earns {money(base)} worldwide. '
            f'Here\u2019s what moves this one up or down from there.</div>{drivers_html(drivers)}{prev_line}{prev_note}')

    # ---- comparables
    st.html('<div class="bc-section">Films like this</div>'
            '<div class="bc-sub">Same genre, closest budget. Worldwide lifetime gross.</div>'
            + comparables(f, p))


# ------------------------------------------------------------------ about
def about():
    M = metrics()
    pre, us, live = M["pre_release"], M["pre_release_us"], M["in_release_projection"]
    e = live["median_abs_pct_error"]
    stages = [("Before release", pre["median_abs_pct_error"]),
              ("After opening weekend", e["after_opening_weekend"]),
              ("After 1 week", e["after_week_1"]),
              ("After 2 weeks", e["after_week_2"]),
              ("After 3 weeks", e["after_week_3"])]
    top = max(v for _, v in stages)
    rows = "".join(f"""<div class="row"><div>{s}</div>
        <div><div class="bar" style="width:{v / top * 100:.0f}%" title="{s}: {v}% typical error"></div></div>
        <div style="text-align:right;font-weight:650">{v:.0f}%</div></div>""" for s, v in stages)
    st.html(f"""
    <div class="bc-hero"><h1>How BoxCast works</h1>
    <p>Search a film and BoxCast picks the right method for where it is in its life:
    a forecast before release, a projection while it's in cinemas, and the real total
    once it has finished.</p></div>
    <div class="bc-section">Typical forecast error on films the model never saw</div>
    <div class="bc-sub">Median gap between the forecast and the final total. Lower is better.
    The number tightens fast once real ticket sales exist.</div>
    <div class="bc-funnel">{rows}</div>
    <div class="bc-prose">
      <h3>Before release</h3>
      <p>The forecast learns from {len(load("pre_release.csv")):,} US releases from 1990 to 2019:
      budget, genre, age rating, runtime, release timing, studio, whether it's a sequel, and the
      past box office of the lead star and director. For sequels it also leans on what the
      previous film in the series earned. It predicts worldwide gross, plus the US opening
      weekend (typical error {us['opening_weekend']['median_abs_pct_error']:.0f}%) and US total
      ({us['domestic_total']['median_abs_pct_error']:.0f}%).</p>
      <p>Before a film opens nobody knows whether audiences will love it, so this is the hardest
      forecast to make. Every forecast comes with a range, and in testing that range contained
      the real result about 80% of the time.</p>
      <h3>While it's in cinemas</h3>
      <p>BoxCast takes the latest worldwide total reported on TMDB and projects the final number
      from how 2,800 wide releases since 2000 typically earned over
      their run. No numbers to enter. After three weeks the typical error is about
      {e['after_week_3']:.0f}%.</p>
      <h3>After it's finished</h3>
      <p>You see the actual worldwide total, its return on budget, and how BoxCast's
      pre-release forecast compared.</p>
      <h3>The technique</h3>
      <p>Gradient-boosted decision trees (LightGBM) trained to predict a low, likely and high
      outcome, with the range calibrated on recent films so it stays honest. "Why this number"
      breaks each forecast into the contribution of each factor.</p>
      <h3>Limits</h3>
      <p>Figures aren't adjusted for inflation, the pre-release training data ends in 2019,
      international hits driven by one market (such as China) are hard to foresee, and
      marketing spend, trailer buzz and reviews aren't included yet. Box office totals come
      from TMDB and can lag by a few days.</p>
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
