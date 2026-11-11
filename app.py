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
from buzz import live_buzz, wiki_title  # noqa: E402
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
def competition(release_date, movie_id, key):
    return TMDB(key).big_releases_near(release_date, exclude_id=movie_id)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def wiki_buzz(wikidata_id, release_date, today):
    title = wiki_title(wikidata_id)
    views, days_before = live_buzz(title, release_date, today)
    return views, days_before


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
    try:
        f["competition_big"] = competition(str(f["release_date"]), movie_id, KEY)
    except Exception:
        pass
    buzz_views = None
    try:
        buzz_views, days_before = wiki_buzz(f.get("wikidata_id"), str(f["release_date"]),
                                            str(date.today()))
        if buzz_views is not None:
            f["log_wiki_views"] = math.log1p(buzz_views)
            f["wiki_days_before"] = days_before
    except Exception:
        pass
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
    signals = []
    if buzz_views is not None:
        signals.append(f"Wikipedia page views in the 4 weeks before {'release' if days_out >= 0 else 'today'}: "
                       f"<b>{buzz_views:,.0f}</b>")
    if pd.notna(f.get("competition_big")):
        n = int(f["competition_big"])
        signals.append(f"Other $50M+ films opening within a week: <b>{n}</b>")
    signals.append(f"Cinema release in China: <b>{'yes' if f.get('china_release') else 'not announced'}</b>")
    prev_note += ('<div class="bc-sub" style="margin:.8rem 0 0">' + "<br>".join(signals) + "</div>")
    heading = "Why this number" if mode == "forecast" else "What the pre-release model saw"
    st.html(f'<div class="bc-section">{heading}</div>'
            f'<div class="bc-sub">A typical film in our data earns {money(base)} worldwide. '
            f'Here\u2019s what moves this one up or down from there.</div>{drivers_html(drivers)}{prev_line}{prev_note}')

    # ---- comparables
    st.html('<div class="bc-section">Films like this</div>'
            '<div class="bc-sub">Same genre, closest budget. Worldwide lifetime gross.</div>'
            + comparables(f, p))


# ------------------------------------------------------------------ about
def hbar_rows(items, max_value, label_fmt="{:.0f}%"):
    """Horizontal bars: items = [(label, value, colour)]. Direct-labelled, with hover text."""
    return "".join(
        f'<div class="bc-hbar"><div class="t">{esc(t)}</div>'
        f'<div class="bar-wrap"><div class="bar" style="width:{max(v / max_value * 100, 1.5):.1f}%;'
        f'background:{c}" title="{esc(t)}: {label_fmt.format(v)}"></div></div>'
        f'<div class="v">{label_fmt.format(v)}</div></div>' for t, v, c in items)


def flow_html(used):
    """How a forecast is made: the input columns, the models, the outputs."""
    groups = [("Film basics", ["Production budget", "Genre", "Age rating", "Runtime", "Studio",
                               "Release month and year"], True),
              ("People", ["Lead star's past box office", "Director's past box office",
                          "How many films each has made"], True),
              ("Franchise", ["Previous film in the series: its worldwide gross",
                             "How many films came before"], "log_prev_gross" in used),
              ("Market signals", [x for x, f in (("Wikipedia page views before release", "log_wiki_views"),
                                                 ("Opening in China", "china_release"),
                                                 ("Big films opening the same week", "competition_big"))
                                  if f in used], any(f in used for f in ("log_wiki_views", "china_release",
                                                                        "competition_big")))]
    cols = "".join(f'<div class="bc-col"><h4>{g}</h4><ul>' + "".join(f"<li>{esc(i)}</li>" for i in items)
                   + "</ul></div>" for g, items, show in groups if show)
    return f"""<div class="bc-flow">
      <div class="bc-cols">{cols}</div>
      <div class="bc-arrow" aria-hidden="true">↓</div>
      <div class="bc-stage"><b>Three gradient-boosted tree models</b> learn from every column at once:
        one for a low case, one for the likely outcome, one for a high case
        <small>trained on {len(load("pre_release.csv")):,} films; each split asks a question like
        "budget over $90M?" and thousands of small trees add up</small></div>
      <div class="bc-arrow" aria-hidden="true">↓</div>
      <div class="bc-stage"><b>Range calibration</b>: widened until it caught the real result about
        80% of the time on recent films<small>so the range is honest, not optimistic</small></div>
      <div class="bc-arrow" aria-hidden="true">↓</div>
      <div class="bc-outs"><div class="bc-out">Worldwide total</div><div class="bc-out">US opening weekend</div>
        <div class="bc-out">US and Canada total</div><div class="bc-out">Chance to break even</div></div>
      <div class="bc-life">
        <div><b><i>1</i>Before release</b><span>The forecast above, from the columns alone.</span></div>
        <div><b><i>2</i>In cinemas</b><span>Earnings so far (from TMDB) ÷ the share a typical film
          has earned by that day = projected final total.</span></div>
        <div><b><i>3</i>Finished</b><span>The actual total, its return on budget, and how the
          forecast compared.</span></div>
      </div></div>"""


def example_film(name):
    """A real film from the test: what the model saw, and how it did."""
    test, pre = load("test_predictions_pre.csv"), load("pre_release.csv")
    t = test[test["name"] == name].iloc[0]
    rows = pre[(pre["name"] == t["name"]) & (pre["year"] == t["year"])]
    if rows.empty:
        return ""
    r = rows.iloc[0]
    def val(v, fmt=str):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else fmt(v)
    items = [("Budget", val(r.get("budget"), money)), ("Genre / age rating", f"{r['genre']} / {r['rating']}"),
             ("Runtime", val(r.get("runtime"), lambda x: f"{int(x)} min")), ("Studio", r["studio"]),
             ("Release", nice_date(r["release_date"])),
             ("Lead star's past films (avg)", val(r.get("star_power"), lambda x: money(math.exp(x)))),
             ("Director's past films (avg)", val(r.get("director_power"), lambda x: money(math.exp(x)))),
             ("Previous film in the series", val(r.get("prev_gross"), money)),
             ("Opening in China", {1: "Yes", 0: "No"}.get(r.get("china_release"), "—")),
             ("Big films opening the same week", val(r.get("competition_big"), lambda x: str(int(x)))),
             ("Wikipedia views, 4 weeks to 30 days out", val(r.get("wiki_30"), lambda x: f"{x:,.0f}"))]
    inside = t["pred_low"] <= t["gross"] <= t["pred_high"]
    off = (t["pred_mid"] - t["gross"]) / t["gross"] * 100
    table = "".join(f"<tr><td>{esc(k)}</td><td>{esc(str(v))}</td></tr>" for k, v in items)
    return f"""<table class="bc-kv">{table}</table>
      <div class="bc-figs">
        {fig("Forecast before release", money(t['pred_mid']), f"{money(t['pred_low'])} to {money(t['pred_high'])}")}
        {fig("Actual worldwide", money(t['gross']), "")}
        {fig("Result", f"{abs(off):.0f}% {'over' if off > 0 else 'under'}",
             '<span class="bc-good">✓ inside the range</span>' if inside else
             '<span class="bc-risk">! outside the range</span>')}
      </div>"""


def example_picker():
    """Let the reader pick any film from the test, hits and misses alike."""
    test = load("test_predictions_pre.csv")
    if test is None or test["year"].max() < 2022:
        return
    test = test.sort_values("gross", ascending=False).drop_duplicates("name")
    films = test["name"].tolist()
    # Start on the biggest film whose result landed inside its range (the typical case)
    inside = test[(test["gross"] >= test["pred_low"]) & (test["gross"] <= test["pred_high"])]
    default = inside["name"].iloc[0] if len(inside) else films[0]
    st.html('<div class="bc-section">One film, start to finish</div>'
            '<div class="bc-sub">Pick any of the films the model was tested on, hits and misses alike. '
            'A model trained only on films up to 2021 saw these columns and nothing else.</div>')
    name = st.selectbox("Film from the test", films, index=films.index(default),
                        label_visibility="collapsed", key="example_film")
    st.html(example_film(name))


def scatter_chart():
    """Forecast vs actual for every unseen test film."""
    import plotly.graph_objects as go
    test = load("test_predictions_pre.csv")
    if test is None:
        return None
    lo, hi = max(test["gross"].min(), 1e5) * 0.6, test["gross"].max() * 1.6
    xs = np.geomspace(lo, hi, 50)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[xs * 1.5, (xs * 0.5)[::-1]], fill="toself",
                             fillcolor="rgba(238,241,246,0.06)", line=dict(width=0), hoverinfo="skip",
                             name="Within ±50%"))
    fig.add_trace(go.Scatter(x=xs, y=xs, mode="lines", line=dict(color="#8C97AB", width=1.5, dash="dot"),
                             hoverinfo="skip", name="Perfect forecast"))
    fig.add_trace(go.Scatter(
        x=test["gross"], y=test["pred_mid"], mode="markers", name="A film",
        marker=dict(color="#B38A14", size=8, opacity=0.75, line=dict(color="#141A26", width=1)),
        customdata=np.c_[test["name"], test["year"]],
        hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>Actual $%{x:,.0f}"
                      "<br>Forecast $%{y:,.0f}<extra></extra>"))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Archivo, sans-serif", color="#C4CBD8", size=13),
                      legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor="#232D40", bordercolor="#232D40", font=dict(color="#EEF1F6")))
    ticks = [v for v in (1e5, 1e6, 1e7, 1e8, 1e9, 1e10) if lo <= v <= hi]
    for ax, title in ((fig.update_xaxes, "Actual worldwide gross"), (fig.update_yaxes, "Forecast before release")):
        ax(type="log", title=title, gridcolor="rgba(238,241,246,0.07)", zeroline=False,
           tickvals=ticks, ticktext=[money(v).replace(".00", "").replace(".0", "") for v in ticks],
           range=[math.log10(lo), math.log10(hi)])
    return fig


def about():
    from predict import DRIVER_GROUPS, model_features
    M = metrics()
    used = set(model_features("pre"))
    pre, us, live = M["pre_release"], M["pre_release_us"], M["in_release_projection"]
    comp = pre.get("feature_comparison") or {}
    e = live["median_abs_pct_error"]
    gold, grey, grey2 = "var(--up)", "#4A5568", "#6B7690"

    st.html(f"""<div class="bc-hero"><h1>How BoxCast works</h1>
      <p>Every forecast is built from the same set of columns about the film, run through models
      that learned from {len(load("pre_release.csv")):,} past releases. Here's the whole journey,
      and how well it works on films it had never seen.</p></div>
      <div class="bc-section">How a forecast is made</div>
      <div class="bc-sub">The columns on top are everything the model knows about a film.</div>
      {flow_html(used)}""")

    example_picker()

    # ---- accuracy: head-to-head
    versions = [("Rule of thumb: budget alone", pre["baseline_budget_only_median_abs_pct_error"], grey)]
    if comp:
        versions.append(("Core model: basics, people, franchise", comp["without_new_signals"], grey2))
        versions.append(("BoxCast with market signals", comp["with_new_signals"], gold))
    else:
        versions.append(("BoxCast", pre["median_abs_pct_error"], gold))
    st.html(f"""<div class="bc-section">How accurate is the forecast before release?</div>
      <div class="bc-sub">Typical gap between forecast and final worldwide total, on
      {pre['films_tested']:,} {pre.get('test_period', 'unseen films').split('(')[0].strip()}. Lower is better.</div>
      {hbar_rows(versions, max(v for _, v, _ in versions) * 1.05)}
      <div class="bc-sub" style="margin-top:.6rem">{pre['within_50pct']:.0f}% of forecasts landed within ±50%
      of the real total, and the range caught the real result {pre['range_coverage']:.0f}% of the time.</div>""")

    fig_ = scatter_chart()
    if fig_ is not None:
        st.html('<div class="bc-section">Every forecast against what really happened</div>'
                '<div class="bc-sub">One dot per film. On the dotted line means a perfect forecast; '
                'the shaded band is within ±50%. Hover a dot to see the film.</div>')
        st.plotly_chart(fig_, config={"displayModeBar": False})

    # ---- funnel
    stages = [("Before release", pre["median_abs_pct_error"]), ("After opening weekend", e["after_opening_weekend"]),
              ("After 1 week", e["after_week_1"]), ("After 2 weeks", e["after_week_2"]),
              ("After 3 weeks", e["after_week_3"])]
    st.html(f"""<div class="bc-section">The forecast sharpens once tickets are sold</div>
      <div class="bc-sub">Typical error at each stage of a film's life. In cinemas, BoxCast projects the
      final total from the latest earnings on TMDB.</div>
      {hbar_rows([(s_, v, gold) for s_, v in stages], max(v for _, v in stages) * 1.05)}""")

    # ---- drivers
    imp = M.get("importance", {}).get("pre_release", {})
    grouped = [(name, sum(imp.get(f, 0) for f in feats) * 100) for name, feats in DRIVER_GROUPS.items()]
    grouped = sorted([g for g in grouped if g[1] > 0.5], key=lambda g: -g[1])
    if grouped:
        st.html('<div class="bc-section">What the model leans on most</div>'
                '<div class="bc-sub">Share of the model\u2019s decisions that use each column group.</div>'
                + hbar_rows([(n, v, gold) for n, v in grouped], grouped[0][1] * 1.05))

    st.html(f"""<div class="bc-prose">
      <h3>Limits</h3>
      <p>Figures aren't adjusted for inflation, surprise hits in a single market are hard to foresee,
      and marketing spend and reviews aren't included yet. Box office totals come from TMDB and can
      lag by a few days. US opening weekend forecasts have a typical error of
      {us['opening_weekend']['median_abs_pct_error']:.0f}%, and the US total
      {us['domestic_total']['median_abs_pct_error']:.0f}%.</p></div>""")


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
