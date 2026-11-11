"""
BoxCast - Hollywood box office forecaster (Streamlit app).

Run locally with:   streamlit run app.py
On Streamlit Community Cloud, point the app at this file.
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from features import GENRES, RATINGS, STUDIOS  # noqa: E402
from predict import metrics, person_power, predict_post, predict_pre  # noqa: E402

DATA = ROOT / "data"
GOLD, TEAL, GREY = "#F5B83D", "#3FB6A8", "#8A8F98"

st.set_page_config(page_title="BoxCast · Box Office Forecaster",
                   page_icon="🎬", layout="wide")


# ---------------------------------------------------------------- helpers
def money(x):
    if x is None or pd.isna(x):
        return "—"
    if abs(x) >= 1e9:
        return f"${x / 1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"${x / 1e6:.1f}M"
    return f"${x / 1e3:.0f}K"


@st.cache_data
def _read(name, modified, kw):
    return pd.read_csv(DATA / name, **dict(kw))


def load(name, **kw):
    """Read a data file. The cache refreshes automatically whenever the file
    changes, so the daily update shows up right away."""
    path = DATA / name
    if not path.exists():
        return None
    return _read(name, path.stat().st_mtime, tuple(sorted(kw.items())))


def style_fig(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=13), legend=dict(orientation="h", y=-0.15))
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.15)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)")
    return fig


pre_hist = load("pre_release.csv", parse_dates=["release_date"])
post_hist = load("post_release.csv", parse_dates=["release_date"])
curves = load("curves.csv")
test_post = load("test_predictions_post.csv", parse_dates=["release_date"])
test_pre = load("test_predictions_pre.csv")
live = load("live_forecasts.csv", parse_dates=["release_date"])
M = metrics()

# ---------------------------------------------------------------- header
st.title("🎬 BoxCast")
st.caption("Forecasts a Hollywood film's lifetime box office before release, "
           "then sharpens the forecast as each weekend's numbers come in.")

tabs = st.tabs(["Upcoming releases", "Pre-release forecast", "Post-release tracker",
                "Replay a real film", "Model accuracy", "How it works"])

# ================================================================ 1. LIVE
with tabs[0]:
    if live is None or live.empty:
        st.info("No live data yet. The daily update fills this tab automatically "
                "once your TMDB key is added to GitHub and the **Daily update** "
                "workflow has run once. See the README, step 5.")
    else:
        st.subheader("Now playing and coming soon in US cinemas")
        st.caption(f"Pulled from TMDB and forecast automatically · last updated "
                   f"{live['updated_at'].iloc[0]} · worldwide lifetime gross")
        c1, c2 = st.columns([1, 3])
        status = c1.radio("Show", ["All", "Coming soon", "Now playing"], horizontal=False)
        min_pop = c1.slider("Hide obscure titles (TMDB popularity)", 0, 50, 5)
        view = live[live["popularity"].fillna(0) >= min_pop]
        if status != "All":
            view = view[view["status"] == status]
        view = view.sort_values("pred_mid", ascending=False)

        top = view.head(15).iloc[::-1]
        fig = go.Figure()
        fig.add_trace(go.Bar(y=top["title"], x=top["pred_high"] - top["pred_low"],
                             base=top["pred_low"], orientation="h",
                             marker_color="rgba(245,184,61,0.35)", name="Likely range (80%)",
                             hovertemplate="%{y}<extra></extra>"))
        fig.add_trace(go.Scatter(y=top["title"], x=top["pred_mid"], mode="markers",
                                 marker=dict(color=GOLD, size=11, symbol="diamond"),
                                 name="Best estimate",
                                 hovertemplate="%{y}: $%{x:,.0f}<extra></extra>"))
        fig.update_xaxes(type="log", title="Forecast worldwide gross (USD, log scale)")
        c2.plotly_chart(style_fig(fig, 520))

        table = view[["title", "status", "release_date", "genre", "star", "director",
                      "budget", "pred_low", "pred_mid", "pred_high"]].copy()
        table["release_date"] = table["release_date"].dt.date
        table[["star", "director"]] = table[["star", "director"]].fillna("—")
        for col in ["budget", "pred_low", "pred_mid", "pred_high"]:
            table[col] = table[col].map(money)
        table.columns = ["Title", "Status", "US release", "Genre", "Lead star", "Director",
                         "Budget", "Low", "Best estimate", "High"]
        st.dataframe(table, hide_index=True)

        hist = load("forecast_history.csv")
        if hist is not None and hist["forecast_date"].nunique() > 1:
            st.subheader("How forecasts have changed")
            pick = st.selectbox("Film", view["title"].tolist())
            h = hist[hist["title"] == pick]
            fig = go.Figure(go.Scatter(x=h["forecast_date"], y=h["pred_mid"],
                                       mode="lines+markers", line_color=GOLD))
            st.plotly_chart(style_fig(fig, 300))

# ================================================================ 2. PRE
with tabs[1]:
    st.subheader("What will this film make worldwide?")
    st.caption("Enter what's known before release. Based on 5,000+ US releases (1990–2019).")
    stars = sorted(pre_hist["star"].dropna().unique())
    directors = sorted(pre_hist["director"].dropna().unique())

    with st.form("pre"):
        a, b, c = st.columns(3)
        title = a.text_input("Film title", "My Upcoming Blockbuster")
        budget_known = a.checkbox("Budget is known", value=True)
        budget_m = a.number_input("Production budget (USD millions)", 1.0, 500.0, 120.0, 5.0)
        genre = b.selectbox("Main genre", GENRES, index=0,
                            help="Pick the alphabetically first of its genres "
                                 "(that's how the training data stores it), "
                                 "e.g. an 'Action, Sci-Fi' film is Action.")
        rating = b.selectbox("Age rating", RATINGS, index=2)
        runtime = b.slider("Runtime (minutes)", 70, 210, 125)
        rel = c.date_input("Release date", date(2026, 12, 18))
        studio = c.selectbox("Studio", STUDIOS, index=0)
        sequel = c.checkbox("Sequel / franchise film", value=True)
        d, e = st.columns(2)
        star = d.selectbox("Lead star", ["(new / not listed)"] + stars,
                           index=(stars.index("Tom Cruise") + 1) if "Tom Cruise" in stars else 0)
        director = e.selectbox("Director", ["(new / not listed)"] + directors, index=0)
        go_btn = st.form_submit_button("Forecast", type="primary")

    s_pow, s_n = person_power(star, "star")
    d_pow, d_n = person_power(director, "director")
    row = pd.DataFrame([{
        "log_budget": np.log(budget_m * 1e6) if budget_known else np.nan,
        "runtime": runtime, "month": rel.month, "year": rel.year,
        "is_sequel": int(sequel), "star_power": s_pow, "star_films": s_n,
        "director_power": d_pow, "director_films": d_n,
        "genre": genre, "rating": rating, "studio": studio}])
    p = predict_pre(row).iloc[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Best estimate", money(p["pred_mid"]))
    m2.metric("Low case (10%)", money(p["pred_low"]))
    m3.metric("High case (90%)", money(p["pred_high"]))
    if budget_known:
        st.caption(f"Best estimate is **{p['pred_mid'] / (budget_m * 1e6):.1f}×** the budget. "
                   "Rule of thumb: a film needs roughly 2.5× its budget worldwide to break even.")
    st.warning(f"Pre-release forecasting is genuinely hard: on films it had never seen, the "
               f"typical error of the best estimate was {M['pre_release']['median_abs_pct_error']}%. "
               "That's why the range matters. Use the **Post-release tracker** once the "
               "opening weekend is known; accuracy jumps sharply.", icon="⚠️")

    if budget_known:
        sim = pre_hist[(pre_hist["genre"] == genre) & pre_hist["budget"].notna()].copy()
        sim["gap"] = (np.log(sim["budget"]) - np.log(budget_m * 1e6)).abs()
        sim = sim.nsmallest(8, "gap").sort_values("gross", ascending=False)
        st.markdown("**Similar past films (same genre, closest budget)**")
        st.dataframe(pd.DataFrame({
            "Film": sim["name"], "Year": sim["year"], "Lead star": sim["star"],
            "Budget": sim["budget"].map(money), "Worldwide gross": sim["gross"].map(money),
            "× budget": (sim["gross"] / sim["budget"]).round(1)}), hide_index=True)

# ================================================================ 3. POST
with tabs[2]:
    st.subheader("Opening weekend is in. Where will it finish?")
    st.caption("Domestic (US + Canada) lifetime gross. Add the 2nd and 3rd weekend "
               "when known and watch the range tighten.")
    with st.form("post"):
        a, b, c = st.columns(3)
        name = a.text_input("Film", "New Release")
        rel2 = a.date_input("Release date", date(2026, 11, 6), key="rel2")
        w1 = b.number_input("Opening weekend, Fri–Sun (USD millions)", 0.5, 400.0, 45.0, 1.0)
        theaters = b.number_input("Opening theaters", 1000, 4800, 3800, 50)
        w2 = c.number_input("2nd weekend (USD millions, 0 = not yet)", 0.0, 400.0, 0.0, 1.0)
        w3 = c.number_input("3rd weekend (USD millions, 0 = not yet)", 0.0, 400.0, 0.0, 1.0)
        major = a.checkbox("Released by a major studio", value=True)
        st.form_submit_button("Update forecast", type="primary")

    w2v = w2 * 1e6 if w2 > 0 else None
    w3v = w3 * 1e6 if (w3 > 0 and w2v) else None
    r = predict_post(w1 * 1e6, theaters, rel2, w2v, w3v, major)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Lifetime best estimate", money(r["mid"]))
    m2.metric("Likely range", f"{money(r['low'])} – {money(r['high'])}")
    m3.metric("Multiplier", f"{r['multiplier']:.2f}× opening")
    err = M["post_release"][f"after_weekend_{r['k']}"]["median_abs_pct_error"]
    m4.metric("Typical error at this stage", f"±{err}%")
    if w2v:
        st.caption(f"2nd-weekend drop: **{(1 - w2v / (w1 * 1e6)) * 100:.0f}%** "
                   "(under 50% = good word of mouth, over 60% = front-loaded)")

    # Trajectory: shape borrowed from the 40 past films with the most similar
    # multiplier ("analog forecasting").
    ph = post_hist[(post_hist["preview_days"] == (4 - pd.Timestamp(rel2).dayofweek) % 7)].copy()
    ph["gap"] = (np.log(ph["multiplier"]) - np.log(r["multiplier"])).abs()
    analogs = ph.nsmallest(40, "gap")
    shape = curves[curves["film_id"].isin(analogs["film_id"])].iloc[:, 1:].median().values
    shape = shape / shape[-1]
    weeks = np.arange(1, 21)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[weeks, weeks[::-1]],
                             y=np.r_[shape * r["high"], (shape * r["low"])[::-1]],
                             fill="toself", fillcolor="rgba(245,184,61,0.18)",
                             line=dict(width=0), name="Likely range", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=weeks, y=shape * r["mid"], mode="lines", line=dict(color=GOLD, width=3),
                             name="Projected total so far",
                             hovertemplate="Week %{x}: $%{y:,.0f}<extra></extra>"))
    known = np.cumsum([v for v in (w1 * 1e6, w2v, w3v) if v])
    fig.add_trace(go.Scatter(x=np.arange(1, len(known) + 1), y=known, mode="markers",
                             marker=dict(color=TEAL, size=10),
                             name="Weekend grosses entered (cumulative)"))
    fig.update_layout(title=f"{name}: projected cumulative domestic gross")
    fig.update_xaxes(title="Week in release")
    fig.update_yaxes(title="USD", rangemode="tozero")
    st.plotly_chart(style_fig(fig, 420))

    comp = post_hist.assign(gap=(post_hist["log_w1"] - np.log(w1 * 1e6)).abs())
    comp = comp.nsmallest(8, "gap").sort_values("release_date", ascending=False)
    st.markdown("**Films with a similar opening weekend**")
    st.dataframe(pd.DataFrame({
        "Film": comp["title"], "Released": comp["release_date"].dt.date,
        "Opening": comp["w1"].map(money),
        "2nd wknd drop": ((1 - comp["drop2"]) * 100).round(0).astype("Int64").astype(str) + "%",
        "Lifetime": comp["lifetime"].map(money),
        "Multiplier": comp["multiplier"].round(2)}), hide_index=True)

# ================================================================ 4. REPLAY
with tabs[3]:
    st.subheader("Would BoxCast have called it?")
    st.caption("These films were hidden from the model during testing "
               "(released Jul 2022 – Aug 2024). Pick one to see what it would have "
               "predicted after each weekend.")
    k1 = test_post[test_post["k"] == 1].sort_values("lifetime", ascending=False)
    pick = st.selectbox("Film", k1["title"].tolist(), index=0)
    rows = test_post[test_post["title"] == pick].sort_values("k")
    actual = rows["lifetime"].iloc[0]
    fig = go.Figure()
    labels = [f"After weekend {k}" for k in rows["k"]]
    fig.add_trace(go.Bar(x=labels, y=rows["pred_high"] - rows["pred_low"], base=rows["pred_low"],
                         marker_color="rgba(245,184,61,0.35)", name="Predicted range",
                         hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=labels, y=rows["pred_mid"], mode="markers+text",
                             marker=dict(color=GOLD, size=12, symbol="diamond"),
                             text=[money(v) for v in rows["pred_mid"]], textposition="middle right",
                             name="Best estimate"))
    fig.add_hline(y=actual, line=dict(color=TEAL, dash="dash", width=2),
                  annotation_text=f"Actual lifetime {money(actual)}", annotation_position="top left")
    fig.update_yaxes(title="Domestic lifetime gross (USD)", rangemode="tozero")
    st.plotly_chart(style_fig(fig, 420))
    first = rows.iloc[0]
    st.markdown(f"Opening weekend **{money(first['w1'])}**. After it, BoxCast predicted "
                f"**{money(first['pred_mid'])}** (range {money(first['pred_low'])} – "
                f"{money(first['pred_high'])}). Actual: **{money(actual)}**, an error of "
                f"**{abs(first['pred_mid'] - actual) / actual * 100:.0f}%**.")

# ================================================================ 5. ACCURACY
with tabs[4]:
    st.subheader("How accurate is it?")
    st.caption("Every number below comes from films the model never saw while training.")
    pre_m, post_m = M["pre_release"], M["post_release"]
    stages = ["Before release", "After weekend 1", "After weekend 2", "After weekend 3"]
    ours = [pre_m["median_abs_pct_error"]] + [post_m[f"after_weekend_{k}"]["median_abs_pct_error"] for k in (1, 2, 3)]
    base = [pre_m["baseline_budget_only_median_abs_pct_error"]] + \
           [post_m[f"after_weekend_{k}"]["baseline_typical_multiplier_median_abs_pct_error"] for k in (1, 2, 3)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=stages, y=base, name="Simple rule of thumb", marker_color=GREY))
    fig.add_trace(go.Bar(x=stages, y=ours, name="BoxCast", marker_color=GOLD,
                         text=[f"{v}%" for v in ours], textposition="outside"))
    fig.update_layout(barmode="group", title="Typical forecast error (median, lower is better)")
    fig.update_yaxes(title="% error", rangemode="tozero")
    st.plotly_chart(style_fig(fig, 400))
    st.caption("Rule of thumb = budget-only line for pre-release; 'typical multiplier × opening "
               "weekend' after release. Pre-release is worldwide gross, post-release is domestic.")

    c1, c2 = st.columns(2)
    k1t = test_post[test_post["k"] == 1]
    fig = go.Figure(go.Scatter(x=k1t["lifetime"], y=k1t["pred_mid"], mode="markers",
                               marker=dict(color=GOLD, size=7, opacity=0.7), text=k1t["title"],
                               hovertemplate="%{text}<br>actual $%{x:,.0f}<br>predicted $%{y:,.0f}<extra></extra>"))
    lim = [k1t["lifetime"].min() * 0.8, k1t["lifetime"].max() * 1.2]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=GREY, dash="dot"),
                             name="Perfect forecast", hoverinfo="skip"))
    fig.update_xaxes(type="log", title="Actual domestic lifetime")
    fig.update_yaxes(type="log", title="Predicted after weekend 1")
    fig.update_layout(title="Predicted vs actual (2022–24 releases)", showlegend=False)
    c1.plotly_chart(style_fig(fig, 420))

    nice = {"log_budget": "Budget", "star_power": "Lead star's track record",
            "director_power": "Director's track record", "year": "Release year",
            "runtime": "Runtime", "genre": "Genre", "rating": "Age rating",
            "studio": "Studio", "month": "Release month", "is_sequel": "Sequel / franchise",
            "star_films": "Lead star's no. of films", "director_films": "Director's no. of films"}
    imp = pd.Series(M["importance"]["pre_release"]).rename(index=nice).sort_values()
    fig = go.Figure(go.Bar(x=imp.values * 100, y=imp.index, orientation="h", marker_color=TEAL))
    fig.update_layout(title="What drives the pre-release forecast")
    fig.update_xaxes(title="% of model's decisions")
    c2.plotly_chart(style_fig(fig, 420))

    cov = [pre_m["range_coverage"]] + [post_m[f"after_weekend_{k}"]["range_coverage"] for k in (1, 2, 3)]
    st.markdown("**Is the range honest?** The 'likely range' is designed to contain the real "
                "result about 80% of the time. On unseen films it did: "
                + ", ".join(f"{s.lower()} **{c}%**" for s, c in zip(stages, cov)) + ".")

# ================================================================ 6. ABOUT
with tabs[5]:
    st.subheader("How BoxCast works")
    st.markdown(f"""
**Two models, two moments in a film's life**

1. **Pre-release forecast**: learns from {len(pre_hist):,} US releases (1990–2019): budget,
   genre, age rating, runtime, release month, studio, sequel status, and the past box office
   of the lead star and director. Predicts **worldwide** lifetime gross.
2. **Post-release tracker**: learns from {len(post_hist):,} wide releases (2000–2024, daily
   box office). From the opening weekend (and the 2nd/3rd weekend drops) it predicts the
   **multiplier**: how many times its opening a film will earn in total. Predicts
   **domestic** lifetime gross.

**The technique.** Gradient-boosted decision trees (LightGBM, a close cousin of XGBoost) trained
with *quantile loss*, giving a low, likely and high estimate. The ranges are then
**conformally calibrated** on a held-back slice of data so they contain the real outcome
about 80% of the time. The weekly curve uses *analog forecasting*: the typical shape of the
40 past films with the most similar multiplier.

**Always up to date.** Every morning a GitHub Action pulls films now playing and coming soon
from TMDB, forecasts each one and saves the results. No manual updating.

**Data sources:** IMDb movie statistics (via the `movie-stats` dataset on GitHub), daily
Box Office Mojo grosses (via the `boxofficemojo-scraper` project on GitHub), and the TMDB API
for live films.

**Known limitations:** money isn't adjusted for inflation; the pre-release model's data ends in
2019, and post-COVID audiences are more front-loaded; marketing spend, trailer buzz and reviews
aren't included yet (planned next: YouTube trailer views and a review-sentiment model).
This product uses the TMDB API but is not endorsed or certified by TMDB.
""")
