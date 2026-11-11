"""style.py - the app's look: colours, type and layout (CSS)."""

CSS = """
<style>
@font-face {
  font-family: 'Archivo';
  src: url('./app/static/fonts/archivo.woff2') format('woff2-variations'),
       url('./app/static/fonts/archivo.woff2') format('woff2');
  font-weight: 100 900;
  font-stretch: 62% 125%;
  font-display: swap;
}

:root {
  --bg: #141A26;
  --panel: #1C2433;
  --panel-2: #232D40;
  --line: rgba(238, 241, 246, 0.08);
  --text: #EEF1F6;
  --muted: #8C97AB;
  --beam: #FFC93C;          /* interface accent: marquee yellow */
  --up: #B38A14;            /* chart: positive / BoxCast (validated for dark mode) */
  --down: #4F86E0;          /* chart: negative */
  --good: #5BD08A;
  --warn: #F08C3A;
  --risk: #F0605A;
}

/* ---------- base ---------- */
html, body, .stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button,
.stMarkdown, [data-testid="stExpander"] summary {
  font-family: 'Archivo', system-ui, sans-serif !important;
}
.stApp { background: var(--bg); color: var(--text); }
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, footer, [data-testid="stStatusWidget"] { display: none !important; }
.block-container { max-width: 1180px; padding: 1.25rem 2rem 4rem; }
a { color: var(--beam); }
:focus-visible { outline: 2px solid var(--beam) !important; outline-offset: 2px; }

/* ---------- top bar ---------- */
.bc-top { display: flex; align-items: baseline; justify-content: space-between;
  padding: 0.25rem 0 1.25rem; border-bottom: 1px solid var(--line); margin-bottom: 2.5rem; }
.bc-mark { font-stretch: 70%; font-weight: 800; font-size: 1.6rem; letter-spacing: 0.01em;
  color: var(--text); text-decoration: none; }
.bc-mark span { color: var(--beam); }
.bc-nav a { color: var(--muted); text-decoration: none; margin-left: 1.5rem; font-size: 0.95rem; }
.bc-nav a:hover, .bc-nav a.on { color: var(--text); }

/* ---------- hero (home) ---------- */
.bc-hero h1 { font-stretch: 68%; font-weight: 800; font-size: clamp(2.6rem, 6vw, 4.6rem);
  line-height: 0.98; letter-spacing: -0.01em; margin: 0 0 0.9rem; max-width: 14ch; color: var(--text); }
.bc-hero p { color: var(--muted); font-size: 1.12rem; max-width: 52ch; margin: 0 0 1.6rem; line-height: 1.5; }

/* search box */
[data-testid="stTextInput"] input {
  font-size: 1.25rem !important; padding: 1.05rem 1.2rem !important; height: auto !important;
  background: var(--panel) !important; color: var(--text) !important; border-radius: 14px !important;
}
[data-testid="stTextInput"] > div > div { border-radius: 14px !important; border: 1px solid var(--line) !important;
  background: var(--panel) !important; }
[data-testid="stTextInput"] > div > div:focus-within { border-color: var(--beam) !important;
  box-shadow: 0 0 0 3px rgba(255, 201, 60, 0.22) !important; }
[data-testid="stTextInput"] input::placeholder { color: #6B7690 !important; }
[data-testid="InputInstructions"] { display: none; }

.bc-section { font-stretch: 80%; font-weight: 700; font-size: 1.35rem; margin: 2.6rem 0 1rem; color: var(--text); }
.bc-sub { color: var(--muted); font-size: 0.95rem; margin: -0.6rem 0 1.1rem; }

/* poster cards: the poster itself is the button */
[class*="st-key-pick_"] button {
  width: 100%; aspect-ratio: 2 / 3; height: auto; padding: 0; border-radius: 10px;
  border: 1px solid var(--line); background-color: var(--panel-2);
  background-size: cover; background-position: center; transition: filter .15s ease, border-color .15s ease;
}
[class*="st-key-pick_"] button p { color: transparent; font-size: 0.1px; }
[class*="st-key-pick_"] button:hover { filter: brightness(1.12); border-color: rgba(255,201,60,.7); }
[class*="st-key-pick_"] button.noposter p, [class*="st-key-pick_"].noposter button p { color: var(--muted); font-size: .9rem; }
.bc-card-t { font-weight: 650; font-size: 0.98rem; line-height: 1.25; margin: 0.55rem 0 0.15rem; color: var(--text); }
.bc-card-m { color: var(--muted); font-size: 0.85rem; }
.bc-card-f { color: var(--beam); font-size: 0.85rem; font-weight: 600; font-variant-numeric: tabular-nums; }

/* plain buttons (back, submit) */
.stButton > button, .stFormSubmitButton > button {
  border-radius: 10px; border: 1px solid var(--line); background: var(--panel); color: var(--text);
  font-weight: 600;
}
.stButton > button:hover, .stFormSubmitButton > button:hover { border-color: var(--beam); color: var(--text); }
.st-key-back button { background: transparent; border: none; color: var(--muted); padding-left: 0; }
.st-key-back button:hover { color: var(--text); }

/* ---------- film page ---------- */
.bc-film { position: relative; border-radius: 18px; overflow: hidden; margin: 0.5rem 0 2rem;
  background: var(--panel); background-size: cover; background-position: center top; }
.bc-film::before { content: ""; position: absolute; inset: 0;
  background: linear-gradient(90deg, rgba(20,26,38,.97) 0%, rgba(20,26,38,.88) 45%, rgba(20,26,38,.55) 100%),
              linear-gradient(0deg, rgba(20,26,38,1) 0%, rgba(20,26,38,0) 40%); }
.bc-film-in { position: relative; display: grid; grid-template-columns: 200px 1fr; gap: 2rem; padding: 2.2rem; }
.bc-poster { width: 200px; aspect-ratio: 2/3; border-radius: 12px; background: var(--panel-2) center/cover;
  box-shadow: 0 18px 40px rgba(0,0,0,.45); }
.bc-film h1 { font-stretch: 70%; font-weight: 800; font-size: clamp(2.2rem, 5vw, 3.6rem); line-height: 1;
  margin: 0.2rem 0 0.7rem; color: var(--text); }
.bc-meta { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1rem; }
.bc-chip { border: 1px solid rgba(238,241,246,.18); border-radius: 999px; padding: 0.18rem 0.7rem;
  font-size: 0.85rem; color: var(--text); }
.bc-people { color: var(--muted); font-size: 0.98rem; line-height: 1.6; max-width: 60ch; }
.bc-people b { color: var(--text); font-weight: 600; }
.bc-overview { color: #C4CBD8; font-size: 1rem; line-height: 1.55; max-width: 62ch; margin-top: 0.9rem; }

/* the marquee number */
.bc-fc { padding: 0.5rem 0 0.5rem; }
.bc-fc-k { color: var(--muted); font-size: 1rem; margin-bottom: 0.2rem; }
.bc-fc-n { font-stretch: 62%; font-weight: 800; font-size: clamp(4.5rem, 12vw, 8.5rem); line-height: 0.9;
  letter-spacing: -0.02em; color: var(--text); font-variant-numeric: tabular-nums; }
.bc-fc-r { color: #C4CBD8; font-size: 1.1rem; margin-top: 0.6rem; }
.bc-track { position: relative; height: 46px; margin: 1.4rem 0 0.4rem; }
.bc-track .rail { position: absolute; top: 18px; left: 0; right: 0; height: 6px; border-radius: 3px; background: var(--panel-2); }
.bc-track .band { position: absolute; top: 14px; height: 14px; border-radius: 7px; background: var(--up); opacity: .85; }
.bc-track .mid { position: absolute; top: 8px; width: 4px; height: 26px; border-radius: 2px; background: var(--text);
  transform: translateX(-2px); }
.bc-track .be { position: absolute; top: 4px; height: 34px; border-left: 2px dashed var(--text); }
.bc-track .lab { position: absolute; top: 34px; font-size: 0.78rem; color: var(--muted); white-space: nowrap; transform: translateX(-50%); }
.bc-track .belab { position: absolute; top: -14px; font-size: 0.78rem; color: var(--text); white-space: nowrap; transform: translateX(-50%); }

.bc-figs { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: var(--line);
  border-radius: 14px; overflow: hidden; margin: 2rem 0 0.5rem; }
.bc-fig { background: var(--panel); padding: 1.2rem 1.3rem; }
.bc-fig .k { color: var(--muted); font-size: 0.9rem; }
.bc-fig .v { font-stretch: 75%; font-weight: 750; font-size: 2.1rem; margin: 0.15rem 0; font-variant-numeric: tabular-nums; }
.bc-fig .s { color: var(--muted); font-size: 0.85rem; }
.bc-good { color: var(--good); } .bc-warn { color: var(--warn); } .bc-risk { color: var(--risk); }

.bc-note { background: var(--panel); border-left: 3px solid var(--beam); border-radius: 0 12px 12px 0;
  padding: 0.9rem 1.1rem; color: #C4CBD8; margin: 1.2rem 0; line-height: 1.5; }

/* drivers */
.bc-drv { display: grid; grid-template-columns: minmax(150px, 230px) 1fr 70px; align-items: center;
  gap: 1rem; padding: 0.55rem 0; border-bottom: 1px solid var(--line); }
.bc-drv .n { font-size: 0.98rem; }
.bc-drv .b { position: relative; height: 10px; }
.bc-drv .b::before { content: ""; position: absolute; left: 50%; top: -4px; bottom: -4px; width: 1px; background: rgba(238,241,246,.25); }
.bc-drv .b i { position: absolute; top: 0; height: 10px; border-radius: 4px; }
.bc-drv .v { text-align: right; font-variant-numeric: tabular-nums; font-weight: 650; }

/* comparable films table */
.bc-tbl { width: 100%; border-collapse: collapse; font-size: 0.95rem; }
.bc-tbl th { text-align: left; color: var(--muted); font-weight: 500; padding: 0.5rem 0.6rem; border-bottom: 1px solid var(--line); }
.bc-tbl td { padding: 0.6rem; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }
.bc-tbl td.r, .bc-tbl th.r { text-align: right; }

[data-testid="stSelectbox"] > div > div { background: var(--panel) !important; border-radius: 12px !important;
  border: 1px solid var(--line) !important; }
[data-testid="stExpander"] { border: 1px solid var(--line) !important; border-radius: 14px !important; background: var(--panel); }
[data-testid="stNumberInput"] input, [data-testid="stDateInput"] input { background: var(--panel-2) !important; }

.bc-foot { margin-top: 4rem; padding-top: 1.2rem; border-top: 1px solid var(--line); color: #6B7690; font-size: 0.82rem; line-height: 1.6; }

/* buttons in the accent colour */
.stFormSubmitButton > button { background: var(--beam); color: #141A26; border-color: var(--beam); }
.stFormSubmitButton > button:hover { color: #141A26; filter: brightness(1.08); }

/* how-a-forecast-is-made diagram */
.bc-flow { margin: 1rem 0 0.5rem; }
.bc-cols { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 0.8rem; }
.bc-col { background: var(--panel); border: 1px solid var(--line); border-top: 3px solid var(--beam);
  border-radius: 12px; padding: 0.9rem 1rem; }
.bc-col h4 { margin: 0 0 0.5rem; font-stretch: 80%; font-weight: 700; font-size: 1.05rem; color: var(--text); }
.bc-col ul { margin: 0; padding-left: 1.1rem; color: #C4CBD8; font-size: 0.92rem; line-height: 1.55; }
.bc-arrow { text-align: center; color: var(--beam); font-size: 1.6rem; line-height: 1; margin: 0.55rem 0; }
.bc-stage { background: var(--panel-2); border: 1px solid var(--line); border-radius: 12px; padding: 0.9rem 1.1rem;
  text-align: center; color: var(--text); }
.bc-stage b { color: var(--beam); font-weight: 650; }
.bc-stage small { display: block; color: var(--muted); font-size: 0.86rem; margin-top: 0.25rem; }
.bc-outs { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.6rem; }
.bc-out { border: 1px solid rgba(255,201,60,.45); border-radius: 10px; padding: 0.6rem 0.8rem; text-align: center;
  color: var(--text); font-size: 0.92rem; }
.bc-life { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.8rem; margin-top: 1rem; }
.bc-life div { background: var(--panel); border-radius: 12px; padding: 0.9rem 1rem; border: 1px solid var(--line); }
.bc-life b { display: block; color: var(--text); margin-bottom: 0.25rem; }
.bc-life span { color: #C4CBD8; font-size: 0.92rem; line-height: 1.5; }
.bc-life i { color: var(--beam); font-style: normal; font-weight: 700; margin-right: 0.35rem; }

/* horizontal bar charts on the about page */
.bc-hbar { display: grid; grid-template-columns: minmax(170px, 300px) 1fr 70px; gap: 1rem; align-items: center;
  padding: 0.45rem 0; }
.bc-hbar .t { color: #C4CBD8; font-size: 0.95rem; }
.bc-hbar .bar { height: 22px; border-radius: 0 4px 4px 0; }
.bc-hbar .v { text-align: right; font-weight: 650; font-variant-numeric: tabular-nums; }
.bc-kv { width: 100%; border-collapse: collapse; font-size: 0.94rem; }
.bc-kv td { padding: 0.45rem 0.6rem; border-bottom: 1px solid var(--line); vertical-align: top; }
.bc-kv td:first-child { color: var(--muted); width: 42%; }

/* about page funnel */
.bc-funnel { display: grid; gap: 0.8rem; margin: 1rem 0 0.5rem; }
.bc-funnel .row { display: grid; grid-template-columns: 170px 1fr 64px; gap: 1rem; align-items: center; }
.bc-funnel .bar { height: 26px; border-radius: 0 4px 4px 0; background: var(--up); }
.bc-funnel .base { height: 6px; border-radius: 0 3px 3px 0; background: #4A5568; margin-top: 4px; }
.bc-prose { max-width: 68ch; color: #C4CBD8; line-height: 1.6; font-size: 1.02rem; }
.bc-prose h3 { color: var(--text); font-stretch: 80%; font-weight: 700; margin: 2rem 0 0.5rem; }

@media (max-width: 760px) {
  .block-container { padding: 1rem 1rem 3rem; }
  .bc-film-in { grid-template-columns: 1fr; padding: 1.3rem; }
  .bc-poster { width: 130px; }
  .bc-figs { grid-template-columns: 1fr; }
  .bc-drv { grid-template-columns: 1fr 70px; }
  .bc-drv .b { grid-column: 1 / -1; order: 3; }
  [data-testid="stHorizontalBlock"]:has([class*="st-key-pick_"]) { flex-wrap: wrap !important; gap: 0.8rem !important; }
  [data-testid="stHorizontalBlock"]:has([class*="st-key-pick_"]) > [data-testid="stColumn"] {
    min-width: calc(33.3% - 0.6rem) !important; flex: 1 1 calc(33.3% - 0.6rem) !important; }
  .bc-funnel .row { grid-template-columns: 110px 1fr 54px; }
  .bc-outs { grid-template-columns: 1fr 1fr; }
  .bc-life { grid-template-columns: 1fr; }
  .bc-hbar { grid-template-columns: 1fr 60px; }
  .bc-hbar .bar-wrap { grid-column: 1 / -1; order: 3; }
  .bc-tbl th:nth-child(2), .bc-tbl td:nth-child(2), .bc-tbl th:nth-child(3), .bc-tbl td:nth-child(3) { display: none; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
"""
