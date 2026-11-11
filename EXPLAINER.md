# BoxCast explained: for your report and viva

## 1. The problem
Studios, distributors and cinema chains commit huge sums (marketing, screens, release dates)
before they know whether a film will work. BoxCast answers two questions:
1. **Before release:** "What is this film likely to earn worldwide?"
2. **After release:** "Given the opening weekend, where will it finish?" The answer gets
   sharper every weekend.

## 2. The data
| Dataset | What it has | Used for |
|---|---|---|
| IMDb movie stats (7,668 films, 1980–2020) | budget, worldwide gross, genre, rating, runtime, star, director, studio | Pre-release model (5,097 US releases from 1990–2019 after cleaning) |
| Daily Box Office Mojo grosses (2000–Jan 2025) | gross and theater count for every film, every day | Post-release model (2,810 wide releases) |
| TMDB API (live, daily) | films now playing and coming soon, with cast, budget, genre... | The daily forecasts |

**Cleaning decisions worth mentioning:**
- Only US releases with a known gross and runtime. A missing budget is allowed, and the model learns to cope.
- Films hit by COVID cinema closures (Jan 2020 – Jun 2021) are removed, because their runs were cut short artificially.
- Films with the same title (e.g. *Halloween* 2007 vs 2018) are separated using a 60-day gap rule.
- "Wide releases" means 1,000+ theaters on opening weekend.

## 3. Feature engineering (turning information into numbers)
**Pre-release:** log of budget, runtime, release month, year, genre, age rating, studio group,
sequel/franchise flag, and **star power / director power**: the average past gross of the
lead star's and director's *earlier* films only. Using only earlier films prevents
**data leakage** (the model can't peek into the future).

**Post-release:** opening weekend (log), number of theaters, release month and year, holiday
or summer flags, preview days (Wednesday/Thursday openings), major studio flag, and the
**weekend-to-weekend drops** (2nd weekend ÷ 1st, 3rd ÷ 2nd), the strongest word-of-mouth signal.

Why logs? Box office is extremely skewed ($1M films next to $2B films). Logs make the patterns
proportional ("a 10% bigger budget → x% more gross"), which models handle far better.

**The target for post-release is the *multiplier*** (lifetime ÷ opening weekend), not the
raw gross. A typical Hollywood film earns about 2.8× its opening weekend. Front-loaded
franchise films earn closer to 2.2×, and word-of-mouth hits earn 4× or more.

## 4. The models
**Gradient-boosted decision trees (LightGBM).** It builds hundreds of small decision trees,
each one correcting the mistakes of the ones before it. It's the same family as XGBoost
(LightGBM is a faster implementation of the same idea), handles missing values natively, and
captures non-linear effects, such as budget mattering more for action than for drama.

**Quantile regression → ranges, not just one number.** Instead of one model, we train three
per question, each with a different *quantile loss*:
- 10th percentile → the **low case**
- 50th percentile (median) → the **best estimate**
- 90th percentile → the **high case**

**Conformal calibration → honest ranges.** Raw quantile models tend to be over-confident (our
range first caught the real result only ~68% of the time instead of 80%). So we hold back the
most recent slice of the training data, measure how far real results fell outside the range,
and widen the range by exactly that amount. On unseen films the range then contained the
actual result about 80% of the time, as designed.

**Analog forecasting for the weekly curve.** The trajectory chart borrows the typical shape
of the 40 past films with the most similar multiplier. **Link to the Bass diffusion model:**
Bass splits adoption into "innovators" (people who come because of marketing, *p*) and
"imitators" (word of mouth, *q*). Most Hollywood films behave like a Bass curve with *p* ≫ *q*:
a big opening, then roughly exponential decay. Word-of-mouth hits have a larger *q* and decay
slowly. Our multiplier and the weekend drops measure exactly this: a low drop means high *q*.
*(Fitting a full Bass model per film is a good extension to mention.)*

**Extra pre-release forecasts.** By matching films across both datasets (same title and
year, 2,169 films), the same pre-release inputs also forecast the **US opening weekend**
(typical error 45%) and the **US total** (53%).

**"Why this number".** For each film the app splits the forecast into the contribution of
each factor (budget, star, director, timing...) using the model's built-in feature
contributions (the same idea as SHAP values). It starts from the typical film and shows
how much each factor moves the forecast up or down.

**Chance to break even.** A rule of thumb says a film needs about 2.5× its budget worldwide
to break even (cinemas keep roughly half the ticket money, plus marketing costs). Treating
the calibrated forecast range as a log-normal distribution gives the probability of
clearing that bar.

**Franchise history and recent films (after the Retrain workflow).** The retrain looks up
every film on TMDB, finds the previous film in its series and records what it earned. That
becomes a model input ("Previous film in the series"), and it also adds 2020–2025
releases so the model learns today's post-COVID market and current stars. Testing then
switches to films released from 2022 onwards. A safety check only replaces the existing
model if the new one is more accurate on those unseen films. (Before retraining, the app
applies a simple fixed rule instead: sequels typically earn ~0.86× their predecessor.)

**Market signals (added to the Retrain workflow).**
- *Online buzz:* daily views of the film's English Wikipedia page, summed over the 4 weeks
  ending 7, 30, 60 and 90 days before release (Wikimedia's free pageviews API; data from
  mid-2015). Each film with buzz data contributes one training row per distance from release,
  weighted ¼, so the model learns how buzz builds and can compare an upcoming film with past
  films at the same distance. Daily data is timestamped, so only pre-release views are used.
  Based on Mestyán, Yasseri & Kertész (PLOS ONE, 2013).
- *China release:* whether TMDB lists a cinema release in China. A China release can multiply a
  film's worldwide total.
- *Competition:* how many other $50M+ films open within a week either side.
- *Fair test:* the retrain trains the model with and without these signals on the same unseen
  films and keeps the more accurate version. Both numbers are saved in `models/metrics.json`
  (`feature_comparison`) and shown on the app's "How it works" page.

**Why not trailer views?** Today's view count for an old trailer mostly reflects whether the
film became a hit, which happened *after* release. Training on it would leak the answer:
great test scores, useless real forecasts. A clean buzz signal needs views recorded before
release, which the app's daily history file will collect over time.

## 5. How we tested it (very important in a viva)
**Time-based split:** train on older films, test on newer films the model has never seen,
exactly like forecasting a real future release. Random splits would be cheating, because the
model would learn from films released after the ones it is "predicting".
- Pre-release: trained on 1990–2015, tested on **701 films from 2016–2019**
- Post-release: trained on 2000–Jun 2022, tested on **~250 films from Jul 2022–Aug 2024**

## 6. Results (median error on unseen films)
| Stage | BoxCast | Simple rule of thumb | Range contained the actual |
|---|---|---|---|
| Before release (worldwide) | **60.5%** | 64.4% (budget-only line) | 79.7% |
| After weekend 1 (domestic) | **16.3%** | 20.4% (typical multiplier) | 73.1% |
| After weekend 2 | **8.5%** | 19.5% | 84.6% |
| After weekend 3 | **5.7%** | 18.0% | 84.7% |

After weekend 1, **67% of forecasts were within ±25%** of the final number. After weekend 2, **93%** were.

**How to talk about the 60% pre-release error:** it's honest, not a failure. Before release
the model can't see the most important factor, which is whether audiences will like the film.
That's exactly why the product (a) always shows a range and (b) re-forecasts as soon as
weekend data arrives. The big story is the **"forecast funnel": 60% → 16% → 8.5% → 5.7%.**

**What drives the forecasts:**
- Pre-release: budget (~41%), then director's and star's track record, release year, runtime, studio.
- Post-release: the 2nd-weekend drop dominates once known (~46%), followed by opening size and theater count.

## 7. Limitations (say these before they're asked)
- Money isn't adjusted for inflation. The year feature only partly captures this.
- The pre-release data ends in 2019, and post-COVID audiences are more front-loaded.
- The sequel flag is a rough title rule (numbers, "Part", a colon).
- Missing so far: marketing spend, trailer views, social buzz, critic and audience reviews.
- The historical data is scraped from public sources. A commercial version must license data.

## 8. What makes it sellable, and the patent angle
- **Customers:** distributors (release dates, screen deals), exhibitors (how many screens in week 2),
  producers and financiers (greenlight and pre-sales), trade press and analysts.
- **Novel technical system (the patent framing):** an automated pipeline that (1) ingests live
  release data daily, (2) produces calibrated quantile forecasts, (3) *re-forecasts* using
  sequential weekend-decay signals, and (4) generates trajectories by analog matching. Under
  India's Section 3(k), the claim should centre on this technical system and its
  data-processing steps, not on "predicting revenue" (which is a business method).
- **Moat:** a public, timestamped track record (`data/forecast_history.csv` grows every day)
  plus licensed data.

## 9. Roadmap
1. Indian version (Bollywood + South) with licensed daily data
2. YouTube trailer views and Google Trends as pre-release buzz signals
3. Review-sentiment model (MuRIL / IndicBERT for Indian languages)
4. "Ask the Analyst" AI chat that calls the forecasting model as a tool

## 10. Likely viva questions
- **Why not linear regression?** We tested a budget-only line as the baseline, and BoxCast beats
  it at every stage. Trees also capture interactions (e.g. genre × budget) automatically.
- **Why predict the log / the multiplier?** Box office is skewed, logs stabilise it. The
  multiplier separates "how big it opened" from "how well it holds".
- **How do you avoid overfitting?** Time-based testing on unseen years, shallow trees
  (15 leaves), row and column subsampling, and comparison against a baseline.
- **How do you know the range is trustworthy?** Conformal calibration, with coverage
  measured on unseen films (~80%).
- **Where does live data come from, and who updates it?** The TMDB API, fetched every morning
  by a GitHub Actions workflow. No manual work.
