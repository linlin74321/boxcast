# 🎬 BoxCast: Hollywood Box Office Forecaster

Forecasts a film's lifetime box office **before release** (worldwide), then
sharpens the forecast **after each weekend** (domestic US + Canada).
Updates itself every morning with films from TMDB.

---

## Setup guide (no coding needed, about 30 minutes)

You already have: a GitHub account, a Streamlit Community Cloud account and a TMDB account.

### Step 1: Unzip the project
Unzip `boxcast.zip` on your computer. You'll get a folder called `movie-forecast`.

### Step 2: Create a GitHub repository
1. Go to github.com and click the **+** (top right), then **New repository**.
2. Name it `boxcast`. Choose **Private** (keeps your idea safe) or Public.
3. Leave everything else as it is and click **Create repository**.

### Step 3: Upload the files
1. On the new repository page, click the link **"uploading an existing file"**.
2. Open the `movie-forecast` folder, select **everything inside it**
   (`app.py`, `requirements.txt`, `README.md`, `EXPLAINER.md`, and the folders `src`, `data`, `models`),
   and drag it all into the browser window.
3. Wait until every file has uploaded, then click **Commit changes**.

**Two hidden folders need adding by hand.** Mac and Windows hide folders whose names start with
a dot, so they usually don't get dragged in. Add them like this:

**(a) The daily-update robot (required)**
1. In your repository click **Add file**, then **Create new file**.
2. In the name box type exactly: `.github/workflows/daily_update.yml`
   (typing the `/` creates the folders automatically).
3. Open the file `movie-forecast/.github/workflows/daily_update.yml` on your computer with
   Notepad or TextEdit, copy everything, and paste it into the big box.
   *(Can't see the file? On Mac press `Cmd + Shift + .` in Finder. On Windows, File Explorer → View → Show → Hidden items.)*
4. Click **Commit changes**.

**(b) The dark cinema theme (optional)**
Same steps, with file name `.streamlit/config.toml`, pasting the contents of
`movie-forecast/.streamlit/config.toml`.

### Step 4: Put the app online with Streamlit
1. Go to share.streamlit.io and click **Create app**, then **Deploy a public app from GitHub**.
2. Repository: `your-username/boxcast` · Branch: `main` · Main file path: `app.py`
3. Optional: choose a nice URL like `boxcast-yourname`.
4. Click **Deploy**. The first start takes 2–4 minutes. You now have a live link!

Everything works at this point except the **Upcoming releases** tab, which needs Step 5.

### Step 5: Turn on the daily automatic update
**5a. Get your TMDB key.** On themoviedb.org: click your profile picture, then
**Settings → API**. Copy the **API Key** (the short one). The long "API Read Access Token" also works.

**5b. Give the key to GitHub (safely).** In your GitHub repository:
**Settings → Secrets and variables → Actions → New repository secret**
- Name: `TMDB_API_KEY`
- Secret: paste your key, then click **Add secret**

**5c. Let the robot save files.** **Settings → Actions → General**, scroll to
**Workflow permissions**, pick **Read and write permissions** and click **Save**.

**5d. Run it once now.** Click the **Actions** tab (click the green button if GitHub asks you to
enable workflows). Pick **Daily update** on the left, then **Run workflow → Run workflow**.
After about 2 minutes you'll see a green tick ✅.

Open your Streamlit app: the **Upcoming releases** tab is now full of real films.
From now on it updates **every morning at about 7:47 AM India time**, automatically.

---

## If something goes wrong
| Problem | Fix |
|---|---|
| Streamlit shows "ModuleNotFoundError" | Check `requirements.txt` was uploaded to the main folder (not inside another folder). |
| Streamlit says it can't find `app.py` | Your files went inside a subfolder. Main file path should then be e.g. `movie-forecast/app.py`. |
| Daily update has a red ❌ | Click it to read the error. "401" means the TMDB key is wrong, so redo step 5b. "403 / permission denied" means redo step 5c. |
| Upcoming tab still empty after a green tick | In Streamlit, click **⋮ → Reboot app**. |
| Action stopped running after a few weeks | GitHub pauses schedules in quiet repositories. Go to Actions → Daily update → **Enable workflow**. |

---

## What's inside
```
app.py                     the Streamlit app (all 6 tabs)
requirements.txt           Python libraries the app needs
src/features.py            turns movie info into model inputs (shared rules)
src/build_datasets.py      downloads historical data and builds training tables
src/train_models.py        trains the models and measures accuracy
src/predict.py             loads the models and makes forecasts
src/fetch_tmdb.py          the daily update (TMDB → forecasts)
data/                      training tables, test results, live forecasts
models/                    the trained models + metrics.json (accuracy numbers)
.github/workflows/         the daily-update robot
EXPLAINER.md               plain-English guide to the method, for your report and viva
```

## Rebuilding the models (optional)
Not needed: trained models are already included. If you ever want to retrain
(for example in Google Colab), run:
```
pip install -r requirements.txt
python src/build_datasets.py
python src/train_models.py
```

*This product uses the TMDB API but is not endorsed or certified by TMDB.
Historical data: IMDb movie stats (danielgrijalva/movie-stats) and daily Box Office Mojo
grosses (tjwaterman99/boxofficemojo-scraper). Fine for a student project. A commercial
version should license its data.*
