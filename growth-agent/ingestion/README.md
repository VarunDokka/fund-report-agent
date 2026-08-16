# Data Ingestion

Honest status as of 2026-08-16: **the Instagram scraper is scaffolded, not verified.** It has not been run against the live dashboard — that requires Varun's own login (2FA included), which this agent will never do on his behalf. See the hard guardrail in the root [CLAUDE.md](../CLAUDE.md). The manual fallback below works today and should be the primary path until the scraper is tested and its selectors confirmed against the real DOM.

## Setup

```bash
cd growth-agent/ingestion
pip install -r requirements.txt
playwright install chromium
```

## One-time login (Varun does this himself)

```bash
python scrape_instagram.py --login
```

This opens a real, visible Chromium window at the Instagram login page. Varun logs in himself, including any 2FA prompt. Once logged in and looking at his feed, he presses Enter in the terminal — the script saves the resulting session cookies to `auth_state.json` (git-ignored, never committed) and closes the browser. No password is ever seen or handled by this agent.

## Daily pull

```bash
python scrape_instagram.py
```

Runs headless, reads the Professional Dashboard / Insights overview, and appends a row to `../analytics/daily-log.csv`. Fields it can't read confidently are left blank rather than guessed — see the `safe_extract` function and the "never fabricate a number" rule in the root CLAUDE.md.

**Expect to need to fix selectors on first real run.** Instagram's Insights page has no stable public DOM contract, so the placeholder selectors in `scrape_instagram.py` will likely need hand-adjustment the first time this runs against the live page. Run it, see what it captures vs. misses, and update the selectors together.

**Per-post metrics (comments, shares, saves, new follows) aren't on the dashboard summary view** — the script leaves them blank. Filling those in currently requires either extending the scraper to visit each post's individual insights panel, or capturing them via `/log-day`.

## TikTok

Not yet built. The brief calls for the same approach against TikTok's Creator/Analytics dashboard — this is future work, not represented in `daily-log.csv` yet. Until it exists, TikTok numbers go through `/log-day` only.

## Manual fallback — /log-day

If the scraper fails, hasn't been run, or a login/2FA blocks it, use the `/log-day` command (`../.claude/commands/log-day.md`) to log the day's key numbers in under 30 seconds. **The pipeline should never go silent just because scraping failed one day** — always fall back to this rather than skip logging entirely.

## What this reads

Varun's own account analytics only — his own Professional Dashboard / Insights. Not a scrape of other people's pages, and not the same thing as `ig-transcript-tool/` elsewhere in this repo (which pulls public reel transcripts via Apify, unrelated to this).
