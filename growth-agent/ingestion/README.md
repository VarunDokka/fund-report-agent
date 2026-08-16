# Data Ingestion

Honest status as of 2026-08-16: **both scrapers are scaffolded, not verified.** Neither has been run against a live dashboard — that requires Varun's own login (2FA included), which this agent will never do on his behalf. See the hard guardrail in the root [CLAUDE.md](../CLAUDE.md). This agent will never see or type a password for any account — the login step below is the one part of this pipeline that only Varun can do, once per platform.

## Setup

```bash
cd growth-agent/ingestion
pip install -r requirements.txt
playwright install chromium
```

## One-time login, once per platform (Varun does this himself)

```bash
python scrape_instagram.py --login
python scrape_tiktok.py --login
```

Each opens a real, visible Chromium window at that platform's login page. Varun logs in himself, including any 2FA prompt. Once logged in, he presses Enter in the terminal — the script saves the resulting session cookies to `auth_state.json` / `auth_state_tiktok.json` (both git-ignored, never committed, never leave this machine) and closes the browser. No password is ever seen or handled by this agent, and these session files are never pushed to GitHub or read by the cloud daily-coach routine — only the numbers they produce are.

## Daily pull — both platforms + push, one command

```bash
python run_daily_pull.py
```

Runs both scrapers headless, appends rows to `../analytics/daily-log.csv`, then commits and pushes — this is what makes the cloud daily-coach routine's data actually current. If a scraper's login hasn't been done yet, it prints a message and skips itself rather than failing the whole run. Fields that can't be read confidently are left blank rather than guessed — see each script's `safe_extract` function and the "never fabricate a number" rule in the root CLAUDE.md.

Wired to run automatically once a day via a local Task Scheduler job — see "Automation" below. This has to run locally, not in the cloud routine: the cloud daily-coach session has no access to a browser session on this machine, and Varun's login cookies should never sit in a cloud environment or a git repo.

**Expect to need to fix selectors on first real run.** Neither Instagram's Insights page nor TikTok Studio has a stable public DOM contract, so the placeholder selectors in both scripts will likely need hand-adjustment the first time they run against the live page. Run once, see what's captured vs. missed, fix together.

**Per-post metrics (comments, shares, saves, new follows) aren't on either platform's summary dashboard** — both scripts leave them blank. Filling those in currently requires either extending the scraper to visit each post's individual insights panel, or capturing them via `/log-day`.

## Automation — local Task Scheduler

A Windows Task Scheduler job runs `python run_daily_pull.py` once a day so this doesn't require Varun to remember to trigger it. Setup is one `schtasks` command — see the root [CLAUDE.md](../CLAUDE.md) for the exact command used and how to check/change it. If the machine is off or asleep at the scheduled time, that day is silently skipped — `daily-coach` will say "no data ingested today" rather than guess, per the no-fabrication rule. Use `/log-day` to backfill a missed day manually.

## Manual fallback — /log-day

If the scraper fails, hasn't been run, or a login/2FA blocks it, use the `/log-day` command (`../.claude/commands/log-day.md`) to log the day's key numbers in under 30 seconds. **The pipeline should never go silent just because scraping failed one day** — always fall back to this rather than skip logging entirely.

## What this reads

Varun's own account analytics only — his own Professional Dashboard / Insights. Not a scrape of other people's pages, and not the same thing as `ig-transcript-tool/` elsewhere in this repo (which pulls public reel transcripts via Apify, unrelated to this).
