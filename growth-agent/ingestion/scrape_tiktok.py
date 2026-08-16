"""
Daily TikTok Creator/Analytics pull for @varun_dokka's own account.

Same contract as scrape_instagram.py: this reads Varun's own account
analytics, never his password. --login opens a real, visible browser
window and Varun logs in himself (including any 2FA). The script only
saves the resulting session cookies to auth_state_tiktok.json (git-
ignored) so future runs don't need a fresh login.

Usage:
    python scrape_tiktok.py --login     # one-time, interactive, run this first
    python scrape_tiktok.py             # daily pull, headless, appends to
                                         # ../analytics/daily-log.csv

TikTok's Creator Center DOM has no stable public selectors either, so
this is best-effort in the same way scrape_instagram.py is — see that
file's docstring for the full reasoning. If a field can't be read with
confidence, it's left blank rather than guessed.
"""

import argparse
import csv
import re
from datetime import date
from pathlib import Path

from playwright.sync_api import sync_playwright

AUTH_STATE_PATH = Path(__file__).parent / "auth_state_tiktok.json"
DAILY_LOG_PATH = Path(__file__).parent.parent / "analytics" / "daily-log.csv"
CREATOR_CENTER_URL = "https://www.tiktok.com/tiktokstudio/analytics/overview"

CSV_FIELDS = [
    "date", "platform", "views", "reach", "followers",
    "profile_visits", "comments", "shares", "saves", "new_follows",
]


def run_login():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.tiktok.com/login")
        print("Log in to TikTok in the opened window (including any 2FA).")
        print("Once you're on your profile/feed, come back here and press Enter.")
        input()
        context.storage_state(path=str(AUTH_STATE_PATH))
        browser.close()
    print(f"Session saved to {AUTH_STATE_PATH}.")


def safe_extract(page, label: str, selector: str) -> str | None:
    try:
        el = page.locator(selector).first
        if el.count() == 0:
            return None
        text = el.inner_text(timeout=3000).strip()
        digits = re.sub(r"[^\d]", "", text)
        return digits or None
    except Exception as e:
        print(f"  [warn] couldn't read '{label}' ({selector}): {e}")
        return None


def run_daily_pull():
    if not AUTH_STATE_PATH.exists():
        print("No saved session found. Run `python scrape_tiktok.py --login` first.")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state=str(AUTH_STATE_PATH))
        page = context.new_page()
        page.goto(CREATOR_CENTER_URL, wait_until="networkidle")

        # NOTE: placeholder selectors, same caveat as scrape_instagram.py —
        # TikTok Studio's DOM will need hand-verification against the live
        # page on first real run.
        row = {
            "date": date.today().isoformat(),
            "platform": "tiktok",
            "views": safe_extract(page, "views", "text=/Video views/i >> xpath=.."),
            "reach": None,  # TikTok Studio doesn't expose a direct "reach" metric the way IG does
            "followers": safe_extract(page, "followers", "text=/Followers/i >> xpath=.."),
            "profile_visits": safe_extract(page, "profile_visits", "text=/Profile views/i >> xpath=.."),
            "comments": None,
            "shares": None,
            "saves": None,
            "new_follows": None,
        }
        browser.close()

    missing = [k for k, v in row.items() if v is None and k not in ("date", "platform")]
    if missing:
        print(f"[warn] could not confidently read: {', '.join(missing)}. "
              f"Leaving those cells blank — fill via /log-day if you have the real numbers.")

    write_row(row)


def write_row(row: dict):
    is_new = not DAILY_LOG_PATH.exists()
    with open(DAILY_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended {row['date']} ({row['platform']}) to {DAILY_LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--login", action="store_true", help="one-time interactive login")
    args = parser.parse_args()

    if args.login:
        run_login()
    else:
        run_daily_pull()
