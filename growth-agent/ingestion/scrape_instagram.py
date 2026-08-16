"""
Daily Instagram Insights pull for @varun_dokka's own Professional Dashboard.

This reads Varun's own account analytics — it does not scrape other accounts.
It never handles Varun's password: the --login step opens a real, visible
browser window and Varun logs in himself (including any 2FA prompt). The
script only saves the resulting session cookies to auth_state.json (git-
ignored) so future runs don't need a fresh login.

Usage:
    python scrape_instagram.py --login     # one-time, interactive, run this first
    python scrape_instagram.py             # daily pull, headless, appends to
                                            # ../analytics/daily-log.csv

Instagram's Insights DOM changes without notice and has no stable public
selectors, so the scrape step below is best-effort. If it can't find a
number with confidence, it skips that field rather than guessing — see
`safe_extract`. Treat a script failure as "no data today," not a crash to
paper over: fall back to /log-day (../.claude/commands/log-day.md) and log
it as a fabrication risk (CLAUDE.md's "never fabricate a number" rule)
rather than inventing a plausible-looking row.
"""

import argparse
import csv
import re
from datetime import date
from pathlib import Path

from playwright.sync_api import sync_playwright

AUTH_STATE_PATH = Path(__file__).parent / "auth_state.json"
DAILY_LOG_PATH = Path(__file__).parent.parent / "analytics" / "daily-log.csv"
DASHBOARD_URL = "https://www.instagram.com/accounts/professional_dashboard/"
INSIGHTS_URL = "https://www.instagram.com/accounts/insights/"

CSV_FIELDS = [
    "date", "platform", "views", "reach", "followers",
    "profile_visits", "comments", "shares", "saves", "new_follows",
]


def run_login():
    """Interactive, one-time: Varun logs in himself, we save the session."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto("https://www.instagram.com/accounts/login/")
        print("Log in to Instagram in the opened window (including any 2FA).")
        print("Once you're on your feed/profile, come back here and press Enter.")
        input()
        context.storage_state(path=str(AUTH_STATE_PATH))
        browser.close()
    print(f"Session saved to {AUTH_STATE_PATH}. Future runs won't need this step "
          f"until Instagram invalidates the session.")


def safe_extract(page, label: str, selector: str) -> str | None:
    """Best-effort text extraction. Returns None (never a guess) on failure."""
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
        print("No saved session found. Run `python scrape_instagram.py --login` first.")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(storage_state=str(AUTH_STATE_PATH))
        page = context.new_page()
        page.goto(INSIGHTS_URL, wait_until="networkidle")

        # NOTE: these selectors are best-effort placeholders. Instagram's
        # Insights page is a dynamic React app with no stable class/test-id
        # scheme, so this WILL likely need hand-adjustment against the live
        # DOM the first time it's run — inspect the page and update the
        # selectors below rather than trusting these blindly.
        row = {
            "date": date.today().isoformat(),
            "platform": "instagram",
            "views": safe_extract(page, "views", "text=/Views/i >> xpath=.."),
            "reach": safe_extract(page, "reach", "text=/Accounts reached/i >> xpath=.."),
            "followers": safe_extract(page, "followers", "text=/Followers/i >> xpath=.."),
            "profile_visits": safe_extract(page, "profile_visits", "text=/Profile visits/i >> xpath=.."),
            "comments": None,   # per-post metric — not on the dashboard summary
            "shares": None,     # per-post metric — not on the dashboard summary
            "saves": None,      # per-post metric — not on the dashboard summary
            "new_follows": None,
        }
        browser.close()

    missing = [k for k, v in row.items() if v is None and k not in ("date", "platform")]
    if missing:
        print(f"[warn] could not confidently read: {', '.join(missing)}. "
              f"Leaving those cells blank rather than guessing — fill them via "
              f"/log-day if you have the real numbers.")

    write_row(row)


def write_row(row: dict):
    is_new = not DAILY_LOG_PATH.exists()
    with open(DAILY_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(f"Appended {row['date']} to {DAILY_LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--login", action="store_true", help="one-time interactive login")
    args = parser.parse_args()

    if args.login:
        run_login()
    else:
        run_daily_pull()
