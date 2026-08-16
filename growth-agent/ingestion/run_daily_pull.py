"""
Single entry point for the daily ingestion job. Runs both platform scrapers
(each skips itself gracefully if its login hasn't been done yet), then
commits and pushes analytics/daily-log.csv so the cloud daily-coach routine
has fresh data to read.

This is what Task Scheduler calls. It does not touch credentials — see
scrape_instagram.py / scrape_tiktok.py docstrings for the login contract.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
INGESTION_DIR = Path(__file__).parent


def run(cmd, cwd=None):
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def main():
    run([sys.executable, "scrape_instagram.py"], cwd=INGESTION_DIR)
    run([sys.executable, "scrape_tiktok.py"], cwd=INGESTION_DIR)

    # Push only if daily-log.csv actually changed — avoids empty commits
    # on days both scrapers failed to append anything.
    status = subprocess.run(
        ["git", "status", "--porcelain", "growth-agent/analytics/daily-log.csv"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    if not status.stdout.strip():
        print("No new data in daily-log.csv today — nothing to push.")
        return

    run(["git", "add", "growth-agent/analytics/daily-log.csv"], cwd=REPO_ROOT)
    run(["git", "commit", "-m", "Daily analytics pull"], cwd=REPO_ROOT)
    run(["git", "push", "origin", "main"], cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
