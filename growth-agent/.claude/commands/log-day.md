---
description: Manual fallback for logging today's Instagram/TikTok numbers in under 30 seconds when the scraper hasn't run or failed.
---

Ask Varun for today's numbers, one line each, accepting "skip" or "don't know" for any field rather than pressing for a guess:

- Platform (instagram / tiktok)
- Views today
- Reach today
- Current follower count
- Profile visits today
- Comments today
- Shares today
- Saves today
- New follows today

Then:

1. Append one row to `analytics/daily-log.csv` with today's date. Leave any skipped field blank — never fill it with a guess, an estimate, or yesterday's number. A blank cell means "no data," which is different from "zero."
2. If this is the only data source for today (the scraper didn't run), say so plainly in the day's context — `daily-coach` needs to know this row came from manual entry, not the automated pull, in case the manual number is less precise (e.g. "around 200" rounded by Varun from memory vs. an exact dashboard read).
3. Do not attempt to also run the scraper in the same session unless Varun asks — this command exists specifically for when that path isn't available.
