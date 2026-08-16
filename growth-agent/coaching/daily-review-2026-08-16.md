# Daily Review — 2026-08-16 (Kickoff Audit)

This is day 0, not a routine daily review — `analytics/daily-log.csv` has no rows yet, so there's no fresh data to react to. This audit runs on the 90-day snapshot pulled 2026-07-08 ([instagram_analysis_varun_dokka.md](../intel/source-research/instagram_analysis_varun_dokka.md)), which is now ~5.5 weeks stale. Treat everything below as the starting brief, not a live reading — see "Data gap" at the end.

## Numbers (stale — last confirmed pull was 2026-07-08)

170 followers · 373,545 accounts reached (97.7% non-followers) · 606,840 views (99.3% Reels) · 6,260 profile visits · 66,822 interactions · 0 shares sampled across every top post · 10 comments on the best-performing (209K-view) post.

## Diagnosis, stated plainly

The account is not a reach problem. 373,545 people saw it and 6,260 clicked through to the profile — that's the algorithm doing its job. The problem is conversion: almost none of those 6,260 visitors became followers, nothing gives a viewer a reason to send the video to a friend, and almost nobody comments. Three specific leaks, same root cause: the account gives people a reason to *watch* but not a reason to *act*.

## Ranked actions for this week, tied to real mechanics

**1. Fix the CTA on every post, starting now (H1 + H2 in experiments.md).** Every caption currently ends on nothing, or on "follow." Nick Tarmo's own signature move — turn the ask into a specific, low-effort comment prompt tied to the joke, not a generic "comment below." Separately, add a direct-address share line to exam-stress posts specifically ("send this to the friend who's currently floating into the stars" — literally the account's own top post's register, just not yet written into the caption). This is the single highest-leverage fix available and it costs nothing to implement — do it before anything else on this list.

**2. Rewrite the bio and pin the strongest post (H3).** Current bio is a stat list ("19 | UoN | London — Vegetarian | Fitness | Uni Career growth") — it says who Varun is, never why a stranger should follow. Pin the 209K-view exam-recovery post as the first grid item; it's the account's clearest, most specific proof of what it actually delivers.

**3. Sharpen specificity, don't broaden it.** Nick Tarmo's "hyper-specific content" mechanic (guru-tracker.md) is already why the top post works — "your next exam is in 12hrs but you're still recovering from the last one" is a hyper-specific moment, not a generic "exams are stressful." The instinct under a 10,000-in-30-days target will be to broaden the topic net to catch more people. Do the opposite — the data says narrower and more specific is what's actually converting.

**4. Turn the internship pillar into a named, repeatable series.** 8+ posts already live at consistent 90–300 engaged with no formal identity. Same caption convention every time ("POV: you're an intern and...") so people start expecting and seeking it out — this is what turns a topic into a series people follow *for*.

**5. Weave in the mastery/stoic pillar starting with the Day 1 mission-statement video.** Cheap to film (talking head, at home, script already written in [Varun_Content_Plan.md](../intel/source-research/Varun_Content_Plan.md)), and it's what makes 100+ future videos read as one story instead of scattered content. This is the one pillar with no CTA at all — the stillness is the point, don't undercut it with a comment-bait ask.

**6. Batch-script a week at once (H7).** Given 1–3 videos/day on top of a PE internship, scripting day-by-day is a real risk to quality under time pressure. Justin Welsh's Content Matrix mechanic — plan a week of scripts in one sitting via `script-coach`, then just film against the plan.

## What not to do

Don't adopt Hormozi's big-text-block visual template or Nick Tarmo's "I don't really care" bravado lines — both are persona, not mechanic, and both actively fight the Da Vinci/Marcus Aurelius identity target in [identity.md](../brand/identity.md). Borrow the CTA structure and hook mechanics; leave the register behind. Don't broaden topics to chase the 30-day number — see point 3.

## Trajectory against the 10,000/30-day goal

Unchanged from the reality check already logged in [goals.md](../brand/goals.md): the math requires roughly a 30–50x jump in profile-visit volume plus a 5–10x conversion-rate improvement, compressed into 30 days. Nothing in this audit changes that math — it's the same six items that would matter regardless of the specific number attached to the goal. Executing all six well is the only real shot at getting close; there's no shortcut being missed.

## Data gap — read before trusting any of the above too far

This entire audit runs on a single 90-day snapshot from 2026-07-08. No `daily-log.csv` data has been ingested since this project was scaffolded today (2026-08-16). Two things should happen before the next review:
1. Run `/log-day` right now for a fresh, dated baseline — see [.claude/commands/log-day.md](../.claude/commands/log-day.md).
2. Either complete the one-time Instagram login for the scraper ([ingestion/README.md](../ingestion/README.md)) or commit to manual `/log-day` entry daily — without one of these, `growth-analyst` has nothing to measure the actions above against.
