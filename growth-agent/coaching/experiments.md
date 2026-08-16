# Experiments

Every entry: **Hypothesis → What we tried → Result (real numbers) → Keep / Kill / Iterate.** No vague entries — see the root [CLAUDE.md](../CLAUDE.md) self-improving-loop rules. `trend-scout` output lands here as a hypothesis, never as an adopted rule. `daily-coach` closes entries out with real numbers or marks them "no data yet" — never fabricates a result.

---

## Open hypotheses

### H1 — Comment-bait CTA replaces "Follow" asks
**Hypothesis:** Ending captions/videos on a specific comment-bait CTA (per the CTA rule in [voice-guide.md](../brand/voice-guide.md)) will lift comment rate off the current ~0.005% baseline (10 comments / 209K views on the best post).
**Source:** Nick Tarmo's signature move — see [guru-tracker.md](../intel/guru-tracker.md).
**Action plan (2026-08-16 audit):** every exam-stress/internship caption from now on ends with a low-effort, joke-specific prompt — not a generic "comment below." Templates: "comment your worst bed-rotting excuse," "comment the acronym that confused you this week," "comment 'notts' if you're also at a non-target uni." Specificity is the mechanic — a vague CTA gets vague (i.e. no) response.
**What we tried:** Not yet started.
**Result:** No data yet.
**Status:** Open — start with the next exam-stress or internship post.

### H2 — Direct-address share prompts fix the zero-share problem
**Hypothesis:** Adding an explicit "send this to..." line will move share count off zero — every sampled post in the last 90 days had 0 shares, including the 209K-view outlier.
**Source:** Pattern observed directly in the data ([instagram_analysis_varun_dokka.md](../intel/source-research/instagram_analysis_varun_dokka.md)), not a guru tactic.
**Action plan (2026-08-16 audit):** pillar 1 (exam-stress) posts specifically get a direct-address tag line — "send this to the friend who's currently floating into the stars" is the literal template from the account's own top post; reuse that pattern, swapped per video. This is the single highest-leverage untested fix in this file — prioritize it over new pillar work.
**What we tried:** Not yet started.
**Result:** No data yet.
**Status:** Open — highest-leverage single fix available; prioritize.

### H3 — Bio + pinned post rewrite lifts profile-visit-to-follow rate — **#1 PRIORITY, single highest-confidence fix in this file**
**Hypothesis:** A bio that states a clear reason to follow, plus pinning the two proven-pillar posts, will lift the visit-to-follow conversion rate above its current baseline.
**Source:** Recommendation #4 in [instagram_analysis_varun_dokka.md](../intel/source-research/instagram_analysis_varun_dokka.md).

**The specific math:** 6,260 profile visits in the last 90 days. The account's own research file already computed that even a mediocre, unremarkable 2–3% visit-to-follow rate would have produced 300+ new followers from this quarter's traffic alone. Total lifetime followers sit at 170 (as of the 2026-07-08 pull) — i.e. even granting every one of those 170 to this single quarter, the account is converting at roughly a third to a half of what a mediocre account manages, off traffic that's already proven (97.7% non-follower reach, so the algorithm is doing its job upstream of the profile).

**The specific root cause:** the bio is a demographic tag list — "19 | UoN | London — Vegetarian | Fitness | Uni Career growth." It answers *who Varun is* (age, university, city, diet, hobby) and never answers *why a stranger who just watched one video should follow*. Compare this to what's actually converting: the 209K-view top post isn't a demographic fact, it's a hyper-specific feeling ("your next exam is in 12hrs but you're still recovering from the last one"). A visitor who watches that, feels read, clicks through to the profile — and lands on a résumé. That mismatch, not lack of interest, is the most likely single cause of the leak.

**The specific fix — exact copy, iterated live against three working comps Varun supplied (2026-08-16):**

- **Erin:** `daily curiosity maxxing` / `currently building in SF` / `neuroscience | learning | productivity` — never mentions Cambridge despite it being her strongest credential. Lesson: line 1 is a practice/identity statement, not a fact about her.
- **Eileen:** `things i'm learning in life and business` — first-person, ongoing, documents a journey. The promise is "follow along," not "I'll teach you."
- **Nick:** `helping you turn your potential into a reality` — second-person coach-promise. **Rejected as a model** — this is teaching/authority framing, which is exactly the persona `identity.md`'s Aspirational Archetype section rules out (Da Vinci/Aurelius, not Hormozi/guru-coach). Varun's content is a diary, not a course; a "helping you" bio would write a check the content doesn't cash.

Also rejected: "maxxing" as vocabulary — flagged by Varun as wrong for him, and it independently breaks `voice-guide.md`'s own tonality rule (flat, restrained, not chasing a trend word).

`things i'm learning about health, finance & ai` — **killed 2026-08-16.** Varun's own read: "still shit." Correct call — health/finance/ai are generic self-improvement nouns, not grounded in any project file, and not distinctive to Varun specifically.

**Version rebuilt strictly from identity.md + Varun_Content_Plan.md's own already-written language, not invented topic nouns — killed same-day, 2026-08-16, by real-world change (internship ended):**
```
recording the chase for freedom
money · body · mind
pe intern in london, non-target uni
```
- Line 1: the documentary premise straight from the Day-1 script in `Varun_Content_Plan.md` ("putting the whole process on here," "this is entry one") — active, ongoing record of a specific chase, not a static interest list (which is what killed the health/finance/ai draft).
- Line 2: Varun's own three-part breakdown, verbatim, as a compact tag line. Using "money, body, mind" (Content_Plan's Day-1 script + identity.md's own breakdown, both consistent) over identity.md's headline phrase "money, body, and time" — flagging that small inconsistency in identity.md for a future cleanup, not resolving it silently.
- Line 3 (`pe intern in london`) is now a stale fact, not a bio — the internship ended the same day this draft was finalized. Per the project's own "never fabricate a fact" rule, this can't stay live.

**Current final version:**
```
recording the chase for freedom
money · body · mind
uon — non-target uni
```
Lines 1–2 unchanged (still accurate, still the strongest material). Line 3 drops `pe intern in london`; the underdog-uni detail stays since it's still true and still the one proven-to-engage specific (LSE/Cambridge comparison post, 293 accounts-engaged). No replacement status invented for the dropped clause — see H11 and identity.md's 2026-08-16 status note. A bio doesn't need to name a "next chapter" that hasn't happened yet; better to under-claim than to post a fact that's already false.

Pin, in order: (1) the "your next exam is in 12hrs..." post (209,496 views — pillar 1's proof point), (2) the "bed rotting before internship" post (116,899 views — pillar 2's proof point, now a completed-arc pillar — see pillars.md). These stay the right pins regardless of the internship ending; they're historical proof points, not status claims.

*Superseded drafts, kept for the record only:* `...pe intern in london, non-target uni` (killed by internship ending) / `things i'm learning about health, finance & ai` / `documenting the lock in` / `notts → pe intern, london. student & intern life, told straight` / `19, uon, pe intern in london. documenting the whole thing — grind, gym, the wins and the mess.`

**What we tried:** Not yet started.
**Result:** No data yet.
**Status:** Open — do this one first, before anything else in this file. It's a single 2-minute edit (bio text + 2 pins) against the account's single biggest, most confidently-diagnosed leak.

### H4 — Mastery/stoic pillar (Da Vinci/Aurelius layer) earns its own audience
**Hypothesis:** Posting the bymiilan-style and Richie-arc diary entries at 2–3x/week (per [Varun_Content_Plan.md](../intel/source-research/Varun_Content_Plan.md) Part 4) will start showing measurable engagement from a secondary audience segment distinct from the exam-stress/internship crowd, validating pillar 3 in [pillars.md](../content/pillars.md).
**Source:** identity.md's Aspirational Archetype section — this is the identity/movement layer, treated as equal-weight to the proven pillars even pre-data.
**What we tried:** Not yet started.
**Result:** No data yet.
**Status:** Open — `identity-curator` checks in on this at the first monthly review.

### H5 — Outlier-hunting process (Nick Tarmo's own method)
**Hypothesis:** Running Nick Tarmo's own weekly process — find a creator's outlier video (3–5x their normal view count relative to follower count), transcribe the first 3 seconds, log the hook — will build a usable hook swipe file faster than ad hoc research.
**Source:** [guru-tracker.md](../intel/guru-tracker.md), Nick Tarmo transcripts 3, 4, 7, 11, 13.
**What we tried:** Not yet started.
**Result:** No data yet.
**Status:** Open — `trend-scout` to run this as a standing weekly process starting week 2.

### H7 — Batch-scripting a week at once (Justin Welsh's Content Matrix mechanic)
**Hypothesis:** Given the 1–3 videos/day cadence in [goals.md](../brand/goals.md) alongside a PE internship, scripting a full week's worth of `idea-bank.md` entries in one sitting via `script-coach` will hold quality/hook-density steadier than scripting day-by-day under time pressure — measured indirectly via whether daily-log.csv shows a quality dip (views-per-follower, comment rate) on days scripted same-day vs. batch-scripted.
**Source:** Justin Welsh's "Content Matrix" system — see [guru-tracker.md](../intel/guru-tracker.md).
**What we tried:** Not yet started.
**Result:** No data yet.
**Status:** Open — cheap to test, no downside; start week 1.

### H6 — Hormozi "Hook, Retain, Reward" structure, without the visual template
**Hypothesis:** The underlying structure (hard hook → one clear payoff, no rambling → simple CTA) improves retention on internship/exam-stress posts. The Hormozi visual template itself (big-text-block, "woosh" sound) is explicitly rejected — see guru-tracker.md's persona-drift note.
**Source:** [guru-tracker.md](../intel/guru-tracker.md).
**What we tried:** Not yet started.
**Result:** No data yet.
**Status:** Open, low priority — structure overlaps heavily with the existing Triple Hook rule, so this mostly confirms rather than changes current practice.

### H8 — Hook specificity/imagery, not topic category, is the real driver within a proven pillar
**Hypothesis:** Within either proven pillar (exam-stress, internship), a hook built around a specific, vivid, visualizable moment (e.g. "floating into the stars") will outperform a hook that generically names the topic (e.g. "locked in," "brain not braining") by a wide margin — wider than the difference between the two pillars themselves.
**Source:** 2026-08-16 deep audit — see [weekly-summary.md](../analytics/weekly-summary.md) Finding 2. The account's own top 2 posts vs. its other 5 same-pillar posts.
**What we tried:** Not yet started.
**Result:** No data yet.
**Status:** Open — highest-priority scripting rule for `script-coach` going forward. Every idea-bank entry should be checked against this before filming, not just format-matched.

### H9 — Exam-stress pillar has real seasonal dependency
**Hypothesis:** Exam-stress content posted without a live, real exam-period backdrop will underperform the pillar's historical numbers, which were driven by posts dated to 23 May (UK exam season).
**Source:** 2026-08-16 audit — see [weekly-summary.md](../analytics/weekly-summary.md) Finding 3; confirms recommendation #1 in [instagram_analysis_varun_dokka.md](../intel/source-research/instagram_analysis_varun_dokka.md).
**What we tried:** Not yet started.
**Result:** No data yet.
**Status:** Open — directly informs pillar prioritization for August 2026: lean on pillar 2 (internship, currently live) over pillar 1 (exam-stress, currently off-season) until a real exam/deadline/results period returns.

### H10 — Smaller, specifically-personal posts may convert followers at a higher rate than broad viral posts (low confidence, small sample)
**Hypothesis:** Posts built from specific facts about Varun (taste, identity, personal detail) convert profile visitors to followers at a higher rate per view than posts built from a broad, universally-relatable joke — even though the broad joke gets far more raw views.
**Source:** 2026-08-16 audit — see [weekly-summary.md](../analytics/weekly-summary.md) Finding 4. Computed from only 4 posts with follow data (6–13 follows each) — explicitly flagged as low-confidence, not a strategy change on its own.
**What we tried:** Not yet started.
**Result:** No data yet.
**Status:** Open, low confidence — the mastery/identity pillar (H4, pillar 3) is the natural test bed for this once it has its own posts and follow data. Do not act on this alone; needs real daily-log.csv data to move past "interesting anomaly."

### H11 — Teaching/explainer content is an emerging pillar worth tracking (unvalidated, n=2)
**Hypothesis:** Varun's 2 most recent TikToks (posted after the internship ended) shifted into teaching/explainer content and may represent a real new pillar distinct from the exam-stress/internship discovery pillars — possibly overlapping with the mastery/Da Vinci layer (pillar 4) if the teaching content draws on finance/markets expertise, or a distinct education-format pillar in its own right (pillar 3 in `pillars.md`).
**Source:** Varun, 2026-08-16, in conversation.

**What we tried:** Nothing yet on the content side — these 2 videos were already posted before this pillar was identified, not a deliberate test.

**Result (2026-08-16, Varun-reported views; view counts independently unverifiable — see note):** 2,500 views and 700 views. Account has 414 TikTok followers (confirmed directly — see note), so both posts reached well beyond the follower base (6.0x and 1.7x follower count respectively), meaning non-follower discovery is happening on TikTok too, same pattern as the Instagram data. No comment/share/save/new-follow numbers available for either post.

**Verification note:** I checked `tiktok.com/@varun_dokka` directly — profile loads and confirms 414 followers, 61.6K total likes (public, no login needed). The per-video list is blocked by TikTok's bot-detection for an automated/logged-out session (confirmed via a direct API check — empty response body, not a UI glitch). Could not independently verify the specific 2,500/700 figures beyond Varun's own report. Treating them as accurate per the account being his own, but flagging that this project can't self-verify TikTok per-video numbers without the TikTok scraper's authenticated login (`ingestion/scrape_tiktok.py --login`) — that's the actual fix for "why can't you just look," not a limitation specific to these two posts.

**Bonus finding, same check:** the TikTok bio is the identical stale one flagged and already fixed on Instagram — `20 | UoN | London / Vegetarian | Fitness | Uni Career growth`. This project's bio fix (H3) was Instagram-only; TikTok needs the same update, and now also needs the "pe intern" framing removed there too, same as Instagram.

**Status:** Open, low-n. Real validation needs `growth-analyst` tracking a run of posts in this format via actual daily-log.csv data, not 2 self-reported numbers. Two concrete next steps: (1) apply the same bio fix from H3 to TikTok, (2) complete the TikTok scraper login so future teaching-format posts get tracked automatically instead of requiring manual reporting.

---

## Closed hypotheses

*None yet — this account has no daily-log.csv data to close a hypothesis against. See [daily-log.csv](../analytics/daily-log.csv).*
