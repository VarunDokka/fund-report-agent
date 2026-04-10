#!/usr/bin/env python3
"""
run_agent.py
============
CLI for the fund-report extraction agent.

Commands
--------
  # Process a directory of PDFs
  python run_agent.py --input reports/

  # Process a single PDF
  python run_agent.py --input reports/fund.pdf

  # Print a formatted table of all flagged items
  python run_agent.py --review

  # Interactive Y/N approval of every unreviewed row in the CSV
  python run_agent.py --approve review/review_queue.csv

Optional flags (work alongside --input)
----------------------------------------
  --output-dir DIR    Destination for output JSON files  (default: output/)
  --review-dir DIR    Destination for review_queue.csv   (default: review/)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── project root on sys.path so `src` resolves when run directly ──────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import ExtractionAgent

# ── logging (concise for the CLI layer; the agent adds its own timestamped lines)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── review CSV schema ──────────────────────────────────────────────────────────
# Original columns written by agent.py
_AGENT_COLS = [
    "filename",
    "field_name",
    "extracted_value",
    "confidence",
    "reason_for_flag",
]
# Extra columns added by --approve
_APPROVE_COLS = ["human_reviewed", "review_timestamp", "reviewer_notes"]
_ALL_COLS = _AGENT_COLS + _APPROVE_COLS

# ── table rendering ────────────────────────────────────────────────────────────

_BOX = {
    "tl": "┌", "tr": "┐", "bl": "└", "br": "┘",
    "h": "─", "v": "│",
    "lt": "├", "rt": "┤", "tt": "┬", "bt": "┴", "x": "┼",
}


def _truncate(text: str, width: int) -> str:
    """Truncate *text* to *width* characters, appending '…' if cut."""
    s = str(text) if text else ""
    return s if len(s) <= width else s[: width - 1] + "…"


def _render_table(
    headers: List[str],
    rows: List[List[str]],
    max_widths: Optional[Dict[str, int]] = None,
) -> str:
    """
    Render a Unicode box-drawing table.

    Parameters
    ----------
    headers:    Column header labels.
    rows:       Data rows (list of lists, same length as headers).
    max_widths: Optional per-column max width caps keyed by header name.
    """
    max_widths = max_widths or {}
    col_widths: List[int] = []
    for i, h in enumerate(headers):
        cap = max_widths.get(h, 999)
        data_max = max((len(_truncate(r[i], cap)) for r in rows), default=0)
        col_widths.append(min(max(len(h), data_max), cap))

    def _row_line(cells: List[str]) -> str:
        parts = [
            f" {_truncate(c, col_widths[i]):<{col_widths[i]}} "
            for i, c in enumerate(cells)
        ]
        return _BOX["v"] + _BOX["v"].join(parts) + _BOX["v"]

    def _sep(left: str, mid: str, right: str) -> str:
        segs = [_BOX["h"] * (w + 2) for w in col_widths]
        return left + mid.join(segs) + right

    lines = [
        _sep(_BOX["tl"], _BOX["tt"], _BOX["tr"]),
        _row_line(headers),
        _sep(_BOX["lt"], _BOX["x"], _BOX["rt"]),
        *[_row_line(r) for r in rows],
        _sep(_BOX["bl"], _BOX["bt"], _BOX["br"]),
    ]
    return "\n".join(lines)


# ── summary banner ─────────────────────────────────────────────────────────────

def _print_run_summary(
    summary: Dict[str, Any],
    output_dir: str,
    processed_stems: List[str],
) -> None:
    """
    Print the end-of-run summary:
      X fields extracted, Y flagged for review, Z passed validation.
    """
    # Count accepted fields by reading the output JSONs just written
    accepted_fields = 0
    for stem in processed_stems:
        json_path = Path(output_dir) / f"{stem}.json"
        try:
            with open(json_path, encoding="utf-8") as fh:
                data = json.load(fh)
            accepted_fields += len(data.get("fields", {}))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    flagged = summary.get("flagged_fields", 0)
    total = accepted_fields + flagged
    errors = summary.get("errors", 0)
    pct_pass = f"{accepted_fields / total * 100:.0f}%" if total else "—"
    pct_flag = f"{flagged / total * 100:.0f}%" if total else "—"

    width = 54
    bar = "═" * width
    print(f"\n╔{bar}╗")
    print(f"║{'  RUN SUMMARY':^{width}}║")
    print(f"╠{bar}╣")
    print(f"║  {'Documents processed':<30} {summary.get('processed', 0):>6}          ║")
    print(f"║  {'Fields extracted (total)':<30} {total:>6}          ║")
    print(f"║  {'Passed validation':<30} {accepted_fields:>6}  ({pct_pass})   ║")
    print(f"║  {'Flagged for review':<30} {flagged:>6}  ({pct_flag})   ║")
    if errors:
        print(f"║  {'Errors (files skipped)':<30} {errors:>6}          ║")
    print(f"╚{bar}╝")
    if flagged:
        print(f"\n  → Run  python run_agent.py --review  to inspect flagged fields.")
        print(f"  → Run  python run_agent.py --approve review/review_queue.csv  to approve.\n")


# ── argument parsing ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_agent.py",
        description="Fund Report Extraction Agent — powered by Claude AI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mutually exclusive primary commands
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--input",
        metavar="PATH",
        help=(
            "PDF file or directory to process. "
            "Directories are scanned recursively for *.pdf files."
        ),
    )
    mode.add_argument(
        "--review",
        action="store_true",
        help="Print a formatted table of all items in review_queue.csv.",
    )
    mode.add_argument(
        "--approve",
        metavar="CSV",
        help=(
            "Path to review_queue.csv. "
            "Prompts Y/N per unreviewed row and writes back a 'human_reviewed' column."
        ),
    )

    # Shared options
    parser.add_argument(
        "--output-dir",
        default="output",
        metavar="DIR",
        help="Where to write accepted extraction JSON files (default: output/).",
    )
    parser.add_argument(
        "--review-dir",
        default="review",
        metavar="DIR",
        help="Where review_queue.csv lives (default: review/).",
    )

    return parser


# ── command: --input ──────────────────────────────────────────────────────────

def cmd_input(args: argparse.Namespace) -> int:
    """Run extraction on a PDF file or directory."""
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error(f"Path not found: {input_path}")
        return 1

    try:
        agent = ExtractionAgent(
            reports_dir=str(input_path.parent) if input_path.is_file() else str(input_path),
            output_dir=args.output_dir,
            review_dir=args.review_dir,
        )
    except EnvironmentError as exc:
        logger.error(str(exc))
        return 1

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            logger.error(f"Not a PDF file: {input_path}")
            return 1

        logger.info(f"Single-file mode: {input_path.name}")
        result = agent.process_single(str(input_path))
        if result is None:
            print("\n✗ Extraction failed — check the log output above.")
            return 1

        # Synthesise a summary dict for _print_run_summary
        n_accepted = len(result.get("fields", {}))
        n_flagged = _count_flagged_in_csv(
            args.review_dir, filename=input_path.name
        )
        summary = {
            "processed": 1,
            "flagged_fields": n_flagged,
            "errors": 0,
        }
        _print_run_summary(summary, args.output_dir, [input_path.stem])

    else:
        # Directory: collect stems before the run so we can look up output files
        pdf_files = sorted(input_path.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No *.pdf files found in {input_path}")
            return 0

        stems = [p.stem for p in pdf_files]
        summary = agent.run()
        _print_run_summary(summary, args.output_dir, stems)

    return 0


def _count_flagged_in_csv(review_dir: str, filename: Optional[str] = None) -> int:
    """Return the number of flagged rows in review_queue.csv, optionally filtered."""
    csv_path = Path(review_dir) / "review_queue.csv"
    if not csv_path.exists():
        return 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if filename:
        rows = [r for r in rows if r.get("filename") == filename]
    return len(rows)


# ── command: --review ─────────────────────────────────────────────────────────

def cmd_review(args: argparse.Namespace) -> int:
    """Print a formatted table of every row in review_queue.csv."""
    csv_path = Path(args.review_dir) / "review_queue.csv"

    if not csv_path.exists():
        print(f"\n  No review queue found at {csv_path}")
        print("  Run  python run_agent.py --input reports/  first.\n")
        return 0

    rows, _ = _load_review_csv(csv_path)

    if not rows:
        print("\n  ✓ Review queue is empty — nothing to review.\n")
        return 0

    # Separate into groups: unreviewed / approved / rejected
    unreviewed = [r for r in rows if not r.get("human_reviewed")]
    approved   = [r for r in rows if r.get("human_reviewed") == "Y"]
    rejected   = [r for r in rows if r.get("human_reviewed") == "N"]

    print(f"\n{'═' * 64}")
    print(f"  REVIEW QUEUE  —  {len(rows)} total flagged field(s)")
    print(
        f"  Unreviewed: {len(unreviewed)}  │  "
        f"Approved: {len(approved)}  │  "
        f"Rejected: {len(rejected)}"
    )
    print(f"{'═' * 64}\n")

    for section_label, section_rows in [
        ("UNREVIEWED", unreviewed),
        ("APPROVED",   approved),
        ("REJECTED",   rejected),
    ]:
        if not section_rows:
            continue

        print(f"  ── {section_label} ({len(section_rows)}) {'─' * (52 - len(section_label))}")
        print()

        table_headers = ["#", "File", "Field", "Value", "Conf", "Reason", "Reviewed"]
        table_rows = [
            [
                str(i + 1),
                r.get("filename", ""),
                r.get("field_name", ""),
                r.get("extracted_value", "") or "—",
                r.get("confidence", ""),
                r.get("reason_for_flag", ""),
                r.get("human_reviewed", "") or "—",
            ]
            for i, r in enumerate(section_rows)
        ]

        print(
            _render_table(
                table_headers,
                table_rows,
                max_widths={"File": 22, "Field": 22, "Value": 16, "Reason": 42},
            )
        )
        print()

    # Footer hint
    if unreviewed:
        print(
            f"  → {len(unreviewed)} item(s) still need review.\n"
            f"     Run:  python run_agent.py --approve {csv_path}\n"
        )

    return 0


# ── command: --approve ────────────────────────────────────────────────────────

def cmd_approve(args: argparse.Namespace) -> int:
    """
    Interactive Y/N approval loop.

    For every row in the CSV that has no human_reviewed value, display
    the row details and prompt the reviewer.  Write back the updated CSV.
    """
    csv_path = Path(args.approve)

    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return 1

    rows, fieldnames = _load_review_csv(csv_path)

    # Ensure approve columns exist in the schema
    for col in _APPROVE_COLS:
        if col not in fieldnames:
            fieldnames.append(col)
        for row in rows:
            row.setdefault(col, "")

    pending = [r for r in rows if not r.get("human_reviewed")]

    if not pending:
        print("\n  ✓ All rows already reviewed — nothing to do.\n")
        _print_approve_summary(rows)
        return 0

    already_done = len(rows) - len(pending)
    print(f"\n{'═' * 60}")
    print(f"  APPROVAL SESSION  —  {len(pending)} unreviewed row(s)")
    if already_done:
        print(f"  ({already_done} row(s) already reviewed, skipping those)")
    print(f"{'═' * 60}")
    print("  Keys:  [Y] approve  [N] reject  [S] skip  [Q] quit\n")

    newly_approved = 0
    newly_rejected = 0
    skipped = 0

    for idx, row in enumerate(rows):
        if row.get("human_reviewed"):
            continue  # already decided — leave untouched

        item_num = idx + 1
        print(f"  ── Item {item_num} of {len(rows)} {'─' * 46}")
        print(f"  File      : {row.get('filename', '—')}")
        print(f"  Field     : {row.get('field_name', '—')}")
        value = row.get("extracted_value") or "(not found / null)"
        print(f"  Value     : {value}")
        print(f"  Confidence: {row.get('confidence', '—')}")
        print(f"  Reason    : {row.get('reason_for_flag', '—')}")
        print()

        while True:
            try:
                raw = input("  Decision [Y/N/S/Q] > ").strip().upper()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  Interrupted — saving progress.")
                _save_review_csv(csv_path, rows, fieldnames)
                _print_approve_summary(rows)
                return 0

            if raw in ("Y", "N", "S", "Q"):
                break
            print("  Please enter Y, N, S, or Q.")

        if raw == "Q":
            print("\n  Quitting — saving progress made so far.")
            break

        if raw == "S":
            skipped += 1
            print()
            continue

        # Y or N — optionally capture notes
        try:
            notes = input("  Notes (optional, press Enter to skip) > ").strip()
        except (EOFError, KeyboardInterrupt):
            notes = ""

        row["human_reviewed"] = raw
        row["review_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row["reviewer_notes"] = notes

        if raw == "Y":
            newly_approved += 1
            print(f"  ✓ Approved\n")
        else:
            newly_rejected += 1
            print(f"  ✗ Rejected\n")

    _save_review_csv(csv_path, rows, fieldnames)
    print(f"\n  Saved: {csv_path}\n")
    _print_approve_summary(rows, newly_approved, newly_rejected, skipped)
    return 0


def _print_approve_summary(
    rows: List[Dict[str, str]],
    newly_approved: int = 0,
    newly_rejected: int = 0,
    skipped: int = 0,
) -> None:
    total      = len(rows)
    approved   = sum(1 for r in rows if r.get("human_reviewed") == "Y")
    rejected   = sum(1 for r in rows if r.get("human_reviewed") == "N")
    unreviewed = sum(1 for r in rows if not r.get("human_reviewed"))

    width = 52
    bar = "═" * width
    print(f"╔{bar}╗")
    print(f"║{'  APPROVAL SUMMARY':^{width}}║")
    print(f"╠{bar}╣")
    print(f"║  {'Total flagged fields':<28} {total:>6}              ║")
    print(f"║  {'Approved (Y)':<28} {approved:>6}              ║")
    print(f"║  {'Rejected (N)':<28} {rejected:>6}              ║")
    print(f"║  {'Still unreviewed':<28} {unreviewed:>6}              ║")
    if newly_approved or newly_rejected or skipped:
        print(f"╠{bar}╣")
        print(f"║  {'This session — approved':<28} {newly_approved:>6}              ║")
        print(f"║  {'This session — rejected':<28} {newly_rejected:>6}              ║")
        print(f"║  {'This session — skipped':<28} {skipped:>6}              ║")
    print(f"╚{bar}╝\n")


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _load_review_csv(
    path: Path,
) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Read review_queue.csv and return (rows, fieldnames).

    Missing approval columns are silently filled with empty strings.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames: List[str] = list(reader.fieldnames or _AGENT_COLS)
        rows: List[Dict[str, str]] = [dict(r) for r in reader]
    return rows, fieldnames


def _save_review_csv(
    path: Path,
    rows: List[Dict[str, str]],
    fieldnames: List[str],
) -> None:
    """Write *rows* back to *path*, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.input:
        return cmd_input(args)
    if args.review:
        return cmd_review(args)
    if args.approve:
        return cmd_approve(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
