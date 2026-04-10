#!/usr/bin/env python3
"""
run_agent.py
============
CLI entry point for the fund-report extraction agent.

Examples
--------
# Process all PDFs in reports/  (default)
python run_agent.py

# Process a single file
python run_agent.py --file reports/q3_2024.pdf

# Use custom directories
python run_agent.py --reports-dir data/pdfs --output-dir data/out --review-dir data/review

# Show how many fields are currently flagged in the review queue
python run_agent.py --review-status

# Print all flagged rows as JSON
python run_agent.py --list-flagged
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import ExtractionAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fund Report Extraction Agent — extract structured financial data "
            "from PDF reports using Claude AI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input / output paths ──────────────────────────────────────────────
    parser.add_argument(
        "--file", "-f",
        metavar="PDF",
        help="Process a single PDF file instead of the whole reports/ directory.",
    )
    parser.add_argument(
        "--reports-dir",
        default="reports",
        metavar="DIR",
        help="Directory to scan for PDF files (default: reports/).",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        metavar="DIR",
        help="Directory for accepted JSON output (default: output/).",
    )
    parser.add_argument(
        "--review-dir",
        default="review",
        metavar="DIR",
        help="Directory for review_queue.csv (default: review/).",
    )

    # ── Review-queue inspection ───────────────────────────────────────────
    rq = parser.add_argument_group("Review queue")
    rq.add_argument(
        "--review-status",
        action="store_true",
        help="Print a count of rows currently in review_queue.csv and exit.",
    )
    rq.add_argument(
        "--list-flagged",
        action="store_true",
        help="Print all rows in review_queue.csv as JSON and exit.",
    )

    return parser.parse_args()


def _review_csv_path(review_dir: str) -> Path:
    return Path(review_dir) / "review_queue.csv"


def cmd_review_status(review_dir: str) -> int:
    """Print the number of flagged rows in review_queue.csv."""
    csv_path = _review_csv_path(review_dir)
    if not csv_path.exists():
        print(json.dumps({"flagged_rows": 0, "note": "review_queue.csv not yet created"}))
        return 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # Count per-field, per-document
    by_file: dict = {}
    for row in rows:
        by_file.setdefault(row["filename"], 0)
        by_file[row["filename"]] += 1
    print(json.dumps({"flagged_rows": len(rows), "by_file": by_file}, indent=2))
    return 0


def cmd_list_flagged(review_dir: str) -> int:
    """Print all rows in review_queue.csv as a JSON array."""
    csv_path = _review_csv_path(review_dir)
    if not csv_path.exists():
        print("[]")
        return 0
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    args = parse_args()

    # ── Review-queue read commands (no extraction needed) ─────────────────
    if args.review_status:
        return cmd_review_status(args.review_dir)

    if args.list_flagged:
        return cmd_list_flagged(args.review_dir)

    # ── Extraction run ────────────────────────────────────────────────────
    try:
        agent = ExtractionAgent(
            reports_dir=args.reports_dir,
            output_dir=args.output_dir,
            review_dir=args.review_dir,
        )
    except EnvironmentError as exc:
        logger.error(str(exc))
        return 1

    if args.file:
        result = agent.process_single(args.file)
        if result is None:
            print("Extraction failed — check logs for details.")
            return 1
        print(f"\nExtraction complete: {args.file}")
        print(f"  Fields accepted : {len(result.get('fields', {}))}")
        print(f"  Output          : {Path(args.output_dir) / (Path(args.file).stem + '.json')}")
        flagged = sum(1 for _ in open(_review_csv_path(args.review_dir))) - 1  # minus header
        print(f"  Review queue    : {max(flagged, 0)} total flagged row(s)")
    else:
        summary = agent.run()
        print("\n── Run summary ───────────────────────────────────────")
        for key, value in summary.items():
            print(f"  {key:<18}: {value}")
        print("──────────────────────────────────────────────────────")

    return 0


if __name__ == "__main__":
    sys.exit(main())
