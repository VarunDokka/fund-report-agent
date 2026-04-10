#!/usr/bin/env python3
"""
run_agent.py
============
CLI entry point for the fund-report extraction agent.

Examples
--------
# Process all PDFs in reports/ (default)
python run_agent.py

# Process a specific file
python run_agent.py --file reports/q3_2024.pdf

# Use custom directories and a stricter confidence threshold
python run_agent.py --reports-dir data/pdfs --output-dir data/out --threshold 0.75

# List items currently waiting for human review
python run_agent.py --review-status
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import ExtractionAgent
from src.review_queue import ReviewQueue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fund Report Extraction Agent – extract structured financial "
                    "data from PDF reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input / output
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
        help="Directory for accepted structured output (default: output/).",
    )
    parser.add_argument(
        "--review-dir",
        default="review",
        metavar="DIR",
        help="Directory for flagged items (default: review/).",
    )

    # Agent behaviour
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        metavar="FLOAT",
        help="Confidence threshold below which results are sent to review "
             "(default: 0.6).",
    )

    # Review-queue commands
    review_group = parser.add_argument_group("Review queue management")
    review_group.add_argument(
        "--review-status",
        action="store_true",
        help="Print a summary of pending / approved / rejected review items and exit.",
    )
    review_group.add_argument(
        "--list-pending",
        action="store_true",
        help="Print all pending review items as JSON and exit.",
    )
    review_group.add_argument(
        "--approve",
        metavar="ITEM_ID",
        help="Approve a pending review item by its ID.",
    )
    review_group.add_argument(
        "--reject",
        metavar="ITEM_ID",
        help="Reject a pending review item by its ID.",
    )
    review_group.add_argument(
        "--notes",
        default="",
        help="Optional reviewer notes to attach when approving or rejecting.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    review_queue = ReviewQueue(review_dir=args.review_dir)

    # ── Review-queue management commands ────────────────────────────────
    if args.review_status:
        summary = review_queue.summary()
        print(json.dumps(summary, indent=2))
        return 0

    if args.list_pending:
        pending = review_queue.list_pending()
        print(json.dumps(pending, indent=2, default=str))
        return 0

    if args.approve:
        ok = review_queue.approve(args.approve, notes=args.notes)
        print("Approved." if ok else "Item not found.")
        return 0 if ok else 1

    if args.reject:
        ok = review_queue.reject(args.reject, notes=args.notes)
        print("Rejected." if ok else "Item not found.")
        return 0 if ok else 1

    # ── Extraction run ───────────────────────────────────────────────────
    agent = ExtractionAgent(
        reports_dir=args.reports_dir,
        output_dir=args.output_dir,
        review_dir=args.review_dir,
        confidence_threshold=args.threshold,
    )

    if args.file:
        pdf_path = Path(args.file)
        if not pdf_path.exists():
            logger.error(f"File not found: {pdf_path}")
            return 1
        metrics = agent.process_single(str(pdf_path))
        if metrics:
            print(f"\nExtraction complete: {metrics.company_name}")
            print(f"  Confidence : {metrics.extraction_confidence:.2f}")
            print(f"  Source     : {metrics.source_file}")
        else:
            print("Extraction failed – check logs for details.")
            return 1
    else:
        summary = agent.run()
        print("\n── Run summary ─────────────────────────────────")
        for key, value in summary.items():
            print(f"  {key:<12}: {value}")
        print("────────────────────────────────────────────────")

    return 0


if __name__ == "__main__":
    sys.exit(main())
