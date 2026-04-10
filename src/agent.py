"""
agent.py
========
Main agent loop for the fund-report extraction pipeline.

Responsibilities
----------------
1. Scan ``reports/`` for new PDF files.
2. Run extraction + validation on each file via FinancialDataExtractor.
3. Route results:
   - High-confidence extractions  → ``output/`` (CSV + Excel)
   - Low-confidence / errors      → ``review/`` via ReviewQueue
4. Log a run summary to stdout and to ``output/run_log.jsonl``.

Typical usage (see run_agent.py for the CLI wrapper):

    from src.agent import ExtractionAgent
    agent = ExtractionAgent()
    agent.run()
"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .extractor import FinancialDataExtractor, FinancialMetrics
from .review_queue import ReviewQueue

logger = logging.getLogger(__name__)


class ExtractionAgent:
    """
    Orchestrates end-to-end extraction for every PDF in *reports_dir*.

    Parameters
    ----------
    reports_dir:
        Directory to scan for input PDF files (default ``reports/``).
    output_dir:
        Destination for accepted structured output (default ``output/``).
    review_dir:
        Destination for flagged items requiring human review
        (default ``review/``).
    confidence_threshold:
        Extractions whose ``extraction_confidence`` falls below this
        value are routed to the review queue rather than written to
        output (default ``0.6``).
    """

    def __init__(
        self,
        reports_dir: str = "reports",
        output_dir: str = "output",
        review_dir: str = "review",
        confidence_threshold: float = 0.6,
    ):
        self.reports_dir = Path(reports_dir)
        self.output_dir = Path(output_dir)
        self.review_dir = Path(review_dir)
        self.confidence_threshold = confidence_threshold

        # Ensure directories exist
        for d in (self.reports_dir, self.output_dir, self.review_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.extractor = FinancialDataExtractor(output_dir=str(self.output_dir))
        self.review_queue = ReviewQueue(review_dir=str(self.review_dir))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, pdf_paths: Optional[List[str]] = None) -> dict:
        """
        Run the extraction agent.

        Args:
            pdf_paths: Optional explicit list of PDF paths to process.
                       When omitted all ``*.pdf`` files in *reports_dir*
                       are used.

        Returns:
            Summary dict with keys ``processed``, ``accepted``,
            ``flagged``, ``errors``.
        """
        if pdf_paths:
            files = [Path(p) for p in pdf_paths]
        else:
            files = sorted(self.reports_dir.glob("*.pdf"))

        if not files:
            logger.warning(f"No PDF files found in {self.reports_dir}")
            return {"processed": 0, "accepted": 0, "flagged": 0, "errors": 0}

        logger.info(f"Agent starting – {len(files)} file(s) to process")

        accepted: List[FinancialMetrics] = []
        flagged = 0
        errors = 0

        for pdf_path in files:
            result = self._process_file(pdf_path)
            if result is None:
                errors += 1
            elif result["routed_to"] == "output":
                accepted.append(result["metrics"])
            else:
                flagged += 1

        # Persist accepted results
        if accepted:
            self._save_output(accepted)

        summary = {
            "processed": len(files),
            "accepted": len(accepted),
            "flagged": flagged,
            "errors": errors,
            "run_timestamp": datetime.now().isoformat(),
        }
        self._log_run(summary)
        logger.info(f"Agent finished – {summary}")
        return summary

    def process_single(self, pdf_path: str) -> Optional[FinancialMetrics]:
        """
        Convenience wrapper to extract a single PDF and return its metrics.

        The result is still routed (output vs review) exactly as in
        batch mode.

        Args:
            pdf_path: Absolute or relative path to the PDF.

        Returns:
            FinancialMetrics on success, None on error.
        """
        result = self._process_file(Path(pdf_path))
        if result:
            if result["routed_to"] == "output":
                self._save_output([result["metrics"]])
            return result["metrics"]
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_file(self, pdf_path: Path) -> Optional[dict]:
        """Extract one PDF, validate it, and route the result."""
        logger.info(f"Processing: {pdf_path.name}")
        try:
            metrics = self.extractor.process_pdf(
                str(pdf_path), company_name=pdf_path.stem
            )
            validation = self.extractor.validator.validate(metrics)
            routed_to = self._route(metrics, validation)
            return {"metrics": metrics, "validation": validation, "routed_to": routed_to}
        except Exception as exc:
            logger.error(f"Failed to process {pdf_path.name}: {exc}")
            return None

    def _route(self, metrics: FinancialMetrics, validation: dict) -> str:
        """
        Decide where a result goes.

        Rules:
        - Any hard validation error                  → review
        - extraction_confidence < threshold          → review
        - Otherwise                                  → output
        """
        if validation.get("errors"):
            reason = f"Validation errors: {'; '.join(validation['errors'])}"
            self.review_queue.add(metrics, validation, reason=reason)
            logger.warning(f"Flagged for review ({metrics.company_name}): {reason}")
            return "review"

        if metrics.extraction_confidence < self.confidence_threshold:
            reason = (
                f"Low confidence ({metrics.extraction_confidence:.2f} "
                f"< {self.confidence_threshold})"
            )
            self.review_queue.add(metrics, validation, reason=reason)
            logger.warning(f"Flagged for review ({metrics.company_name}): {reason}")
            return "review"

        return "output"

    def _save_output(self, metrics_list: List[FinancialMetrics]) -> None:
        """Write accepted metrics to CSV and Excel in output_dir."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.extractor.export_to_csv(metrics_list, f"financial_data_{ts}.csv")
        self.extractor.export_to_excel(metrics_list, f"financial_data_{ts}.xlsx")

    def _log_run(self, summary: dict) -> None:
        """Append run summary to a newline-delimited JSON log file."""
        log_path = self.output_dir / "run_log.jsonl"
        with open(log_path, "a") as fh:
            fh.write(json.dumps(summary) + "\n")
