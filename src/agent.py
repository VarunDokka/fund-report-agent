"""
agent.py
========
Agentic extraction loop powered by the Anthropic Claude API.

Pipeline (per PDF in reports/)
-------------------------------
1. Ingest the PDF with pdfplumber (text + table content)
2. Call claude-sonnet-4-6 via the Anthropic SDK (streaming + forced tool use)
   to extract: NAV, capital called, distributions, IRR, MOIC,
   gross/net returns, portfolio company count, top holdings,
   plus balance-sheet fields required for DataValidator checks.
3. Assign per-field confidence: HIGH / MEDIUM / LOW
     HIGH   – value directly and unambiguously stated in text
     MEDIUM – value inferred, calculated, or mentioned with some ambiguity
     LOW    – value missing, conflicting, or unclear
4. Run DataValidator checks (balance-sheet equation, net-debt calc,
   EBITDA-margin reasonability, range checks)
5. Fields with LOW confidence OR validation failures
   → review/review_queue.csv  (filename, field_name, extracted_value,
                                confidence, reason_for_flag)
6. Fields with HIGH / MEDIUM confidence and no validation failures
   → output/{filename}.json
7. Every step is logged to console with timestamps.

Usage
-----
    from src.agent import ExtractionAgent
    agent = ExtractionAgent()
    summary = agent.run()

Note on model: the user-specified model (claude-3-5-sonnet) is retired as of
Oct 2025.  The closest current equivalent, claude-sonnet-4-6, is used instead.

Author: Arnab Banerjee
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import anthropic
import pdfplumber
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator

from .extractor import FinancialMetrics
from .validator import DataValidator

# ── Environment & logging ──────────────────────────────────────────────────────

load_dotenv()  # reads ANTHROPIC_API_KEY (and anything else) from .env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# claude-3-5-sonnet is retired (Oct 2025); claude-sonnet-4-6 is the current equivalent
MODEL = "claude-sonnet-4-6"

# ── Confidence type ────────────────────────────────────────────────────────────

Confidence = Literal["HIGH", "MEDIUM", "LOW"]

# ── Pydantic models for structured output ─────────────────────────────────────


class FieldExtraction(BaseModel):
    """A single extracted metric with provenance metadata."""

    value: Optional[str] = None
    confidence: Confidence
    reason: str  # why this confidence level was assigned


class FundMetrics(BaseModel):
    """All fields extracted from one fund report."""

    # ── Primary fund metrics (per user specification) ──
    nav: FieldExtraction
    capital_called: FieldExtraction
    distributions: FieldExtraction
    irr: FieldExtraction
    moic: FieldExtraction
    gross_returns: FieldExtraction
    net_returns: FieldExtraction
    portfolio_company_count: FieldExtraction
    top_holdings: FieldExtraction

    # ── Balance-sheet / income fields (for DataValidator) ──
    total_assets: FieldExtraction
    total_liabilities: FieldExtraction
    total_equity: FieldExtraction
    total_debt: FieldExtraction
    cash_and_equivalents: FieldExtraction
    net_debt: FieldExtraction
    revenue: FieldExtraction
    ebitda: FieldExtraction


# ── Tool schema (explicit JSON Schema avoids $defs / $ref edge-cases) ─────────

_FIELD = {
    "type": "object",
    "properties": {
        "value": {
            "type": ["string", "null"],
            "description": "Extracted value as a string (e.g. '$2.3B', '12.5%', '2.1x'), or null if not found.",
        },
        "confidence": {
            "type": "string",
            "enum": ["HIGH", "MEDIUM", "LOW"],
            "description": (
                "HIGH = directly stated in text without ambiguity; "
                "MEDIUM = inferred or calculated; "
                "LOW = not found, missing, or conflicting."
            ),
        },
        "reason": {
            "type": "string",
            "description": "Brief explanation for the confidence level assigned.",
        },
    },
    "required": ["value", "confidence", "reason"],
    "additionalProperties": False,
}

EXTRACTION_TOOL: Dict[str, Any] = {
    "name": "record_fund_metrics",
    "description": (
        "Record every extracted fund metric together with its confidence level. "
        "Must be called for every document — set value=null and confidence=LOW "
        "for any field not found in the text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            # Primary fund fields
            "nav": {**_FIELD, "description": "Net Asset Value (NAV) of the fund."},
            "capital_called": {**_FIELD, "description": "Total capital called / drawn down from LPs."},
            "distributions": {**_FIELD, "description": "Total distributions returned to LPs."},
            "irr": {**_FIELD, "description": "Internal Rate of Return (IRR); note gross vs net if labelled."},
            "moic": {**_FIELD, "description": "Multiple on Invested Capital (MOIC / TVPI / investment multiple)."},
            "gross_returns": {**_FIELD, "description": "Gross return or gross IRR as stated."},
            "net_returns": {**_FIELD, "description": "Net return or net IRR as stated."},
            "portfolio_company_count": {**_FIELD, "description": "Number of portfolio companies in the fund."},
            "top_holdings": {
                **_FIELD,
                "description": (
                    "Comma-separated list of top portfolio company names if mentioned; "
                    "null if not present."
                ),
            },
            # Balance-sheet / income fields
            "total_assets": {**_FIELD, "description": "Total assets (balance sheet)."},
            "total_liabilities": {**_FIELD, "description": "Total liabilities (balance sheet)."},
            "total_equity": {**_FIELD, "description": "Total equity / shareholders' equity / net assets."},
            "total_debt": {**_FIELD, "description": "Total debt or total borrowings."},
            "cash_and_equivalents": {**_FIELD, "description": "Cash and cash equivalents."},
            "net_debt": {**_FIELD, "description": "Net debt (total debt minus cash)."},
            "revenue": {**_FIELD, "description": "Total revenue or net revenue."},
            "ebitda": {**_FIELD, "description": "EBITDA (earnings before interest, taxes, depreciation, amortisation)."},
        },
        "required": [
            "nav", "capital_called", "distributions", "irr", "moic",
            "gross_returns", "net_returns", "portfolio_company_count", "top_holdings",
            "total_assets", "total_liabilities", "total_equity",
            "total_debt", "cash_and_equivalents", "net_debt",
            "revenue", "ebitda",
        ],
        "additionalProperties": False,
    },
}

# ── Review CSV columns ─────────────────────────────────────────────────────────

_REVIEW_HEADERS = [
    "filename",
    "field_name",
    "extracted_value",
    "confidence",
    "reason_for_flag",
]

# ── Numeric parsing ────────────────────────────────────────────────────────────


def _parse_numeric(value: Optional[str]) -> Optional[float]:
    """
    Convert a free-form value string to float.

    Handles common fund-report notations:
      "$2.3B"  → 2_300_000_000.0
      "450M"   → 450_000_000.0
      "12.5%"  → 12.5
      "2.1x"   → 2.1
      "1,234"  → 1234.0
    """
    if not value:
        return None
    s = value.strip()
    # Remove currency symbols and thousands commas
    s = re.sub(r"[$£€,]", "", s)
    # Strip trailing unit markers and note multiplier
    multiplier = 1.0
    for suffix, mult in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if s.upper().endswith(suffix):
            s = s[:-1]
            multiplier = mult
            break
    # Strip % and x/X (MOIC)
    s = re.sub(r"[%xX]", "", s).strip()
    try:
        return float(s) * multiplier
    except (ValueError, TypeError):
        return None


# ── Agent ──────────────────────────────────────────────────────────────────────


class ExtractionAgent:
    """
    Agentic loop: for every PDF in *reports_dir*, ingest → extract →
    validate → route each field to output JSON or the review CSV.

    Parameters
    ----------
    reports_dir : str
        Folder to scan for ``*.pdf`` files (default ``"reports"``).
    output_dir : str
        Destination for accepted per-document JSON files (default ``"output"``).
    review_dir : str
        Destination for ``review_queue.csv`` (default ``"review"``).
    """

    def __init__(
        self,
        reports_dir: str = "reports",
        output_dir: str = "output",
        review_dir: str = "review",
    ) -> None:
        self.reports_dir = Path(reports_dir)
        self.output_dir = Path(output_dir)
        self.review_dir = Path(review_dir)

        for d in (self.reports_dir, self.output_dir, self.review_dir):
            d.mkdir(parents=True, exist_ok=True)

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to your .env file or export it in your shell."
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.validator = DataValidator()

        self.review_csv = self.review_dir / "review_queue.csv"
        self._init_review_csv()

    # ── Public interface ───────────────────────────────────────────────────────

    def run(self, pdf_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the agent against every PDF in *reports_dir* (or *pdf_paths*).

        Returns
        -------
        dict
            Keys: ``processed``, ``output_files``, ``flagged_fields``, ``errors``.
        """
        files: List[Path] = (
            [Path(p) for p in pdf_paths]
            if pdf_paths
            else sorted(self.reports_dir.glob("*.pdf"))
        )

        if not files:
            logger.warning(
                f"[{_now()}] No PDF files found in '{self.reports_dir}'. "
                "Drop PDF reports there and re-run."
            )
            return {"processed": 0, "output_files": 0, "flagged_fields": 0, "errors": 0}

        logger.info(
            f"[{_now()}] ══ ExtractionAgent starting — "
            f"{len(files)} PDF(s) · model={MODEL} ══"
        )

        output_files = 0
        flagged_total = 0
        errors = 0

        for pdf_path in files:
            logger.info(f"[{_now()}] ┌─ {pdf_path.name}")
            try:
                flagged = self._process_file(pdf_path)
                output_files += 1
                flagged_total += flagged
            except anthropic.AuthenticationError:
                logger.error(
                    f"[{_now()}] │  Authentication failed — check ANTHROPIC_API_KEY."
                )
                errors += 1
            except anthropic.RateLimitError:
                logger.error(
                    f"[{_now()}] │  Rate-limited by Anthropic API — retry later."
                )
                errors += 1
            except anthropic.APIError as exc:
                logger.error(f"[{_now()}] │  Anthropic API error: {exc}")
                errors += 1
            except Exception as exc:
                logger.error(f"[{_now()}] │  Unexpected error: {exc}", exc_info=True)
                errors += 1
            logger.info(f"[{_now()}] └─ Done: {pdf_path.name}")

        summary = {
            "processed": len(files),
            "output_files": output_files,
            "flagged_fields": flagged_total,
            "errors": errors,
            "run_timestamp": datetime.now().isoformat(),
        }
        self._append_run_log(summary)
        logger.info(f"[{_now()}] ══ Agent finished — {summary} ══")
        return summary

    def process_single(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """
        Extract a single PDF.

        Returns the clean-fields dict written to output/, or None on error.
        """
        p = Path(pdf_path)
        if not p.exists():
            logger.error(f"[{_now()}] File not found: {p}")
            return None
        logger.info(f"[{_now()}] ┌─ {p.name} (single-file mode)")
        try:
            self._process_file(p)
            output_path = self.output_dir / f"{p.stem}.json"
            with open(output_path, encoding="utf-8") as fh:
                result = json.load(fh)
            logger.info(f"[{_now()}] └─ Done")
            return result
        except Exception as exc:
            logger.error(f"[{_now()}] └─ Error: {exc}", exc_info=True)
            return None

    # ── Step 1 — Ingest PDF ────────────────────────────────────────────────────

    def _ingest_pdf(self, pdf_path: Path) -> str:
        """Extract all text and table content from a PDF with pdfplumber."""
        logger.info(f"[{_now()}] │  Step 1 – Ingesting PDF…")

        pages: List[str] = []
        page_count = 0

        with pdfplumber.open(pdf_path) as pdf:
            page_count = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""

                # Serialise tables as pipe-delimited rows so Claude can read them
                table_lines: List[str] = []
                for table in page.extract_tables() or []:
                    for row in table:
                        table_lines.append(
                            " | ".join(str(cell or "").strip() for cell in row)
                        )

                page_str = f"=== Page {i} ===\n{text}"
                if table_lines:
                    page_str += "\n[TABLES]\n" + "\n".join(table_lines)
                pages.append(page_str)

        full_text = "\n\n".join(pages)
        logger.info(
            f"[{_now()}] │  Extracted {page_count} page(s) — "
            f"{len(full_text):,} chars"
        )
        return full_text

    # ── Steps 2 & 3 — Extract with Claude and assign confidence ───────────────

    _SYSTEM_PROMPT = """\
You are a senior financial analyst specialising in private equity and credit fund reporting.
Your task is to extract specific metrics from the text of a fund report PDF.

Confidence rules — apply exactly:
  HIGH   – the value is explicitly and unambiguously stated in the source text.
  MEDIUM – the value is implied, inferred from context, or calculated from
           other stated figures.
  LOW    – the value is absent, ambiguous, or contradicted elsewhere in the document.

Additional rules:
• Preserve original notation (e.g. "$2.3B", "12.5%", "2.1x").
• For top_holdings: return a comma-separated list of names; null if not mentioned.
• For every field: if the value is null, confidence MUST be LOW.
• Populate the 'reason' field with a concise explanation of your confidence level.
• Do not guess. Do not hallucinate numbers not present in the text.\
"""

    def _extract_with_claude(self, text: str, filename: str) -> FundMetrics:
        """
        Call Claude (streaming, forced tool use) and return validated FundMetrics.

        Steps 2 (extraction) and 3 (confidence assignment) happen here.
        """
        logger.info(
            f"[{_now()}] │  Step 2 – Calling {MODEL} for extraction + "
            "confidence scoring…"
        )

        # Truncate very long documents to stay well inside the context window
        max_chars = 60_000
        if len(text) > max_chars:
            logger.warning(
                f"[{_now()}] │  Document truncated to {max_chars:,} chars "
                f"(original: {len(text):,})"
            )
            text = text[:max_chars] + "\n\n[DOCUMENT TRUNCATED]"

        user_message = (
            f"Extract financial metrics from the following fund report.\n"
            f"File: {filename}\n\n"
            "DOCUMENT:\n"
            '"""\n'
            f"{text}\n"
            '"""\n\n'
            "Call `record_fund_metrics` with every field populated. "
            "Set value=null and confidence=LOW for any field absent from the document."
        )

        # Stream the response; print dots to show live progress
        logger.info(f"[{_now()}] │  Streaming…  ", )
        print(f"  [{_now()}]   ", end="", flush=True)

        with self.client.messages.stream(
            model=MODEL,
            max_tokens=4_096,
            system=self._SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "record_fund_metrics"},
        ) as stream:
            for event in stream:
                # Print a dot for every JSON chunk to show live progress
                if (
                    hasattr(event, "type")
                    and event.type == "content_block_delta"
                    and hasattr(event, "delta")
                    and getattr(event.delta, "type", None) == "input_json_delta"
                ):
                    print(".", end="", flush=True)
            print()  # newline after progress dots

            final_message = stream.get_final_message()

        logger.info(
            f"[{_now()}] │  API call complete — "
            f"input_tokens={final_message.usage.input_tokens}, "
            f"output_tokens={final_message.usage.output_tokens}"
        )

        # Extract the tool input from the response
        tool_input: Optional[Dict[str, Any]] = None
        for block in final_message.content:
            if (
                block.type == "tool_use"
                and block.name == "record_fund_metrics"
            ):
                tool_input = block.input
                break

        if tool_input is None:
            raise ValueError(
                "Claude did not return a 'record_fund_metrics' tool call. "
                "Response content: " + str(final_message.content)
            )

        logger.info(f"[{_now()}] │  Step 3 – Parsing confidence levels…")
        metrics = FundMetrics.model_validate(tool_input)

        # Log a one-line confidence summary
        conf_summary = {
            name: field.confidence
            for name, field in metrics.model_dump().items()
            if isinstance(field, dict)
        }
        # Rebuild from model fields for clean display
        conf_summary = {
            name: getattr(metrics, name).confidence
            for name in metrics.model_fields
        }
        high = sum(1 for c in conf_summary.values() if c == "HIGH")
        med  = sum(1 for c in conf_summary.values() if c == "MEDIUM")
        low  = sum(1 for c in conf_summary.values() if c == "LOW")
        logger.info(
            f"[{_now()}] │  Confidence summary — "
            f"HIGH={high}, MEDIUM={med}, LOW={low}"
        )

        return metrics

    # ── Step 4 — Validation ────────────────────────────────────────────────────

    def _run_validation(
        self, metrics: FundMetrics
    ) -> Dict[str, List[str]]:
        """
        Run DataValidator checks and return a dict mapping field name →
        list of failure strings (errors and/or warnings).
        """
        logger.info(f"[{_now()}] │  Step 4 – Running DataValidator checks…")

        # Build a FinancialMetrics dataclass from the extracted values so the
        # existing validator can operate on it without modification
        fm = FinancialMetrics(
            company_name="fund",
            report_date=datetime.now().strftime("%Y-%m-%d"),
            report_type="Fund Report",
            total_assets=_parse_numeric(metrics.total_assets.value),
            total_liabilities=_parse_numeric(metrics.total_liabilities.value),
            total_equity=_parse_numeric(metrics.total_equity.value),
            total_debt=_parse_numeric(metrics.total_debt.value),
            cash_and_equivalents=_parse_numeric(metrics.cash_and_equivalents.value),
            net_debt=_parse_numeric(metrics.net_debt.value),
            revenue=_parse_numeric(metrics.revenue.value),
            ebitda=_parse_numeric(metrics.ebitda.value),
        )

        result = self.validator.validate(fm)

        # Map validation messages back to specific field names
        field_failures: Dict[str, List[str]] = {}

        for msg in result.get("errors", []):
            # e.g. "revenue: value 0 outside valid range [0, 1000000000000]"
            field = msg.split(":")[0].strip()
            field_failures.setdefault(field, []).append(f"VALIDATION ERROR: {msg}")

        for msg in result.get("warnings", []):
            msg_lower = msg.lower()
            if "balance sheet" in msg_lower:
                for f in ("total_assets", "total_liabilities", "total_equity"):
                    field_failures.setdefault(f, []).append(f"VALIDATION WARNING: {msg}")
            elif "net debt" in msg_lower:
                for f in ("net_debt", "total_debt", "cash_and_equivalents"):
                    field_failures.setdefault(f, []).append(f"VALIDATION WARNING: {msg}")
            elif "ebitda margin" in msg_lower:
                for f in ("ebitda", "revenue"):
                    field_failures.setdefault(f, []).append(f"VALIDATION WARNING: {msg}")
            else:
                field_failures.setdefault("_general", []).append(
                    f"VALIDATION WARNING: {msg}"
                )

        if field_failures:
            logger.warning(
                f"[{_now()}] │  Validation issues on: "
                + ", ".join(k for k in field_failures if k != "_general")
            )
        else:
            logger.info(f"[{_now()}] │  All validation checks passed ✓")

        return field_failures

    # ── Steps 5 & 6 — Route fields ────────────────────────────────────────────

    def _route_and_write(
        self,
        metrics: FundMetrics,
        validation_failures: Dict[str, List[str]],
        filename: str,
    ) -> int:
        """
        Route each field:
          • LOW confidence OR validation failure → review/review_queue.csv
          • HIGH / MEDIUM + no failures          → output/{stem}.json

        Returns
        -------
        int
            Number of fields sent to the review queue.
        """
        logger.info(f"[{_now()}] │  Steps 5 & 6 – Routing fields to output / review…")

        clean: Dict[str, Any] = {}
        flagged_count = 0

        for field_name in metrics.model_fields:
            fe: FieldExtraction = getattr(metrics, field_name)

            flag_reasons: List[str] = []

            # Rule: LOW confidence always flags
            if fe.confidence == "LOW":
                flag_reasons.append(f"Low confidence: {fe.reason}")

            # Rule: validation failures flag regardless of confidence
            for val_msg in validation_failures.get(field_name, []):
                flag_reasons.append(val_msg)

            if flag_reasons:
                self._append_review_row(
                    filename=filename,
                    field_name=field_name,
                    extracted_value=fe.value,
                    confidence=fe.confidence,
                    reason="; ".join(flag_reasons),
                )
                flagged_count += 1
                logger.warning(
                    f"[{_now()}] │  ⚑  Flagged  '{field_name}' "
                    f"[{fe.confidence}] → review_queue.csv"
                )
            else:
                clean[field_name] = {
                    "value": fe.value,
                    "confidence": fe.confidence,
                    "reason": fe.reason,
                }
                logger.info(
                    f"[{_now()}] │  ✓  Accepted '{field_name}' "
                    f"[{fe.confidence}] = {fe.value!r}"
                )

        # Write clean fields to output/{stem}.json
        stem = Path(filename).stem
        output_path = self.output_dir / f"{stem}.json"
        payload = {
            "source_file": filename,
            "extraction_model": MODEL,
            "extraction_timestamp": datetime.now().isoformat(),
            "fields": clean,
        }
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        logger.info(
            f"[{_now()}] │  Wrote {len(clean)} accepted field(s)  → {output_path}"
        )
        logger.info(
            f"[{_now()}] │  Flagged {flagged_count} field(s)       → {self.review_csv}"
        )
        return flagged_count

    # ── Orchestrator ───────────────────────────────────────────────────────────

    def _process_file(self, pdf_path: Path) -> int:
        """Full pipeline for one PDF. Returns number of flagged fields."""
        text = self._ingest_pdf(pdf_path)                       # Step 1
        metrics = self._extract_with_claude(text, pdf_path.name)  # Steps 2 & 3
        failures = self._run_validation(metrics)                  # Step 4
        flagged = self._route_and_write(metrics, failures, pdf_path.name)  # Steps 5 & 6
        return flagged

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _init_review_csv(self) -> None:
        """Create review_queue.csv with column headers if it does not exist."""
        if not self.review_csv.exists():
            with open(self.review_csv, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=_REVIEW_HEADERS).writeheader()
            logger.info(f"[{_now()}] Created review queue: {self.review_csv}")

    def _append_review_row(
        self,
        *,
        filename: str,
        field_name: str,
        extracted_value: Optional[str],
        confidence: str,
        reason: str,
    ) -> None:
        """Append one flagged field as a row in review_queue.csv."""
        with open(self.review_csv, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=_REVIEW_HEADERS).writerow(
                {
                    "filename": filename,
                    "field_name": field_name,
                    "extracted_value": extracted_value or "",
                    "confidence": confidence,
                    "reason_for_flag": reason,
                }
            )

    def _append_run_log(self, summary: Dict[str, Any]) -> None:
        """Append a run summary entry to output/run_log.jsonl."""
        log_path = self.output_dir / "run_log.jsonl"
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary) + "\n")


# ── Timestamp helper ───────────────────────────────────────────────────────────


def _now() -> str:
    """Return current timestamp string for log messages."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
