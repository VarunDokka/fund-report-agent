"""
extractor.py
============
PDF text/table extraction, LLM prompt building, and the main
FinancialDataExtractor orchestrator.

All DataValidator logic lives in validator.py.

Author: Arnab Banerjee
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import pdfplumber

from .validator import DataValidator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FinancialMetrics:
    """Standardised container for extracted financial metrics."""

    company_name: str
    report_date: str
    report_type: str

    # Income Statement
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    ebitda: Optional[float] = None
    net_income: Optional[float] = None

    # Balance Sheet
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    total_debt: Optional[float] = None
    net_debt: Optional[float] = None

    # Key Ratios
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None

    # Metadata
    extraction_confidence: float = 0.0
    source_file: str = ""
    extraction_timestamp: str = ""


# ---------------------------------------------------------------------------
# LLM prompt construction
# ---------------------------------------------------------------------------

class LLMPromptBuilder:
    """
    Builds structured prompts for LLM-based financial data extraction.
    Compatible with OpenAI GPT-4, Anthropic Claude, or similar models.
    """

    SYSTEM_PROMPT = """You are a financial data extraction specialist. Your task is to
extract specific financial metrics from unstructured text extracted from fund reports
and financial statements.

Rules:
1. Extract only explicitly stated values - do not calculate or infer
2. Maintain original currency and units
3. Convert values to millions (M) for consistency
4. Return null for metrics not found in the text
5. Include confidence score (0-1) for each extraction
"""

    @staticmethod
    def build_extraction_prompt(text: str, target_metrics: List[str]) -> str:
        """
        Build a structured prompt for financial metric extraction.

        Args:
            text: Raw text extracted from PDF
            target_metrics: List of metrics to extract

        Returns:
            Formatted prompt string for LLM
        """
        metrics_list = "\n".join([f"- {m}" for m in target_metrics])

        prompt = f"""
### TASK: Extract Financial Metrics

### TARGET METRICS:
{metrics_list}

### SOURCE TEXT:
\"\"\"
{text[:8000]}  # Truncate to fit context window
\"\"\"

### INSTRUCTIONS:
1. Identify each metric from the source text
2. Extract the numerical value with its unit/currency
3. Note the fiscal period if mentioned
4. Provide confidence score (0-1) based on clarity of the data

### OUTPUT FORMAT (JSON):
{{
    "metrics": {{
        "metric_name": {{
            "value": <number or null>,
            "unit": "<currency/unit>",
            "period": "<fiscal period>",
            "confidence": <0-1>,
            "source_context": "<relevant text snippet>"
        }}
    }},
    "tables_identified": <number>,
    "overall_confidence": <0-1>
}}
"""
        return prompt

    @staticmethod
    def build_table_identification_prompt(text: str) -> str:
        """Build prompt to identify and classify financial tables."""
        return f"""
### TASK: Identify Financial Tables

Analyze the following text and identify what type of financial statements/tables are present.

### SOURCE TEXT:
\"\"\"
{text[:6000]}
\"\"\"

### IDENTIFY:
1. Balance Sheet - Assets, Liabilities, Equity
2. Income Statement - Revenue, Expenses, Net Income
3. Cash Flow Statement - Operating, Investing, Financing
4. Portfolio Holdings - Investment positions, valuations
5. Fund Performance - NAV, Returns, Distributions

### OUTPUT FORMAT (JSON):
{{
    "tables_found": [
        {{
            "type": "<table type>",
            "confidence": <0-1>,
            "key_items": ["<item1>", "<item2>"]
        }}
    ]
}}
"""


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------

class PDFTextExtractor:
    """Handles PDF text and table extraction with quality checks."""

    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)
        self.text_content: Dict[int, str] = {}
        self.tables: Dict[int, List] = {}
        self.metadata: Dict = {}

    def extract_all(self) -> Tuple[str, List[pd.DataFrame]]:
        """
        Extract all text and tables from PDF.

        Returns:
            Tuple of (full text, list of dataframes)
        """
        logger.info(f"Extracting content from: {self.pdf_path}")

        full_text = []
        all_tables = []

        with pdfplumber.open(self.pdf_path) as pdf:
            self.metadata = {
                "pages": len(pdf.pages),
                "filename": self.pdf_path.name,
            }

            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                self.text_content[i] = page_text
                full_text.append(f"\n--- Page {i+1} ---\n{page_text}")

                tables = page.extract_tables()
                if tables:
                    self.tables[i] = tables
                    for table in tables:
                        if table and len(table) > 1:
                            try:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                df["source_page"] = i + 1
                                all_tables.append(df)
                            except Exception as e:
                                logger.warning(f"Table parsing error on page {i+1}: {e}")

        logger.info(f"Extracted {len(full_text)} pages, {len(all_tables)} tables")
        return "\n".join(full_text), all_tables

    def extract_financial_tables(self) -> List[pd.DataFrame]:
        """
        Extract and identify financial statement tables specifically.
        Uses pattern matching to identify Balance Sheet, Income Statement, etc.
        """
        _, tables = self.extract_all()

        financial_patterns = {
            "balance_sheet": r"(assets|liabilities|equity|balance\s*sheet)",
            "income_statement": r"(revenue|income|expense|profit|loss)",
            "cash_flow": r"(cash\s*flow|operating|investing|financing)",
            "holdings": r"(portfolio|holdings|investments|positions)",
        }

        classified_tables = []
        for df in tables:
            header_text = " ".join(str(col).lower() for col in df.columns)
            first_col_text = " ".join(
                str(val).lower() for val in df.iloc[:, 0] if pd.notna(val)
            )
            combined_text = header_text + " " + first_col_text

            for table_type, pattern in financial_patterns.items():
                if re.search(pattern, combined_text, re.IGNORECASE):
                    df["table_type"] = table_type
                    classified_tables.append(df)
                    break

        return classified_tables


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class FinancialDataExtractor:
    """
    Main orchestrator: PDF extraction → LLM parsing → validation → export.
    """

    TARGET_METRICS = [
        "Revenue", "Gross Profit", "Operating Income", "EBITDA", "Net Income",
        "Total Assets", "Total Liabilities", "Total Equity",
        "Cash and Equivalents", "Total Debt", "Net Debt",
        "Debt to Equity Ratio", "Current Ratio",
    ]

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.prompt_builder = LLMPromptBuilder()
        self.validator = DataValidator()

    def process_pdf(self, pdf_path: str, company_name: str = "Unknown") -> FinancialMetrics:
        """
        Process a single PDF and return extracted, validated FinancialMetrics.

        Args:
            pdf_path: Path to the PDF file
            company_name: Name of the company/fund

        Returns:
            Populated FinancialMetrics dataclass
        """
        logger.info(f"Processing: {pdf_path}")

        # Step 1 – Extract text and tables
        extractor = PDFTextExtractor(pdf_path)
        full_text, tables = extractor.extract_all()

        # Step 2 – Build LLM prompt
        extraction_prompt = self.prompt_builder.build_extraction_prompt(
            full_text, self.TARGET_METRICS
        )

        # Step 3 – Call LLM (placeholder; integrate actual API in production)
        llm_response = self._call_llm(extraction_prompt)

        # Step 4 – Parse response into structured metrics
        metrics = self._parse_llm_response(llm_response, company_name, pdf_path)

        # Step 5 – Supplement with direct table extraction
        metrics = self._supplement_from_tables(metrics, tables)

        # Step 6 – Validate
        validation_results = self.validator.validate(metrics)
        for warning in validation_results.get("warnings", []):
            logger.warning(f"Validation: {warning}")

        metrics.extraction_confidence = (
            validation_results["checks_passed"]
            / max(validation_results["checks_total"], 1)
        )

        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> Dict:
        """
        Call LLM API for extraction.

        In production, replace the placeholder below with one of:
          • OpenAI:    openai.ChatCompletion.create(model="gpt-4", ...)
          • Anthropic: anthropic.Anthropic().messages.create(...)
          • Azure:     Similar to OpenAI

        Returns:
            Parsed JSON response dict from the LLM
        """
        logger.info("LLM extraction (placeholder – integrate actual API)")
        return {"metrics": {}, "overall_confidence": 0.0}

    def _parse_llm_response(
        self, response: Dict, company_name: str, source_file: str
    ) -> FinancialMetrics:
        """Parse LLM JSON response into a FinancialMetrics dataclass."""
        metrics_data = response.get("metrics", {})

        def get_value(key: str) -> Optional[float]:
            if key in metrics_data and metrics_data[key].get("value") is not None:
                return float(metrics_data[key]["value"])
            return None

        return FinancialMetrics(
            company_name=company_name,
            report_date=datetime.now().strftime("%Y-%m-%d"),
            report_type="Fund Report",
            revenue=get_value("revenue"),
            gross_profit=get_value("gross_profit"),
            operating_income=get_value("operating_income"),
            ebitda=get_value("ebitda"),
            net_income=get_value("net_income"),
            total_assets=get_value("total_assets"),
            total_liabilities=get_value("total_liabilities"),
            total_equity=get_value("total_equity"),
            cash_and_equivalents=get_value("cash_and_equivalents"),
            total_debt=get_value("total_debt"),
            net_debt=get_value("net_debt"),
            debt_to_equity=get_value("debt_to_equity"),
            current_ratio=get_value("current_ratio"),
            extraction_confidence=response.get("overall_confidence", 0.0),
            source_file=str(source_file),
            extraction_timestamp=datetime.now().isoformat(),
        )

    def _supplement_from_tables(
        self, metrics: FinancialMetrics, tables: List[pd.DataFrame]
    ) -> FinancialMetrics:
        """
        Fill any None fields by direct regex scanning of extracted tables.
        Acts as a fallback when the LLM did not return a value.
        """
        patterns = {
            "revenue": r"^(total\s*)?revenue|net\s*sales|total\s*sales",
            "ebitda": r"ebitda|operating\s*income\s*before",
            "net_income": r"^net\s*(income|profit|loss)",
            "total_assets": r"^total\s*assets",
            "total_debt": r"^total\s*(debt|borrowings)",
        }

        for df in tables:
            if df.empty:
                continue

            line_item_col = df.columns[0]
            value_cols = [
                c for c in df.columns if c not in (line_item_col, "source_page")
            ]
            if not value_cols:
                continue

            for _, row in df.iterrows():
                line_item = str(row[line_item_col]).lower().strip()
                for field, pattern in patterns.items():
                    if re.match(pattern, line_item, re.IGNORECASE):
                        if getattr(metrics, field) is None:
                            for col in value_cols:
                                try:
                                    raw = str(row[col]).replace(",", "").replace("$", "")
                                    value = float(re.sub(r"[^\d.-]", "", raw))
                                    setattr(metrics, field, value)
                                    logger.info(f"Table fallback – {field}: {value}")
                                    break
                                except (ValueError, TypeError):
                                    continue

        return metrics

    # ------------------------------------------------------------------
    # Batch processing & export
    # ------------------------------------------------------------------

    def batch_process(self, pdf_dir: str) -> pd.DataFrame:
        """
        Process all PDFs in a directory and return a summary DataFrame.

        Args:
            pdf_dir: Directory containing PDF files

        Returns:
            DataFrame with one row per processed document
        """
        pdf_files = list(Path(pdf_dir).glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")

        all_metrics = []
        for pdf_path in pdf_files:
            try:
                metrics = self.process_pdf(str(pdf_path), pdf_path.stem)
                all_metrics.append(asdict(metrics))
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")

        return pd.DataFrame(all_metrics)

    def export_to_csv(self, metrics: List[FinancialMetrics], filename: str) -> Path:
        """Export extracted metrics to CSV."""
        df = pd.DataFrame([asdict(m) for m in metrics])
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Exported CSV: {output_path}")
        return output_path

    def export_to_excel(self, metrics: List[FinancialMetrics], filename: str) -> Path:
        """Export extracted metrics to Excel with a validation summary sheet."""
        df = pd.DataFrame([asdict(m) for m in metrics])
        output_path = self.output_dir / filename

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Extracted Metrics", index=False)
            df[["company_name", "extraction_confidence", "source_file"]].to_excel(
                writer, sheet_name="Validation Summary", index=False
            )

        logger.info(f"Exported Excel: {output_path}")
        return output_path
