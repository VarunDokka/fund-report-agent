"""
Financial Data Extraction Tool (AI & NLP)
==========================================
Automated pipeline to extract structured financial data from unstructured PDF reports
using OCR and Large Language Models (LLMs).

Author: Arnab Banerjee
GitHub: github.com/arnab2308

Use Case: Private capital fund reports often contain financial statements in unstructured
PDF formats. This tool automates extraction of key metrics like EBITDA, Net Debt, Revenue
into clean, analysis-ready formats (CSV/Excel).
"""

import os
import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
import pdfplumber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FinancialMetrics:
    """Data class for standardized financial metrics extraction."""
    company_name: str
    report_date: str
    report_type: str
    
    # Income Statement Metrics
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    ebitda: Optional[float] = None
    net_income: Optional[float] = None
    
    # Balance Sheet Metrics
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


class LLMPromptBuilder:
    """
    Builds structured prompts for LLM-based financial data extraction.
    Designed to work with OpenAI GPT-4, Anthropic Claude, or similar models.
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
                "filename": self.pdf_path.name
            }
            
            for i, page in enumerate(pdf.pages):
                # Extract text
                page_text = page.extract_text() or ""
                self.text_content[i] = page_text
                full_text.append(f"\n--- Page {i+1} ---\n{page_text}")
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    self.tables[i] = tables
                    for table in tables:
                        if table and len(table) > 1:
                            try:
                                # First row as header
                                df = pd.DataFrame(table[1:], columns=table[0])
                                df['source_page'] = i + 1
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
            'balance_sheet': r'(assets|liabilities|equity|balance\s*sheet)',
            'income_statement': r'(revenue|income|expense|profit|loss)',
            'cash_flow': r'(cash\s*flow|operating|investing|financing)',
            'holdings': r'(portfolio|holdings|investments|positions)'
        }
        
        classified_tables = []
        for df in tables:
            # Check header and first column for financial terms
            header_text = ' '.join(str(col).lower() for col in df.columns)
            first_col_text = ' '.join(str(val).lower() for val in df.iloc[:, 0] if pd.notna(val))
            combined_text = header_text + ' ' + first_col_text
            
            for table_type, pattern in financial_patterns.items():
                if re.search(pattern, combined_text, re.IGNORECASE):
                    df['table_type'] = table_type
                    classified_tables.append(df)
                    break
        
        return classified_tables


class DataValidator:
    """
    Validates extracted financial data for quality and consistency.
    Implements automated checks against historical data and logical rules.
    """
    
    VALIDATION_RULES = {
        'revenue': {'min': 0, 'max': 1e12},  # Revenue should be positive
        'ebitda': {'min': -1e11, 'max': 1e11},  # EBITDA can be negative
        'net_debt': {'min': -1e11, 'max': 1e11},
        'total_assets': {'min': 0, 'max': 1e13},
        'debt_to_equity': {'min': -10, 'max': 50},
    }
    
    def __init__(self, historical_data: Optional[pd.DataFrame] = None):
        """
        Initialize validator with optional historical data for comparison.
        
        Args:
            historical_data: DataFrame with historical metrics for comparison
        """
        self.historical_data = historical_data
        self.validation_results: List[Dict] = []
        
    def validate(self, metrics: FinancialMetrics) -> Dict[str, Any]:
        """
        Run all validation checks on extracted metrics.
        
        Args:
            metrics: Extracted financial metrics
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'checks_passed': 0,
            'checks_total': 0
        }
        
        # Range validation
        metrics_dict = asdict(metrics)
        for field, rules in self.VALIDATION_RULES.items():
            value = metrics_dict.get(field)
            if value is not None:
                results['checks_total'] += 1
                if rules['min'] <= value <= rules['max']:
                    results['checks_passed'] += 1
                else:
                    results['errors'].append(
                        f"{field}: Value {value} outside valid range [{rules['min']}, {rules['max']}]"
                    )
                    results['is_valid'] = False
        
        # Logical consistency checks
        results['checks_total'] += 3
        
        # Check: Total Assets = Total Liabilities + Total Equity
        if all(metrics_dict.get(f) is not None for f in ['total_assets', 'total_liabilities', 'total_equity']):
            balance = metrics.total_assets - (metrics.total_liabilities + metrics.total_equity)
            if abs(balance) > 0.01 * metrics.total_assets:  # 1% tolerance
                results['warnings'].append(
                    f"Balance sheet imbalance: Assets - (Liabilities + Equity) = {balance:,.0f}"
                )
            else:
                results['checks_passed'] += 1
        
        # Check: Net Debt = Total Debt - Cash
        if all(metrics_dict.get(f) is not None for f in ['net_debt', 'total_debt', 'cash_and_equivalents']):
            expected_net_debt = metrics.total_debt - metrics.cash_and_equivalents
            if abs(metrics.net_debt - expected_net_debt) > 0.01 * abs(expected_net_debt):
                results['warnings'].append(
                    f"Net debt inconsistency: Expected {expected_net_debt:,.0f}, got {metrics.net_debt:,.0f}"
                )
            else:
                results['checks_passed'] += 1
        
        # Check: EBITDA margin reasonable (if revenue present)
        if metrics.ebitda is not None and metrics.revenue is not None and metrics.revenue > 0:
            ebitda_margin = metrics.ebitda / metrics.revenue
            if not -0.5 <= ebitda_margin <= 0.8:
                results['warnings'].append(
                    f"Unusual EBITDA margin: {ebitda_margin:.1%}"
                )
            else:
                results['checks_passed'] += 1
        
        # Historical comparison (if available)
        if self.historical_data is not None:
            self._compare_to_historical(metrics, results)
        
        return results
    
    def _compare_to_historical(self, metrics: FinancialMetrics, results: Dict):
        """Compare extracted metrics to historical data for anomaly detection."""
        try:
            historical = self.historical_data[
                self.historical_data['company_name'] == metrics.company_name
            ]
            
            if len(historical) > 0:
                for field in ['revenue', 'ebitda', 'total_assets']:
                    current = getattr(metrics, field)
                    if current is not None and field in historical.columns:
                        hist_mean = historical[field].mean()
                        hist_std = historical[field].std()
                        
                        if hist_std > 0:
                            z_score = (current - hist_mean) / hist_std
                            if abs(z_score) > 3:
                                results['warnings'].append(
                                    f"{field}: Significant deviation from historical (z={z_score:.1f})"
                                )
        except Exception as e:
            logger.warning(f"Historical comparison failed: {e}")


class FinancialDataExtractor:
    """
    Main orchestrator class for the financial data extraction pipeline.
    Combines PDF extraction, LLM parsing, and data validation.
    """
    
    TARGET_METRICS = [
        'Revenue', 'Gross Profit', 'Operating Income', 'EBITDA', 'Net Income',
        'Total Assets', 'Total Liabilities', 'Total Equity', 
        'Cash and Equivalents', 'Total Debt', 'Net Debt',
        'Debt to Equity Ratio', 'Current Ratio'
    ]
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.prompt_builder = LLMPromptBuilder()
        self.validator = DataValidator()
        
    def process_pdf(self, pdf_path: str, company_name: str = "Unknown") -> FinancialMetrics:
        """
        Process a single PDF and extract financial metrics.
        
        Args:
            pdf_path: Path to the PDF file
            company_name: Name of the company/fund
            
        Returns:
            Extracted and validated FinancialMetrics
        """
        logger.info(f"Processing: {pdf_path}")
        
        # Step 1: Extract text and tables from PDF
        extractor = PDFTextExtractor(pdf_path)
        full_text, tables = extractor.extract_all()
        
        # Step 2: Build LLM prompt
        extraction_prompt = self.prompt_builder.build_extraction_prompt(
            full_text, self.TARGET_METRICS
        )
        
        # Step 3: Call LLM for extraction (placeholder - integrate actual API)
        # In production, this would call OpenAI/Anthropic API
        llm_response = self._call_llm(extraction_prompt)
        
        # Step 4: Parse LLM response into structured metrics
        metrics = self._parse_llm_response(llm_response, company_name, pdf_path)
        
        # Step 5: Supplement with direct table extraction
        metrics = self._supplement_from_tables(metrics, tables)
        
        # Step 6: Validate extracted data
        validation_results = self.validator.validate(metrics)
        if validation_results['warnings']:
            for warning in validation_results['warnings']:
                logger.warning(f"Validation: {warning}")
        
        metrics.extraction_confidence = validation_results['checks_passed'] / max(validation_results['checks_total'], 1)
        
        return metrics
    
    def _call_llm(self, prompt: str) -> Dict:
        """
        Call LLM API for extraction.
        
        In production, integrate with:
        - OpenAI: openai.ChatCompletion.create()
        - Anthropic: anthropic.Anthropic().messages.create()
        - Azure OpenAI: Similar to OpenAI
        
        Returns:
            Parsed JSON response from LLM
        """
        # Placeholder: Return structure for demonstration
        # In production, this would make actual API calls
        logger.info("LLM extraction (placeholder - integrate actual API)")
        
        # Example of what integration would look like:
        """
        # OpenAI Integration
        import openai
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": LLMPromptBuilder.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,  # Low temperature for factual extraction
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
        
        # Anthropic Claude Integration
        import anthropic
        
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )
        return json.loads(response.content[0].text)
        """
        
        return {"metrics": {}, "overall_confidence": 0.0}
    
    def _parse_llm_response(self, response: Dict, company_name: str, 
                           source_file: str) -> FinancialMetrics:
        """Parse LLM JSON response into FinancialMetrics dataclass."""
        metrics_data = response.get('metrics', {})
        
        def get_value(key: str) -> Optional[float]:
            if key in metrics_data and metrics_data[key].get('value') is not None:
                return float(metrics_data[key]['value'])
            return None
        
        return FinancialMetrics(
            company_name=company_name,
            report_date=datetime.now().strftime("%Y-%m-%d"),
            report_type="Fund Report",
            revenue=get_value('revenue'),
            gross_profit=get_value('gross_profit'),
            operating_income=get_value('operating_income'),
            ebitda=get_value('ebitda'),
            net_income=get_value('net_income'),
            total_assets=get_value('total_assets'),
            total_liabilities=get_value('total_liabilities'),
            total_equity=get_value('total_equity'),
            cash_and_equivalents=get_value('cash_and_equivalents'),
            total_debt=get_value('total_debt'),
            net_debt=get_value('net_debt'),
            debt_to_equity=get_value('debt_to_equity'),
            current_ratio=get_value('current_ratio'),
            extraction_confidence=response.get('overall_confidence', 0.0),
            source_file=str(source_file),
            extraction_timestamp=datetime.now().isoformat()
        )
    
    def _supplement_from_tables(self, metrics: FinancialMetrics, 
                                tables: List[pd.DataFrame]) -> FinancialMetrics:
        """
        Supplement LLM extraction with direct table parsing.
        Uses regex patterns to identify financial line items.
        """
        patterns = {
            'revenue': r'^(total\s*)?revenue|net\s*sales|total\s*sales',
            'ebitda': r'ebitda|operating\s*income\s*before',
            'net_income': r'^net\s*(income|profit|loss)',
            'total_assets': r'^total\s*assets',
            'total_debt': r'^total\s*(debt|borrowings)',
        }
        
        for df in tables:
            if df.empty:
                continue
                
            # Assume first column is line item description
            line_item_col = df.columns[0]
            value_cols = [c for c in df.columns if c != line_item_col and c != 'source_page']
            
            if not value_cols:
                continue
                
            for _, row in df.iterrows():
                line_item = str(row[line_item_col]).lower().strip()
                
                for field, pattern in patterns.items():
                    if re.match(pattern, line_item, re.IGNORECASE):
                        current_value = getattr(metrics, field)
                        if current_value is None:
                            # Try to extract numeric value
                            for col in value_cols:
                                try:
                                    value_str = str(row[col]).replace(',', '').replace('$', '')
                                    value = float(re.sub(r'[^\d.-]', '', value_str))
                                    setattr(metrics, field, value)
                                    logger.info(f"Extracted {field}: {value} from table")
                                    break
                                except (ValueError, TypeError):
                                    continue
        
        return metrics
    
    def batch_process(self, pdf_dir: str) -> pd.DataFrame:
        """
        Process all PDFs in a directory.
        
        Args:
            pdf_dir: Directory containing PDF files
            
        Returns:
            DataFrame with all extracted metrics
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
        
        df = pd.DataFrame(all_metrics)
        return df
    
    def export_to_csv(self, metrics: List[FinancialMetrics], filename: str):
        """Export extracted metrics to CSV."""
        df = pd.DataFrame([asdict(m) for m in metrics])
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)
        logger.info(f"Exported to: {output_path}")
        return output_path
    
    def export_to_excel(self, metrics: List[FinancialMetrics], filename: str):
        """Export extracted metrics to Excel with formatting."""
        df = pd.DataFrame([asdict(m) for m in metrics])
        output_path = self.output_dir / filename
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Extracted Metrics', index=False)
            
            # Add validation summary sheet
            validation_df = df[['company_name', 'extraction_confidence', 'source_file']]
            validation_df.to_excel(writer, sheet_name='Validation Summary', index=False)
        
        logger.info(f"Exported to: {output_path}")
        return output_path


def main():
    """Main entry point for the extraction pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract financial data from PDF reports using AI/NLP"
    )
    parser.add_argument('input', help="PDF file or directory to process")
    parser.add_argument('--output', '-o', default='output', help="Output directory")
    parser.add_argument('--format', '-f', choices=['csv', 'excel', 'both'], 
                       default='both', help="Output format")
    parser.add_argument('--company', '-c', default='Unknown', help="Company name")
    
    args = parser.parse_args()
    
    extractor = FinancialDataExtractor(output_dir=args.output)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        metrics = extractor.process_pdf(str(input_path), args.company)
        metrics_list = [metrics]
    else:
        df = extractor.batch_process(str(input_path))
        metrics_list = [FinancialMetrics(**row) for _, row in df.iterrows()]
    
    # Export results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.format in ['csv', 'both']:
        extractor.export_to_csv(metrics_list, f"financial_data_{timestamp}.csv")
    if args.format in ['excel', 'both']:
        extractor.export_to_excel(metrics_list, f"financial_data_{timestamp}.xlsx")
    
    print(f"\nExtraction complete. Processed {len(metrics_list)} documents.")


if __name__ == "__main__":
    main()
