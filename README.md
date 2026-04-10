[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arnab2308/financial-data-extractor/blob/main/Financial_Data_Extractor_Colab.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Automated Financial Data Extraction Tool (AI & NLP)

An AI-powered Python pipeline for extracting structured financial data from unstructured PDF reports. Designed for private capital fund analysis, portfolio company monitoring, and financial data aggregation.

## 🎯 Problem Statement

Private market data often comes in **unstructured PDFs**—quarterly fund reports, portfolio company financials, and partnership statements. Manual extraction is:
- **Time-consuming**: Hours spent copying data into spreadsheets
- **Error-prone**: Human transcription errors compound over time  
- **Unscalable**: Cannot keep up with growing portfolio sizes

This tool automates the extraction of key financial metrics using **OCR and Large Language Models (LLMs)**, converting messy PDF reports into clean, analysis-ready data.

## ✨ Features

- **PDF Text & Table Extraction**: Uses `pdfplumber` to extract text and tabular data from fund reports
- **LLM-Powered Parsing**: Leverages GPT-4/Claude to identify and extract specific financial metrics from unstructured text
- **Structured Prompts**: Engineered prompts for accurate extraction of Balance Sheet, Income Statement, and Cash Flow items
- **Data Validation**: Automated quality checks (balance sheet equation, net debt calculation, historical comparison)
- **Multi-Format Export**: Output to CSV, Excel, or SQL-ready formats
- **Batch Processing**: Process entire folders of fund reports

## 📊 Extracted Metrics

| Category | Metrics |
|----------|---------|
| **Income Statement** | Revenue, Gross Profit, EBITDA, Operating Income, Net Income |
| **Balance Sheet** | Total Assets, Total Liabilities, Total Equity, Cash, Total Debt, Net Debt |
| **Credit Metrics** | Debt/Equity, Net Debt/EBITDA, Current Ratio, Interest Coverage |
| **Fund-Level** | NAV, Capital Called, Distributions, IRR |

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PDF Reports   │────▶│  Text/Table     │────▶│  LLM Extraction │
│  (Unstructured) │     │  Extraction     │     │  (GPT-4/Claude) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   CSV/Excel     │◀────│  Data           │◀────│  Structured     │
│   Output        │     │  Validation     │     │  Metrics        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/arnab2308/financial-data-extractor.git
cd financial-data-extractor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.financial_extractor import FinancialDataExtractor

# Initialize extractor
extractor = FinancialDataExtractor(output_dir="output")

# Process a single PDF
metrics = extractor.process_pdf(
    pdf_path="reports/fund_q3_2024.pdf",
    company_name="TechFlow Solutions"
)

# Export results
extractor.export_to_excel([metrics], "extracted_data.xlsx")
```

### Command Line

```bash
# Process single file
python -m src.financial_extractor reports/fund_report.pdf -o output/ -f excel

# Batch process directory
python -m src.financial_extractor reports/ -o output/ -f both
```

### Run Demo

```bash
python demo.py
```

## 🔧 Configuration

### LLM Integration

The tool supports multiple LLM providers. Configure your API keys:

```python
# OpenAI
import openai
openai.api_key = "your-api-key"

# Anthropic Claude
import anthropic
client = anthropic.Anthropic(api_key="your-api-key")
```

### Custom Metrics

Extend `TARGET_METRICS` in `FinancialDataExtractor` to capture additional fields:

```python
TARGET_METRICS = [
    'Revenue', 'EBITDA', 'Net Debt',
    'ARR',  # Add SaaS-specific metrics
    'LTV/CAC',
    'Net Revenue Retention'
]
```

## 📁 Project Structure

```
financial-data-extractor/
├── src/
│   ├── __init__.py
│   └── financial_extractor.py    # Main extraction logic
├── data/
│   └── sample_reports/           # Sample PDF reports
├── output/                       # Extracted data output
├── tests/
│   └── test_extractor.py        # Unit tests
├── demo.py                       # Demonstration script
├── requirements.txt
└── README.md
```

## 📋 Sample Output

### Extracted Portfolio Companies (CSV/Excel)

| company_name | industry | revenue | ebitda | ebitda_margin | net_debt | net_debt_to_ebitda |
|--------------|----------|---------|--------|---------------|----------|-------------------|
| TechFlow Solutions | Software/SaaS | 287,400,000 | 74,724,000 | 26.0% | 182,700,000 | 2.4x |
| MedSupply Holdings | Healthcare | 198,600,000 | 29,790,000 | 15.0% | 113,500,000 | 3.8x |

## ✅ Data Validation

The tool implements automated quality checks:

1. **Balance Sheet Equation**: `Assets = Liabilities + Equity` (±1% tolerance)
2. **Net Debt Calculation**: `Net Debt = Total Debt - Cash`
3. **EBITDA Margin Reasonability**: Flags margins outside -50% to +80%
4. **Historical Comparison**: Z-score analysis against prior periods (optional)

```python
# Validation output example
{
    'is_valid': True,
    'checks_passed': 5,
    'checks_total': 6,
    'warnings': ['EBITDA margin slightly below sector average']
}
```

## 🛠️ Technologies Used

- **Python 3.8+**
- **pdfplumber**: PDF text and table extraction
- **pandas**: Data manipulation and export
- **OpenAI/Anthropic API**: LLM-based extraction
- **openpyxl**: Excel file generation

## 📈 Use Cases

- **Private Credit Fund Reporting**: Extract portfolio company financials from quarterly reports
- **LP Transparency**: Aggregate underlying holdings for institutional asset owners
- **Due Diligence**: Quickly digitize data rooms during deal evaluation
- **Portfolio Monitoring**: Track financial covenant compliance across investments

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 👤 Author

**Arnab Banerjee**
- GitHub: [@arnab2308](https://github.com/arnab2308)
- LinkedIn: [arnabbanerjee2308](https://linkedin.com/in/arnabbanerjee2308)

---

*Built to solve real problems in private capital data management. Feedback and suggestions welcome!*
