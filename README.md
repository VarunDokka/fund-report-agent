# fund-report-agent

Agentic pipeline that extracts structured data from private fund PDFs with confidence scoring and human review queue.

---

## Architecture

```
                        ┌─────────────────────────────────────┐
                        │           Agent Loop                │
                        │                                     │
  PDF Report  ─────────▶│  1. Ingest PDF (pdfplumber)         │
                        │  2. Call Claude API (tool use)      │
                        │  3. Parse into 17 structured fields │
                        │  4. Run validation checks           │
                        │                                     │
                        └──────────────┬──────────────────────┘
                                       │
                          Per-field confidence routing
                                       │
               ┌───────────────────────┴───────────────────────┐
               │                                               │
               ▼                                               ▼
     HIGH / MEDIUM confidence                        LOW confidence
     + validation passed                             or validation failed
               │                                               │
               ▼                                               ▼
    output/{filename}.json                     review/review_queue.csv
                                                               │
                                                               ▼
                                                  python run_agent.py --review
                                                  python run_agent.py --approve
                                                               │
                                                               ▼
                                                       Human Review [Y/N]
```

Retries are capped at 2 per document. A JSON parse failure re-prompts Claude to return clean JSON. A balance-sheet validation failure re-prompts Claude with its prior extraction in context. After two failed retries all fields are flagged LOW and sent to the review queue.

---

## Quick Start

```bash
git clone https://github.com/VarunDokka/fund-report-agent.git
cd fund-report-agent

pip install -r requirements.txt

cp .env.example .env          # add your ANTHROPIC_API_KEY
python run_agent.py --input reports/
```

Place PDF reports in `reports/`. Extracted data lands in `output/` and flagged fields in `review/review_queue.csv`.

---

## CLI Reference

```bash
# Process a single file or an entire directory
python run_agent.py --input reports/q3_2024.pdf
python run_agent.py --input reports/

# Print a formatted table of all flagged fields
python run_agent.py --review

# Interactively approve or reject flagged fields (Y/N/S/Q)
python run_agent.py --approve review/review_queue.csv
```

---

## Example Output

`output/q3_2024.json`

```json
{
  "source_file": "q3_2024.pdf",
  "extracted_at": "2024-10-14T09:23:41",
  "fields": {
    "nav": {
      "value": "487500000",
      "confidence": "HIGH",
      "reason": "Stated directly on page 3 as $487.5M NAV as of September 30, 2024"
    },
    "irr": {
      "value": "18.4",
      "confidence": "HIGH",
      "reason": "Net IRR since inception reported in the performance summary table"
    },
    "moic": {
      "value": "2.1",
      "confidence": "MEDIUM",
      "reason": "MOIC calculated from distributed and unrealised value figures"
    },
    "capital_called": {
      "value": "312000000",
      "confidence": "HIGH",
      "reason": "Cumulative capital called stated as $312M in the capital account summary"
    },
    "ebitda": {
      "value": "74200000",
      "confidence": "MEDIUM",
      "reason": "Aggregate EBITDA inferred by summing portfolio company contributions"
    },
    "net_debt": {
      "value": null,
      "confidence": "LOW",
      "reason": "Debt and cash figures appear on different pages with conflicting dates"
    }
  },
  "validation": {
    "balance_sheet_ok": true,
    "net_debt_ok": false,
    "ebitda_margin_ok": true
  }
}
```

---

## Example Review Queue

`review/review_queue.csv`

```
filename,field_name,extracted_value,confidence,reason_for_flag,human_reviewed,reviewer_notes
q3_2024.pdf,net_debt,,LOW,Debt and cash figures appear on different pages with conflicting dates,,
q3_2024.pdf,distributions,15200000,LOW,Distribution figure mentioned in two places with a $200k discrepancy,,
q2_2024.pdf,total_equity,94300000,MEDIUM,Balance sheet equation fails: assets != liabilities + equity (delta 1.4%),,
q2_2024.pdf,ebitda,,LOW,No EBITDA or operating income line found in document,,
```

Running `--approve` steps through each unreviewed row. Approved rows are written back with `human_reviewed = TRUE` and an optional reviewer note.

---

## Extracted Fields

| Category | Fields |
|---|---|
| Fund-level | NAV, capital called, distributions, IRR, MOIC, gross returns, net returns, portfolio company count, top holdings |
| Balance sheet | Total assets, total liabilities, total equity, total debt, cash and equivalents, net debt |
| Income statement | Revenue, EBITDA |

---

## Validation Checks

The `DataValidator` runs four checks after every extraction:

- **Balance sheet equation** — `total_assets = total_liabilities + total_equity` within 1% tolerance
- **Net debt consistency** — `net_debt = total_debt - cash_and_equivalents` within 2% tolerance
- **EBITDA margin reasonability** — flags margins outside the range -50% to +80%
- **Range checks** — field-level min/max guards against implausible values

Failing fields are routed to the review queue regardless of Claude's confidence rating.

---

## Project Structure

```
fund-report-agent/
├── src/
│   ├── agent.py          # Agentic extraction loop, retry logic, routing
│   ├── extractor.py      # PDF ingestion, prompt builder, FinancialMetrics dataclass
│   ├── validator.py      # DataValidator with four automated checks
│   └── review_queue.py   # Review queue persistence and approval workflow
├── reports/              # Drop input PDFs here
├── output/               # Per-document JSON (git-ignored)
├── review/               # review_queue.csv and pending/approved/rejected dirs
├── run_agent.py          # CLI entry point (--input, --review, --approve)
├── requirements.txt
└── .env.example
```

---

## Technical Stack

| Component | Library / Service |
|---|---|
| LLM extraction | Anthropic Claude API (`claude-sonnet-4-6`), streaming + forced tool use |
| PDF ingestion | pdfplumber |
| Structured output | Pydantic v2 |
| Data manipulation | pandas |
| Environment config | python-dotenv |
| Runtime | Python 3.8+ |

---

## Configuration

Copy `.env.example` to `.env` and set your API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

No other configuration is required. The agent reads PDFs from `reports/`, writes JSON to `output/`, and appends flagged fields to `review/review_queue.csv` automatically.
