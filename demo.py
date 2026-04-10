"""
Demo: Financial Data Extraction from Sample Fund Report
========================================================
This script demonstrates the extraction pipeline on a sample private credit fund report.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime
from src.financial_extractor import (
    FinancialDataExtractor, 
    FinancialMetrics,
    LLMPromptBuilder,
    DataValidator
)

# Simulated extracted text from a Private Credit Fund Report
SAMPLE_FUND_REPORT_TEXT = """
BLACKROCK PRIVATE CREDIT FUND II
QUARTERLY REPORT - Q3 2024

PORTFOLIO SUMMARY
=================
As of September 30, 2024

Net Asset Value (NAV): $2,847,500,000
Total Commitments: $3,500,000,000
Capital Called: 81.4%

UNDERLYING PORTFOLIO COMPANIES - FINANCIAL HIGHLIGHTS
=====================================================

Company: TechFlow Solutions LLC
Industry: Software/SaaS
Investment Type: Senior Secured Term Loan
Principal Amount: $125,000,000
Interest Rate: SOFR + 575 bps

Financial Metrics (LTM September 2024):
- Revenue: $287,400,000
- Gross Profit: $201,180,000 (70% margin)
- EBITDA: $74,724,000 (26% margin)
- Net Income: $31,614,000
- Total Assets: $412,500,000
- Total Liabilities: $298,750,000
- Total Equity: $113,750,000
- Cash and Equivalents: $42,300,000
- Total Debt: $225,000,000
- Net Debt: $182,700,000

Credit Metrics:
- Net Debt/EBITDA: 2.4x
- Interest Coverage: 4.2x
- Current Ratio: 1.8x

---

Company: MedSupply Holdings Inc.
Industry: Healthcare Distribution
Investment Type: Unitranche Facility
Principal Amount: $85,000,000
Interest Rate: SOFR + 625 bps

Financial Metrics (LTM September 2024):
- Revenue: $198,600,000
- Gross Profit: $49,650,000 (25% margin)
- EBITDA: $29,790,000 (15% margin)  
- Net Income: $12,100,000
- Total Assets: $278,400,000
- Total Liabilities: $189,200,000
- Total Equity: $89,200,000
- Cash and Equivalents: $28,500,000
- Total Debt: $142,000,000
- Net Debt: $113,500,000

Credit Metrics:
- Net Debt/EBITDA: 3.8x
- Interest Coverage: 2.9x
- Current Ratio: 1.4x
"""

def demonstrate_extraction_pipeline():
    """Demonstrate the full extraction and validation pipeline."""
    
    print("=" * 70)
    print("FINANCIAL DATA EXTRACTION TOOL - DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Step 1: Show prompt building for LLM
    print("STEP 1: Building LLM Extraction Prompt")
    print("-" * 40)
    
    prompt_builder = LLMPromptBuilder()
    target_metrics = [
        'Revenue', 'EBITDA', 'Net Debt', 'Total Assets', 
        'Total Liabilities', 'Total Equity'
    ]
    
    extraction_prompt = prompt_builder.build_extraction_prompt(
        SAMPLE_FUND_REPORT_TEXT, 
        target_metrics
    )
    
    print("Prompt Preview (first 500 chars):")
    print(extraction_prompt[:500] + "...")
    print()
    
    # Step 2: Simulate LLM extraction response
    print("STEP 2: LLM Extraction Results (Simulated)")
    print("-" * 40)
    
    # In production, this would come from actual LLM API call
    extracted_companies = [
        {
            'company_name': 'TechFlow Solutions LLC',
            'industry': 'Software/SaaS',
            'revenue': 287400000,
            'gross_profit': 201180000,
            'ebitda': 74724000,
            'net_income': 31614000,
            'total_assets': 412500000,
            'total_liabilities': 298750000,
            'total_equity': 113750000,
            'cash_and_equivalents': 42300000,
            'total_debt': 225000000,
            'net_debt': 182700000,
            'debt_to_equity': 1.98,
            'current_ratio': 1.8
        },
        {
            'company_name': 'MedSupply Holdings Inc.',
            'industry': 'Healthcare Distribution',
            'revenue': 198600000,
            'gross_profit': 49650000,
            'ebitda': 29790000,
            'net_income': 12100000,
            'total_assets': 278400000,
            'total_liabilities': 189200000,
            'total_equity': 89200000,
            'cash_and_equivalents': 28500000,
            'total_debt': 142000000,
            'net_debt': 113500000,
            'debt_to_equity': 2.12,
            'current_ratio': 1.4
        }
    ]
    
    for company in extracted_companies:
        print(f"\n{company['company_name']}:")
        print(f"  Revenue:       ${company['revenue']:>15,}")
        print(f"  EBITDA:        ${company['ebitda']:>15,}")
        print(f"  Net Debt:      ${company['net_debt']:>15,}")
        print(f"  EBITDA Margin: {company['ebitda']/company['revenue']*100:>14.1f}%")
    
    print()
    
    # Step 3: Validate extracted data
    print("STEP 3: Data Validation")
    print("-" * 40)
    
    validator = DataValidator()
    
    for company in extracted_companies:
        metrics = FinancialMetrics(
            company_name=company['company_name'],
            report_date='2024-09-30',
            report_type='Quarterly Fund Report',
            revenue=company['revenue'],
            gross_profit=company.get('gross_profit'),
            ebitda=company['ebitda'],
            net_income=company.get('net_income'),
            total_assets=company['total_assets'],
            total_liabilities=company['total_liabilities'],
            total_equity=company['total_equity'],
            cash_and_equivalents=company['cash_and_equivalents'],
            total_debt=company['total_debt'],
            net_debt=company['net_debt'],
            debt_to_equity=company.get('debt_to_equity'),
            current_ratio=company.get('current_ratio'),
            source_file='sample_fund_report.pdf',
            extraction_timestamp=datetime.now().isoformat()
        )
        
        results = validator.validate(metrics)
        
        print(f"\n{company['company_name']}:")
        print(f"  Validation Status: {'✓ PASSED' if results['is_valid'] else '✗ FAILED'}")
        print(f"  Checks Passed: {results['checks_passed']}/{results['checks_total']}")
        
        if results['warnings']:
            print("  Warnings:")
            for warning in results['warnings']:
                print(f"    ⚠ {warning}")
        
        # Verify balance sheet equation: Assets = Liabilities + Equity
        balance_check = company['total_assets'] - (company['total_liabilities'] + company['total_equity'])
        print(f"  Balance Sheet Check: Assets - (Liab + Equity) = ${balance_check:,}")
        
        # Verify net debt calculation: Net Debt = Total Debt - Cash
        net_debt_check = company['total_debt'] - company['cash_and_equivalents']
        print(f"  Net Debt Check: Calculated = ${net_debt_check:,}, Reported = ${company['net_debt']:,}")
    
    print()
    
    # Step 4: Export to structured format
    print("STEP 4: Export to Structured Format")
    print("-" * 40)
    
    # Create DataFrame for export
    df = pd.DataFrame(extracted_companies)
    
    # Add calculated fields
    df['ebitda_margin'] = (df['ebitda'] / df['revenue'] * 100).round(1)
    df['net_debt_to_ebitda'] = (df['net_debt'] / df['ebitda']).round(2)
    df['extraction_date'] = datetime.now().strftime('%Y-%m-%d')
    df['source_document'] = 'BlackRock Private Credit Fund II Q3 2024'
    
    # Reorder columns for clean output
    output_columns = [
        'company_name', 'industry', 'revenue', 'ebitda', 'ebitda_margin',
        'net_debt', 'net_debt_to_ebitda', 'total_assets', 'total_equity',
        'debt_to_equity', 'extraction_date', 'source_document'
    ]
    
    df_output = df[output_columns]
    
    # Save to CSV
    output_path = 'output/extracted_portfolio_companies.csv'
    os.makedirs('output', exist_ok=True)
    df_output.to_csv(output_path, index=False)
    
    print(f"\nExtracted data saved to: {output_path}")
    print("\nPreview of extracted data:")
    print(df_output.to_string(index=False))
    
    # Save formatted Excel
    excel_path = 'output/extracted_portfolio_companies.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df_output.to_excel(writer, sheet_name='Portfolio Companies', index=False)
        
        # Add summary sheet
        summary_data = {
            'Metric': ['Total Companies', 'Aggregate Revenue', 'Aggregate EBITDA', 
                      'Weighted Avg EBITDA Margin', 'Extraction Date'],
            'Value': [
                len(df),
                f"${df['revenue'].sum():,.0f}",
                f"${df['ebitda'].sum():,.0f}",
                f"{(df['ebitda'].sum() / df['revenue'].sum() * 100):.1f}%",
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"Excel report saved to: {excel_path}")
    
    print()
    print("=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print("""
Key Capabilities Demonstrated:
1. ✓ Parsed unstructured financial data from fund report text
2. ✓ Extracted key metrics: Revenue, EBITDA, Net Debt, Assets/Liabilities
3. ✓ Validated data quality (balance sheet equation, net debt calculation)
4. ✓ Exported to structured formats (CSV/Excel) ready for analysis

In Production:
- Replace simulated LLM with actual API calls (OpenAI/Anthropic)
- Add PDF text extraction using pdfplumber
- Scale to batch processing of multiple fund reports
- Integrate with database for historical comparison
""")

    return df_output


if __name__ == "__main__":
    demonstrate_extraction_pipeline()
