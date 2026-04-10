"""
fund-report-agent · src
=======================
AI-powered extraction of financial metrics from unstructured PDF reports.
"""

from .extractor import FinancialDataExtractor, FinancialMetrics, PDFTextExtractor, LLMPromptBuilder
from .validator import DataValidator
from .agent import ExtractionAgent
from .review_queue import ReviewQueue

__version__ = "2.0.0"
__author__ = "Arnab Banerjee"
