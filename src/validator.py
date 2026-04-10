"""
validator.py
============
Validates extracted financial data for quality and consistency.

Implements range checks, logical balance-sheet rules, net-debt sanity
checks, EBITDA-margin plausibility, and optional historical z-score
anomaly detection.

Author: Arnab Banerjee
"""

import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Runs automated quality checks on a FinancialMetrics instance.

    Usage
    -----
    validator = DataValidator(historical_data=df)   # df is optional
    result = validator.validate(metrics)
    # result["is_valid"]  -> bool
    # result["errors"]    -> list of hard failures
    # result["warnings"]  -> list of soft anomalies
    """

    VALIDATION_RULES: Dict[str, Dict[str, float]] = {
        "revenue":        {"min": 0,     "max": 1e12},
        "ebitda":         {"min": -1e11, "max": 1e11},
        "net_debt":       {"min": -1e11, "max": 1e11},
        "total_assets":   {"min": 0,     "max": 1e13},
        "debt_to_equity": {"min": -10,   "max": 50},
    }

    def __init__(self, historical_data: Optional[pd.DataFrame] = None):
        """
        Args:
            historical_data: Optional DataFrame with prior-period metrics
                             for z-score comparison.  Must contain a
                             ``company_name`` column.
        """
        self.historical_data = historical_data
        self.validation_results: List[Dict] = []

    def validate(self, metrics) -> Dict[str, Any]:
        """
        Run all validation checks on *metrics*.

        Args:
            metrics: A FinancialMetrics dataclass instance.

        Returns:
            Dict with keys:
              ``is_valid``      – False if any hard error was found
              ``errors``        – list of error strings
              ``warnings``      – list of warning strings
              ``checks_passed`` – count of checks that passed
              ``checks_total``  – total checks attempted
        """
        results: Dict[str, Any] = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "checks_passed": 0,
            "checks_total": 0,
        }

        metrics_dict = asdict(metrics)

        # ── Range checks ────────────────────────────────────────────────
        for field, rules in self.VALIDATION_RULES.items():
            value = metrics_dict.get(field)
            if value is not None:
                results["checks_total"] += 1
                if rules["min"] <= value <= rules["max"]:
                    results["checks_passed"] += 1
                else:
                    results["errors"].append(
                        f"{field}: value {value} outside valid range "
                        f"[{rules['min']}, {rules['max']}]"
                    )
                    results["is_valid"] = False

        # ── Logical consistency checks ───────────────────────────────────
        results["checks_total"] += 3

        # 1. Balance sheet equation: Assets = Liabilities + Equity
        bs_fields = ["total_assets", "total_liabilities", "total_equity"]
        if all(metrics_dict.get(f) is not None for f in bs_fields):
            imbalance = metrics.total_assets - (
                metrics.total_liabilities + metrics.total_equity
            )
            if abs(imbalance) > 0.01 * metrics.total_assets:
                results["warnings"].append(
                    f"Balance sheet imbalance: Assets − (Liabilities + Equity) "
                    f"= {imbalance:,.0f}"
                )
            else:
                results["checks_passed"] += 1

        # 2. Net Debt = Total Debt − Cash
        nd_fields = ["net_debt", "total_debt", "cash_and_equivalents"]
        if all(metrics_dict.get(f) is not None for f in nd_fields):
            expected_nd = metrics.total_debt - metrics.cash_and_equivalents
            if abs(metrics.net_debt - expected_nd) > 0.01 * max(abs(expected_nd), 1):
                results["warnings"].append(
                    f"Net debt inconsistency: expected {expected_nd:,.0f}, "
                    f"got {metrics.net_debt:,.0f}"
                )
            else:
                results["checks_passed"] += 1

        # 3. EBITDA margin plausibility
        if (
            metrics.ebitda is not None
            and metrics.revenue is not None
            and metrics.revenue > 0
        ):
            margin = metrics.ebitda / metrics.revenue
            if not -0.5 <= margin <= 0.8:
                results["warnings"].append(
                    f"Unusual EBITDA margin: {margin:.1%}"
                )
            else:
                results["checks_passed"] += 1

        # ── Historical comparison ────────────────────────────────────────
        if self.historical_data is not None:
            self._compare_to_historical(metrics, results)

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compare_to_historical(self, metrics, results: Dict[str, Any]) -> None:
        """Flag metrics that deviate by more than 3 std-devs from history."""
        try:
            historical = self.historical_data[
                self.historical_data["company_name"] == metrics.company_name
            ]
            if historical.empty:
                return

            for field in ("revenue", "ebitda", "total_assets"):
                current = getattr(metrics, field)
                if current is None or field not in historical.columns:
                    continue

                hist_mean = historical[field].mean()
                hist_std = historical[field].std()

                if hist_std and hist_std > 0:
                    z = (current - hist_mean) / hist_std
                    if abs(z) > 3:
                        results["warnings"].append(
                            f"{field}: significant deviation from history (z={z:.1f})"
                        )
        except Exception as exc:
            logger.warning(f"Historical comparison failed: {exc}")
