"""
review_queue.py
===============
Human-in-the-loop review management for flagged extractions.

Flagged items are persisted as individual JSON files inside *review_dir*
so they survive process restarts and can be inspected manually.

Typical workflow
----------------
1. The agent adds items:
       queue = ReviewQueue("review/")
       queue.add(metrics, validation_result, reason="Low confidence (0.42)")

2. A human (or downstream script) lists pending items:
       pending = queue.list_pending()

3. After manual inspection the item is approved or rejected:
       queue.approve(item_id)   # moves JSON to review/approved/
       queue.reject(item_id)    # moves JSON to review/rejected/

Item IDs are auto-generated as ``<company_name>_<ISO-timestamp>``.
"""

import json
import logging
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sub-directories inside review_dir
_PENDING  = "pending"
_APPROVED = "approved"
_REJECTED = "rejected"


class ReviewQueue:
    """
    Persistent queue for extractions that require human sign-off.

    Parameters
    ----------
    review_dir:
        Root directory for review artefacts (default ``review/``).
        Sub-directories ``pending/``, ``approved/``, and ``rejected/``
        are created automatically.
    """

    def __init__(self, review_dir: str = "review"):
        self.review_dir = Path(review_dir)
        self._pending_dir  = self.review_dir / _PENDING
        self._approved_dir = self.review_dir / _APPROVED
        self._rejected_dir = self.review_dir / _REJECTED

        for d in (self._pending_dir, self._approved_dir, self._rejected_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, metrics, validation_result: Dict[str, Any], reason: str) -> str:
        """
        Add a flagged extraction to the pending queue.

        Args:
            metrics:           FinancialMetrics dataclass instance.
            validation_result: Dict returned by DataValidator.validate().
            reason:            Human-readable explanation for the flag.

        Returns:
            The generated item_id string.
        """
        item_id = self._make_id(metrics.company_name)
        item = {
            "item_id":         item_id,
            "reason":          reason,
            "queued_at":       datetime.now().isoformat(),
            "status":          "pending",
            "metrics":         asdict(metrics),
            "validation":      validation_result,
        }
        path = self._pending_dir / f"{item_id}.json"
        self._write_json(path, item)
        logger.info(f"Review queue – added: {item_id}")
        return item_id

    def list_pending(self) -> List[Dict[str, Any]]:
        """
        Return all items currently awaiting review, sorted by queue time.

        Returns:
            List of item dicts (same structure as written by :meth:`add`).
        """
        items = [
            self._read_json(p)
            for p in sorted(self._pending_dir.glob("*.json"))
        ]
        return [i for i in items if i is not None]

    def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single pending item by its ID.

        Args:
            item_id: The ID returned by :meth:`add`.

        Returns:
            Item dict, or None if not found in pending.
        """
        path = self._pending_dir / f"{item_id}.json"
        if not path.exists():
            logger.warning(f"Item not found in pending queue: {item_id}")
            return None
        return self._read_json(path)

    def approve(self, item_id: str, notes: str = "") -> bool:
        """
        Approve a pending item and move it to ``approved/``.

        Args:
            item_id: The ID of the item to approve.
            notes:   Optional reviewer notes recorded in the artefact.

        Returns:
            True on success, False if the item was not found.
        """
        return self._transition(item_id, "approved", notes)

    def reject(self, item_id: str, notes: str = "") -> bool:
        """
        Reject a pending item and move it to ``rejected/``.

        Args:
            item_id: The ID of the item to reject.
            notes:   Optional reviewer notes recorded in the artefact.

        Returns:
            True on success, False if the item was not found.
        """
        return self._transition(item_id, "rejected", notes)

    def summary(self) -> Dict[str, int]:
        """
        Return counts of items in each sub-queue.

        Returns:
            Dict with keys ``pending``, ``approved``, ``rejected``.
        """
        return {
            "pending":  len(list(self._pending_dir.glob("*.json"))),
            "approved": len(list(self._approved_dir.glob("*.json"))),
            "rejected": len(list(self._rejected_dir.glob("*.json"))),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _transition(self, item_id: str, new_status: str, notes: str) -> bool:
        """Move an item from pending to *new_status* sub-directory."""
        src = self._pending_dir / f"{item_id}.json"
        if not src.exists():
            logger.warning(f"Cannot {new_status} – item not found: {item_id}")
            return False

        item = self._read_json(src)
        if item is None:
            return False

        item["status"]      = new_status
        item["reviewed_at"] = datetime.now().isoformat()
        item["notes"]       = notes

        dest_dir = self._approved_dir if new_status == "approved" else self._rejected_dir
        dest = dest_dir / f"{item_id}.json"
        self._write_json(dest, item)
        src.unlink()

        logger.info(f"Review queue – {new_status}: {item_id}")
        return True

    @staticmethod
    def _make_id(company_name: str) -> str:
        """Generate a filesystem-safe unique item ID."""
        safe_name = re.sub(r"[^\w]", "_", company_name).lower()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{safe_name}_{ts}"

    @staticmethod
    def _write_json(path: Path, data: Dict) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

    @staticmethod
    def _read_json(path: Path) -> Optional[Dict]:
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception as exc:
            logger.error(f"Failed to read {path}: {exc}")
            return None
