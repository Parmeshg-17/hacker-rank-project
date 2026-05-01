"""
utils/csv_handler.py
--------------------
CSV reading and writing utilities.
"""

from __future__ import annotations

import csv
import pathlib


def read_tickets(path: pathlib.Path) -> list[dict]:
    """Read support ticket CSV and return list of row dicts."""
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_output(results: list[dict], path: pathlib.Path) -> None:
    """Write agent output CSV."""
    fieldnames = [
        "issue", "subject", "company",
        "status", "product_area", "response",
        "justification", "request_type",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(results)
