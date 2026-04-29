"""CSV loader for digitised experimental data.

Datasets live in ``rydberg_trampoline/data/experimental/`` as CSVs with two
or three columns: ``x, y[, yerr]``. Each CSV has a sibling ``.yaml`` sidecar
that records provenance (source figure, extraction date, accuracy claim).

The expected schemas, by figure:

* ``fig2_decay.csv`` — columns ``time_us, m_afm_res`` for a single Δ_l trace.
* ``fig3_gamma.csv`` — columns ``inv_delta_l, gamma_per_us``.
* ``fig4_bubble_hist.csv`` — columns ``L, sigma_off, sigma_on``.

Until digitisation is complete, this module returns an empty record and a
clear message; figure scripts can no-op the overlay without breaking. See
``rydberg_trampoline/data/experimental/PROVENANCE.md`` for the methodology.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

DATA_ROOT = Path(__file__).resolve().parent / "experimental"


@dataclass(slots=True)
class ExperimentalDataset:
    """In-memory representation of a digitised CSV."""

    name: str
    columns: list[str]
    rows: list[list[float]] = field(default_factory=list)
    provenance: str = ""

    def column(self, name: str) -> list[float]:
        idx = self.columns.index(name)
        return [row[idx] for row in self.rows]


def available_datasets() -> Sequence[str]:
    """Return the list of dataset names (CSV stems) currently shipped."""
    if not DATA_ROOT.exists():
        return []
    return sorted(p.stem for p in DATA_ROOT.glob("*.csv"))


def load_experimental_csv(name: str) -> ExperimentalDataset | None:
    """Load a digitised dataset by stem name (e.g. ``"fig3_gamma"``).

    Returns ``None`` if the CSV is missing — callers (figure scripts) should
    treat this as "no overlay available" and continue, since digitisation is
    a manual upstream step.
    """
    csv_path = DATA_ROOT / f"{name}.csv"
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        rows = [row for row in reader if row and not row[0].startswith("#")]
    if not rows:
        return None
    columns = rows[0]
    data = [[float(x) for x in row] for row in rows[1:]]
    yaml_path = DATA_ROOT / f"{name}.yaml"
    provenance = yaml_path.read_text(encoding="utf-8") if yaml_path.exists() else ""
    return ExperimentalDataset(
        name=name, columns=columns, rows=data, provenance=provenance
    )
