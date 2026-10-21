"""Shared dataset identity and operation-context contract.

This module defines the data structures used to identify selected datasets and
pass dataset selections between independent or parent/child operations.

It intentionally does not perform data discovery, extraction, validation,
normalization, bar generation, backtesting, persistence, or duplicate detection.
Those responsibilities belong to the relevant orchestration and pipeline modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import Literal
import uuid


SelectionOrigin = Literal["explicit", "inherited", "saved"]


def canonical_dataset_name(value: str) -> str:
    """Return a dataset name without a recognized data-file extension."""
    name = PurePath(value).name

    for suffix in (".parquet", ".csv", ".zip"):
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break

    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid dataset name: {value!r}")

    return name


@dataclass(frozen=True)
class DatasetRef:
    """Stable logical identity for one source dataset.

    Paths are deliberately not stored here. Pipeline stages resolve their own
    artefact paths from this logical identity and their directory conventions.
    """

    symbol: str
    source: str
    dataset_name: str

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty")
        if not self.source.strip():
            raise ValueError("source cannot be empty")

        object.__setattr__(
            self,
            "dataset_name",
            canonical_dataset_name(self.dataset_name),
        )

    @property
    def dataset_id(self) -> str:
        """Return a deterministic identity for this logical dataset."""
        return f"{self.source}:{self.symbol}:{self.dataset_name}"

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation for process handoff."""
        return {
            "dataset_id": self.dataset_id,
            "symbol": self.symbol,
            "source": self.source,
            "dataset_name": self.dataset_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetRef":
        """Reconstruct a reference and verify any supplied dataset ID."""
        ref = cls(
            symbol=data["symbol"],
            source=data["source"],
            dataset_name=data["dataset_name"],
        )
        supplied_id = data.get("dataset_id")
        if supplied_id is not None and supplied_id != ref.dataset_id:
            raise ValueError(
                "Dataset reference identity mismatch: "
                f"expected {ref.dataset_id!r}, received {supplied_id!r}"
            )
        return ref


@dataclass(frozen=True)
class DatasetSelection:
    """Immutable, ordered selection of one or more dataset references."""

    datasets: tuple[DatasetRef, ...]