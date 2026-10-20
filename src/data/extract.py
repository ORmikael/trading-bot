"""
Market Data Extractor

Extracts archived market-data files into:

    data/extracted/<SYMBOL>/<SOURCE>/

Expected TrueFX archive structure:

    outer.zip
        └── inner.zip
                └── CSV

Primary interface:

    extract_source(symbol, source)

CLI:

    python -m src.data.extract --symbol EURUSD --source TrueFX
"""

from __future__ import annotations

import argparse
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTRACTED_DIR = PROJECT_ROOT / "data" / "extracted"


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class ExtractionResult:
    """Result of an extraction operation."""

    symbol: str
    source: str
    archives_processed: int
    files_extracted: int
    extracted_files: list[Path]
    output_directory: Path


# ============================================================
# ARCHIVE EXTRACTION
# ============================================================

def extract_archive(
    archive_path: Path,
    output_dir: Path,
) -> list[Path]:
    """
    Extract CSV datasets from a nested archive.

    Expected structure:

        outer.zip
            └── inner.zip
                    └── CSV

    Returns:
        List of extracted CSV paths.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    extracted_csvs: list[Path] = []

    with zipfile.ZipFile(
        archive_path,
        "r",
    ) as outer_zip:

        inner_archives = [
            name
            for name in outer_zip.namelist()
            if name.lower().endswith(".zip")
        ]

        if not inner_archives:
            raise ValueError(
                f"No inner ZIP archive found in "
                f"{archive_path}"
            )

        for inner_name in inner_archives:

            inner_data = outer_zip.read(
                inner_name
            )

            with zipfile.ZipFile(
                io.BytesIO(inner_data),
                "r",
            ) as inner_zip:

                csv_files = [
                    name
                    for name in inner_zip.namelist()
                    if name.lower().endswith(".csv")
                ]

                if not csv_files:
                    raise ValueError(
                        f"No CSV dataset found inside "
                        f"{inner_name}"
                    )

                for csv_name in csv_files:

                    destination = (
                        output_dir
                        / Path(csv_name).name
                    )

                    with (
                        inner_zip.open(csv_name) as source,
                        destination.open("wb") as target,
                    ):
                        target.write(source.read())

                    extracted_csvs.append(
                        destination
                    )

    return extracted_csvs


# ============================================================
# SOURCE EXTRACTION
# ============================================================

def extract_source(
    symbol: str,
    source: str,
) -> ExtractionResult:
    """
    Extract all archives belonging to a
    symbol/data source.

    This is the primary interface used by main.py.
    """

    raw_dir = (
        RAW_DIR
        / symbol
        / source
    )

    extracted_dir = (
        EXTRACTED_DIR
        / symbol
        / source
    )

    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory does not exist:\n"
            f"{raw_dir}"
        )

    archives = sorted(
        raw_dir.glob("*.zip")
    )

    if not archives:
        raise FileNotFoundError(
            f"No ZIP archives found in:\n"
            f"{raw_dir}"
        )

    extracted_files: list[Path] = []

    for archive in archives:

        files = extract_archive(
            archive_path=archive,
            output_dir=extracted_dir,
        )

        extracted_files.extend(files)

    return ExtractionResult(
        symbol=symbol,
        source=source,
        archives_processed=len(archives),
        files_extracted=len(extracted_files),
        extracted_files=extracted_files,
        output_directory=extracted_dir,
    )


# ============================================================
# REPORTING
# ============================================================

def print_result(
    result: ExtractionResult,
) -> None:
    """
    Print a human-readable extraction report.
    """

    print()
    print("=" * 70)
    print("EXTRACTION REPORTeeeeeeeeeeee")
    print("=" * 70)

    print()
    print("Configuration")
    print("-------------")

    print(
        f"Symbol:              {result.symbol}"
    )

    print(
        f"Source:              {result.source}"
    )

    print()
    print("Extraction")
    print("----------")

    print(
        f"Archives processed:  "
        f"{result.archives_processed}"
    )

    print(
        f"Files extracted:     "
        f"{result.files_extracted}"
    )

    print(
        f"Output directory:    "
        f"{result.output_directory}"
    )

    print()
    print("Files")
    print("-----")

    for path in result.extracted_files:
        print(f"  {path}")

    print()
    print("Status")
    print("------")
    print("PASS")

    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def main() -> None:
    """CLI wrapper for standalone execution."""

    parser = argparse.ArgumentParser(
        description="Extract market-data archives."
    )

    parser.add_argument(
        "--symbol",
        required=True,
        help="Trading symbol, e.g. EURUSD",
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Data source, e.g. TrueFX",
    )

    args = parser.parse_args()

    result = extract_source(
        symbol=args.symbol,
        source=args.source,
    )

    print_result(result)


if __name__ == "__main__":
    main()