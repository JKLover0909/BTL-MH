#!/usr/bin/env python
"""Extract native PDF text and a provenance-preserving OCR comparison."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from food_safety_gnn.context_extraction import ContextExtractionError, extract_context


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse the Phase 2 extraction command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPOSITORY_ROOT / "X.pdf")
    parser.add_argument("--output", type=Path, default=REPOSITORY_ROOT / "context_X.md")
    parser.add_argument(
        "--provenance",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts" / "phase2" / "context_extraction_provenance.json",
    )
    parser.add_argument("--language", default="eng")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--ocr",
        choices=("on", "off"),
        default="on",
        help="Use Tesseract for comparison in addition to native PDF extraction.",
    )
    return parser.parse_args()


def main() -> int:
    """Run extraction and report output/provenance locations."""
    arguments = parse_arguments()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        metadata = extract_context(
            source_pdf=arguments.input,
            output_markdown=arguments.output,
            provenance_path=arguments.provenance,
            ocr_enabled=arguments.ocr == "on",
            language=arguments.language,
            dpi=arguments.dpi,
        )
    except ContextExtractionError as error:
        logging.error("Context extraction failed: %s", error)
        return 2

    logging.info(
        "Extracted %s pages from %s into %s (OCR=%s).",
        metadata["page_count"],
        metadata["source"],
        metadata["output"],
        metadata["ocr_requested"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
