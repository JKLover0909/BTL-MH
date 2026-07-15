#!/usr/bin/env python
"""Create validated, Parquet graph-ready artifacts from food-inspections.csv."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from food_safety_gnn.data_pipeline import PreprocessingConfig, export_graph_ready_data

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    """Parse input, output, and configuration locations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPOSITORY_ROOT / "configs/phase3/preprocessing_v1.yaml"
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def main() -> int:
    """Build artifacts and emit a compact quality summary."""
    arguments = parse_arguments()
    configuration = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    input_path = arguments.input or REPOSITORY_ROOT / configuration["raw_input"]
    output_directory = arguments.output_directory or REPOSITORY_ROOT / configuration["output_directory"]
    config = PreprocessingConfig(
        target_horizon_days=configuration["cleaning"]["target_horizon_days"],
        duplicate_policy=configuration["cleaning"]["duplicate_policy"],
        entity_resolution_version=configuration["entity_resolution"]["version"],
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        report = export_graph_ready_data(input_path, output_directory, config)
    except (OSError, ValueError) as error:
        logging.error("Graph-ready preprocessing failed: %s", error)
        return 2
    logging.info("Graph-ready data created in %s", output_directory)
    logging.info(
        "Rows=%s canonical=%s entities=%s eligible_targets=%s",
        report["input_rows"],
        report["canonical_rows"],
        report["entity_count"],
        report["eligible_label_count"],
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
