"""Leakage-safe preprocessing, target construction, and graph-ready data export."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from food_safety_gnn.provenance import atomic_write_json, sha256_file, utc_timestamp

REQUIRED_COLUMNS = {
    "Inspection ID",
    "DBA Name",
    "License #",
    "Facility Type",
    "Risk",
    "Address",
    "Zip",
    "Inspection Date",
    "Inspection Type",
    "Results",
    "Violations",
    "Latitude",
    "Longitude",
}

POSITIVE_OUTCOMES = {"Fail"}
NEGATIVE_OUTCOMES = {"Pass", "Pass w/ Conditions"}
EXCLUDED_OUTCOMES = {
    "Out of Business",
    "No Entry",
    "Not Ready",
    "Business Not Located",
}


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for deterministic restaurant-level preprocessing."""

    target_horizon_days: int = 365
    duplicate_policy: str = "drop_exact_duplicates_only"
    entity_resolution_version: str = "license_address_zip_v1"
    same_day_policy: str = "exclude_same_day_as_not_prior"


def normalize_text(value: object) -> str:
    """Normalize an identifier component without imputing missing values."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()


def stable_entity_id(row: pd.Series) -> tuple[str, str]:
    """Return an entity ID and documented fallback method for one inspection row."""
    license_value = normalize_text(row["License #"])
    address = normalize_text(row["Address"])
    zip_code = normalize_text(row["Zip"])
    name = normalize_text(row["DBA Name"])
    if license_value and address:
        source = f"license_address_zip|{license_value}|{address}|{zip_code}"
        method = "license_address_zip"
    else:
        source = f"name_address_zip|{name}|{address}|{zip_code}"
        method = "name_address_zip_fallback"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:20], method


def read_and_validate_raw(csv_path: Path) -> pd.DataFrame:
    """Read raw inspections and validate immutable schema-level assumptions."""
    dataframe = pd.read_csv(csv_path, low_memory=False)
    missing = sorted(REQUIRED_COLUMNS - set(dataframe.columns))
    if missing:
        raise ValueError(f"Raw CSV is missing required columns: {missing}")
    dataframe["Inspection Date"] = pd.to_datetime(
        dataframe["Inspection Date"], errors="coerce", utc=True
    ).dt.tz_localize(None)
    if dataframe["Inspection Date"].isna().any():
        raise ValueError("Inspection Date contains unparseable values.")
    return dataframe


def canonicalize_inspections(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Drop only byte-equivalent rows, validate duplicate IDs, and resolve entities."""
    duplicate_rows = dataframe[dataframe.duplicated(["Inspection ID"], keep=False)].copy()
    non_id_columns = [column for column in dataframe.columns if column != "Inspection ID"]
    conflicting_ids = [
        str(inspection_id)
        for inspection_id, group in duplicate_rows.groupby("Inspection ID", dropna=False)
        if group[non_id_columns].drop_duplicates().shape[0] > 1
    ]
    if conflicting_ids:
        preview = ", ".join(conflicting_ids[:5])
        raise ValueError(
            "Inspection ID collisions contain non-identical rows; "
            f"manual reconciliation is required: {preview}"
        )
    canonical = dataframe.drop_duplicates().copy()
    entity_pairs = canonical.apply(stable_entity_id, axis=1)
    canonical["entity_id"] = [pair[0] for pair in entity_pairs]
    canonical["entity_resolution_method"] = [pair[1] for pair in entity_pairs]
    canonical = canonical.sort_values(
        ["entity_id", "Inspection Date", "Inspection ID"]
    ).reset_index(drop=True)
    report = {
        "input_rows": int(len(dataframe)),
        "duplicate_rows_by_inspection_id": int(len(duplicate_rows)),
        "exact_duplicate_rows_removed": int(len(dataframe) - len(canonical)),
        "conflicting_duplicate_inspection_ids": 0,
        "canonical_rows": int(len(canonical)),
        "entity_count": int(canonical["entity_id"].nunique()),
        "entity_resolution_methods": (
            canonical["entity_resolution_method"].value_counts().to_dict()
        ),
    }
    return canonical, report


def parse_violation_codes(value: object) -> list[str]:
    """Extract distinct observed violation-code prefixes from inspection text."""
    if pd.isna(value):
        return []
    return sorted(set(re.findall(r"(?:^|\|)\s*(\d+)\.", str(value))))


def add_next_inspection_targets(
    canonical: pd.DataFrame, horizon_days: int
) -> pd.DataFrame:
    """Label each event from the next eligible observation within a fixed horizon.

    Same-day records are treated as simultaneous under the date-only timestamp policy.
    Features for an anchor must use records with Inspection Date strictly earlier than
    the anchor date. Target eligibility also requires a strictly later calendar date.
    """
    labeled = canonical.copy()
    labeled["next_inspection_date"] = pd.NaT
    labeled["next_inspection_result"] = pd.NA
    labeled["next_inspection_id"] = pd.NA
    labeled["target_label"] = pd.Series(pd.NA, index=labeled.index, dtype="Int64")
    labeled["target_status"] = "right_censored"
    horizon = pd.Timedelta(days=horizon_days)

    for _, positions in labeled.groupby("entity_id", sort=False).groups.items():
        rows = list(positions)
        for index, current_position in enumerate(rows):
            current_date = labeled.at[current_position, "Inspection Date"]
            for next_position in rows[index + 1 :]:
                next_date = labeled.at[next_position, "Inspection Date"]
                if next_date <= current_date:
                    continue
                if next_date - current_date > horizon:
                    break
                next_result = labeled.at[next_position, "Results"]
                if next_result in POSITIVE_OUTCOMES:
                    labeled.at[current_position, "next_inspection_date"] = next_date
                    labeled.at[current_position, "next_inspection_result"] = next_result
                    labeled.at[current_position, "next_inspection_id"] = labeled.at[
                        next_position, "Inspection ID"
                    ]
                    labeled.at[current_position, "target_label"] = 1
                    labeled.at[current_position, "target_status"] = "eligible"
                    break
                if next_result in NEGATIVE_OUTCOMES:
                    labeled.at[current_position, "next_inspection_date"] = next_date
                    labeled.at[current_position, "next_inspection_result"] = next_result
                    labeled.at[current_position, "next_inspection_id"] = labeled.at[
                        next_position, "Inspection ID"
                    ]
                    labeled.at[current_position, "target_label"] = 0
                    labeled.at[current_position, "target_status"] = "eligible"
                    break
                if next_result in EXCLUDED_OUTCOMES:
                    continue
    return labeled


def build_entity_registry(canonical: pd.DataFrame) -> pd.DataFrame:
    """Build a convenience latest-known registry. Do not use as historical features."""
    latest = (
        canonical.sort_values(["Inspection Date", "Inspection ID"])
        .groupby("entity_id", as_index=False)
        .tail(1)
    )
    columns = [
        "entity_id",
        "entity_resolution_method",
        "License #",
        "DBA Name",
        "Address",
        "Zip",
        "Facility Type",
        "Latitude",
        "Longitude",
        "Inspection Date",
    ]
    return latest[columns].rename(
        columns={"Inspection Date": "latest_observed_inspection_date"}
    )


def build_violation_events(canonical: pd.DataFrame) -> pd.DataFrame:
    """Normalize violation-code events while retaining their original inspection time."""
    selected = canonical[
        ["Inspection ID", "entity_id", "Inspection Date", "Violations"]
    ].copy()
    selected["violation_code"] = selected["Violations"].map(parse_violation_codes)
    return (
        selected.explode("violation_code")
        .dropna(subset=["violation_code"])
        .drop(columns="Violations")
    )


def export_graph_ready_data(
    csv_path: Path, output_directory: Path, config: PreprocessingConfig
) -> dict[str, Any]:
    """Create Parquet graph-ready tables plus immutable provenance and quality metrics."""
    raw_hash = sha256_file(csv_path)
    raw = read_and_validate_raw(csv_path)
    canonical, quality = canonicalize_inspections(raw)
    labeled = add_next_inspection_targets(canonical, config.target_horizon_days)
    registry = build_entity_registry(canonical)
    violations = build_violation_events(canonical)

    output_directory.mkdir(parents=True, exist_ok=True)
    tables = {
        "canonical_inspections.parquet": labeled,
        "entity_registry.parquet": registry,
        "violation_code_events.parquet": violations,
        "eligible_next_inspection_labels.parquet": labeled[
            labeled["target_status"].eq("eligible")
        ].copy(),
    }
    outputs: dict[str, str] = {}
    for filename, table in tables.items():
        destination = output_directory / filename
        table.to_parquet(destination, index=False)
        outputs[filename] = str(destination)

    outcome_counts = raw["Results"].fillna("<missing>").value_counts().to_dict()
    missingness = raw.isna().sum().sort_values(ascending=False).to_dict()
    quality.update(
        {
            "raw_csv": str(csv_path.resolve()),
            "raw_sha256": raw_hash,
            "date_min": raw["Inspection Date"].min().isoformat(),
            "date_max": raw["Inspection Date"].max().isoformat(),
            "outcome_counts": outcome_counts,
            "missingness": {key: int(value) for key, value in missingness.items()},
            "eligible_label_count": int(labeled["target_status"].eq("eligible").sum()),
            "positive_label_count": int((labeled["target_label"] == 1).sum()),
            "negative_label_count": int((labeled["target_label"] == 0).sum()),
            "created_at": utc_timestamp(),
            "config": {
                "target_horizon_days": config.target_horizon_days,
                "duplicate_policy": config.duplicate_policy,
                "entity_resolution_version": config.entity_resolution_version,
                "same_day_policy": config.same_day_policy,
            },
            "outputs": outputs,
        }
    )
    atomic_write_json(output_directory / "quality_report.json", quality)
    return quality
