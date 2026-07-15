"""Temporal graph snapshot construction for multi-relational restaurant graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import BallTree
from sklearn.preprocessing import StandardScaler

from food_safety_gnn.provenance import atomic_write_json, utc_timestamp

EARTH_RADIUS_M = 6_371_000.0


@dataclass(frozen=True)
class GraphBuildConfig:
    """Deterministic graph construction settings for one snapshot."""

    geo_k: int = 8
    geo_radius_m: float = 750.0
    facility_peer_cap: int = 8
    zip_peer_cap: int = 8
    history_k: int = 8
    history_dim: int = 24
    max_nodes: int | None = None
    seed: int = 42


def _history_before(history: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    return history.loc[history["Inspection Date"] < cutoff].copy()


def build_asof_entity_state(
    canonical: pd.DataFrame, cutoff: pd.Timestamp, max_nodes: int | None = None
) -> pd.DataFrame:
    """Return one entity state using only inspections strictly before cutoff."""
    history = _history_before(canonical, cutoff)
    if history.empty:
        raise ValueError(f"No historical inspections before cutoff {cutoff.date()}")
    history = history.sort_values(["entity_id", "Inspection Date", "Inspection ID"])
    state = history.groupby("entity_id", as_index=False).tail(1).copy()
    counts = history.groupby("entity_id").size().rename("prior_inspection_count")
    fails = (
        history.assign(is_fail=history["Results"].eq("Fail").astype(int))
        .groupby("entity_id")["is_fail"]
        .sum()
        .rename("prior_fail_count")
    )
    state = state.merge(counts, on="entity_id", how="left").merge(
        fails, on="entity_id", how="left"
    )
    state["days_since_last_inspection"] = (
        cutoff - state["Inspection Date"]
    ).dt.days.astype(float)
    state["prior_fail_rate"] = (
        state["prior_fail_count"] / state["prior_inspection_count"].clip(lower=1)
    )
    state["missing_coordinates"] = (
        state["Latitude"].isna() | state["Longitude"].isna()
    ).astype(int)
    state["snapshot_cutoff"] = cutoff
    state = state.sort_values("entity_id").reset_index(drop=True)
    if max_nodes is not None and len(state) > max_nodes:
        # Deterministic downsampling for notebook-scale training.
        state = state.sample(n=max_nodes, random_state=42).sort_values("entity_id")
        state = state.reset_index(drop=True)
    state["node_index"] = np.arange(len(state), dtype=np.int64)
    return state


def _peer_edges(groups: pd.Series, node_index: pd.Series, peer_cap: int) -> np.ndarray:
    edges: list[list[int]] = []
    frame = pd.DataFrame({"group": groups, "node_index": node_index}).dropna()
    for _, members in frame.groupby("group")["node_index"]:
        ordered = sorted(int(value) for value in members.tolist())
        if len(ordered) < 2:
            continue
        for position, source in enumerate(ordered):
            right = ordered[position + 1 : position + 1 + peer_cap]
            left_start = max(0, position - peer_cap)
            left = ordered[left_start:position]
            for target in left + right:
                if source != target:
                    edges.append([source, target])
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def _geo_edges(state: pd.DataFrame, geo_k: int, geo_radius_m: float) -> np.ndarray:
    coords = state.loc[state["missing_coordinates"].eq(0), ["Latitude", "Longitude", "node_index"]]
    if len(coords) < 2:
        return np.zeros((2, 0), dtype=np.int64)
    radians = np.radians(coords[["Latitude", "Longitude"]].to_numpy(dtype=np.float64))
    tree = BallTree(radians, metric="haversine")
    k = min(geo_k + 1, len(coords))
    distances, indices = tree.query(radians, k=k)
    edges: list[list[int]] = []
    node_ids = coords["node_index"].to_numpy(dtype=np.int64)
    max_distance = geo_radius_m / EARTH_RADIUS_M
    for row_index, neighbors in enumerate(indices):
        source = int(node_ids[row_index])
        for neighbor_pos, neighbor_index in enumerate(neighbors):
            if neighbor_pos == 0:
                continue
            if distances[row_index, neighbor_pos] > max_distance:
                continue
            target = int(node_ids[neighbor_index])
            if source != target:
                edges.append([source, target])
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def _history_edges(
    state: pd.DataFrame,
    violation_events: pd.DataFrame,
    cutoff: pd.Timestamp,
    history_k: int,
    history_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    history = violation_events.loc[violation_events["Inspection Date"] < cutoff].copy()
    if history.empty:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((len(state), history_dim), dtype=np.float32)
    codes = sorted(history["violation_code"].astype(str).unique().tolist())[:history_dim]
    code_to_index = {code: index for index, code in enumerate(codes)}
    features = np.zeros((len(state), history_dim), dtype=np.float32)
    entity_to_row = {
        entity_id: index for index, entity_id in enumerate(state["entity_id"].tolist())
    }
    for entity_id, group in history.groupby("entity_id"):
        row = entity_to_row.get(entity_id)
        if row is None:
            continue
        for code, count in group["violation_code"].astype(str).value_counts().items():
            column = code_to_index.get(code)
            if column is not None:
                features[row, column] = float(count)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = features / norms
    active = np.where(features.sum(axis=1) > 0)[0]
    edges: list[list[int]] = []
    if len(active) >= 2:
        similarity = normalized[active] @ normalized[active].T
        np.fill_diagonal(similarity, -np.inf)
        k = min(history_k, len(active) - 1)
        top = np.argpartition(similarity, -k, axis=1)[:, -k:]
        for local_index, neighbor_locals in enumerate(top):
            source = int(active[local_index])
            for neighbor_local in neighbor_locals:
                target = int(active[neighbor_local])
                if source != target and similarity[local_index, neighbor_local] > 0:
                    edges.append([source, target])
    edge_index = (
        np.asarray(edges, dtype=np.int64).T
        if edges
        else np.zeros((2, 0), dtype=np.int64)
    )
    return edge_index, features


def build_node_features(state: pd.DataFrame, history_features: np.ndarray) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """Create a compact numeric feature matrix for GraphSAGE nodes."""
    categorical = pd.get_dummies(
        state[["Risk", "Facility Type"]].fillna("Unknown").astype(str),
        dummy_na=False,
    )
    numeric = state[
        [
            "prior_inspection_count",
            "prior_fail_count",
            "prior_fail_rate",
            "days_since_last_inspection",
            "missing_coordinates",
            "Latitude",
            "Longitude",
        ]
    ].copy()
    numeric["Latitude"] = numeric["Latitude"].fillna(numeric["Latitude"].median())
    numeric["Longitude"] = numeric["Longitude"].fillna(numeric["Longitude"].median())
    matrix = np.hstack(
        [
            numeric.to_numpy(dtype=np.float32),
            categorical.to_numpy(dtype=np.float32),
            history_features.astype(np.float32),
        ]
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix).astype(np.float32)
    feature_names = (
        numeric.columns.tolist()
        + categorical.columns.tolist()
        + [f"hist_{index}" for index in range(history_features.shape[1])]
    )
    return scaled, scaler, feature_names


def build_snapshot(
    canonical: pd.DataFrame,
    violation_events: pd.DataFrame,
    cutoff: pd.Timestamp | str,
    config: GraphBuildConfig,
    output_directory: Path | None = None,
) -> dict[str, Any]:
    """Build one leakage-safe multi-relational graph snapshot."""
    cutoff_ts = pd.Timestamp(cutoff)
    state = build_asof_entity_state(canonical, cutoff_ts, max_nodes=config.max_nodes)
    history_edges, history_features = _history_edges(
        state,
        violation_events,
        cutoff_ts,
        history_k=config.history_k,
        history_dim=config.history_dim,
    )
    features, scaler, feature_names = build_node_features(state, history_features)
    relations = {
        "geo": _geo_edges(state, config.geo_k, config.geo_radius_m),
        "facility": _peer_edges(state["Facility Type"], state["node_index"], config.facility_peer_cap),
        "zip": _peer_edges(state["Zip"], state["node_index"], config.zip_peer_cap),
        "history": history_edges,
    }
    diagnostics = {
        "cutoff": cutoff_ts.date().isoformat(),
        "num_nodes": int(len(state)),
        "feature_dim": int(features.shape[1]),
        "relations": {
            name: {
                "num_edges": int(edge_index.shape[1]),
                "avg_degree": float(
                    edge_index.shape[1] / max(len(state), 1)
                ),
            }
            for name, edge_index in relations.items()
        },
        "created_at": utc_timestamp(),
        "config": asdict(config),
    }
    payload = {
        "cutoff": cutoff_ts,
        "state": state,
        "x": torch.tensor(features, dtype=torch.float32),
        "edge_index": {
            name: torch.tensor(edge_index, dtype=torch.long)
            for name, edge_index in relations.items()
        },
        "feature_names": feature_names,
        "scaler": scaler,
        "diagnostics": diagnostics,
    }
    if output_directory is not None:
        output_directory.mkdir(parents=True, exist_ok=True)
        state.to_parquet(output_directory / "node_state.parquet", index=False)
        torch.save(
            {
                "x": payload["x"],
                "edge_index": payload["edge_index"],
                "feature_names": feature_names,
                "cutoff": cutoff_ts.date().isoformat(),
            },
            output_directory / "graph.pt",
        )
        atomic_write_json(output_directory / "graph_diagnostics.json", diagnostics)
    return payload
