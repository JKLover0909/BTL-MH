"""Native multi-relational unsupervised GraphSAGE trained with negative sampling."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from food_safety_gnn.provenance import atomic_write_json, utc_timestamp


class CUDAUnavailableError(RuntimeError):
    """Raised when a requested CUDA training run cannot execute on GPU."""


@dataclass(frozen=True)
class GraphSAGEConfig:
    """Training hyperparameters for unsupervised GraphSAGE."""

    embedding_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 2
    learning_rate: float = 1e-3
    epochs: int = 25
    batch_size: int = 1024
    num_negatives: int = 5
    seed: int = 42
    device: str = "cuda:0"


def require_cuda(device: str = "cuda:0") -> torch.device:
    """Validate CUDA availability and run a tiny allocation smoke test."""
    if not device.startswith("cuda"):
        raise CUDAUnavailableError(
            f"Requested device '{device}' is not a CUDA device. "
            "Set device to cuda:<index> for Phase 4 training."
        )
    if not torch.cuda.is_available():
        raise CUDAUnavailableError(
            "torch.cuda.is_available() is False. Install a CUDA-enabled PyTorch build "
            "in meibook-dev and verify the NVIDIA driver."
        )
    selected = torch.device(device)
    try:
        sample = torch.randn((32, 32), device=selected)
        _ = sample @ sample
        torch.cuda.synchronize(selected)
    except Exception as error:  # noqa: BLE001 - surface exact CUDA failure
        raise CUDAUnavailableError(
            f"CUDA smoke test failed on {device}: {error}"
        ) from error
    return selected


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs used by the trainer."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RelationMeanAggregator(nn.Module):
    """Mean-aggregate neighbor messages for one relation."""

    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros_like(node_features)
        source, destination = edge_index[0], edge_index[1]
        messages = node_features.index_select(0, source)
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, destination, messages)
        degree = torch.bincount(destination, minlength=node_features.size(0)).clamp_min(1)
        return aggregated / degree.unsqueeze(1).to(node_features.dtype)


class MultiRelationalGraphSAGE(nn.Module):
    """Two-layer multi-relational GraphSAGE encoder implemented in pure PyTorch."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        relation_names: list[str],
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.relation_names = relation_names
        self.num_layers = num_layers
        self.aggregators = nn.ModuleDict(
            {name: RelationMeanAggregator() for name in relation_names}
        )
        input_width = in_dim * (1 + len(relation_names))
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_width, hidden_dim))
        for _ in range(max(num_layers - 2, 0)):
            self.layers.append(nn.Linear(hidden_dim * (1 + len(relation_names)), hidden_dim))
        self.layers.append(
            nn.Linear(
                hidden_dim * (1 + len(relation_names)) if num_layers > 1 else input_width,
                out_dim,
            )
        )

    def _combine(
        self, features: torch.Tensor, edge_index: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        parts = [features]
        for name in self.relation_names:
            parts.append(self.aggregators[name](features, edge_index[name]))
        return torch.cat(parts, dim=1)

    def forward(
        self, features: torch.Tensor, edge_index: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        hidden = features
        for layer_index, layer in enumerate(self.layers):
            combined = self._combine(hidden, edge_index)
            hidden = layer(combined)
            if layer_index < len(self.layers) - 1:
                hidden = F.relu(hidden)
        return F.normalize(hidden, p=2, dim=1)


def _positive_edges(edge_index: dict[str, torch.Tensor]) -> torch.Tensor:
    chunks = [edges for edges in edge_index.values() if edges.numel()]
    if not chunks:
        raise ValueError("Graph has no edges; cannot train unsupervised GraphSAGE.")
    return torch.cat(chunks, dim=1)


def unsupervised_loss(
    embeddings: torch.Tensor,
    positive_edges: torch.Tensor,
    num_nodes: int,
    num_negatives: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Binary logistic loss over positive graph edges and random negatives."""
    if positive_edges.size(1) == 0:
        raise ValueError("No positive edges available for the current batch.")
    permutation = torch.randperm(positive_edges.size(1), device=device)[:batch_size]
    batch = positive_edges[:, permutation]
    anchors = embeddings.index_select(0, batch[0])
    positives = embeddings.index_select(0, batch[1])
    positive_scores = (anchors * positives).sum(dim=1)
    loss = -F.logsigmoid(positive_scores).mean()
    for _ in range(num_negatives):
        negatives = torch.randint(0, num_nodes, (batch.size(1),), device=device)
        negative_scores = (anchors * embeddings.index_select(0, negatives)).sum(dim=1)
        loss = loss - F.logsigmoid(-negative_scores).mean()
    return loss / (1 + num_negatives)


def train_unsupervised_graphsage(
    features: torch.Tensor,
    edge_index: dict[str, torch.Tensor],
    config: GraphSAGEConfig,
    output_directory: Path,
    entity_ids: list[str],
    cutoff: str,
) -> dict[str, Any]:
    """Train GraphSAGE on CUDA, save weights/embeddings, and return run metadata."""
    set_seed(config.seed)
    device = require_cuda(config.device)
    output_directory.mkdir(parents=True, exist_ok=True)
    relation_names = sorted(edge_index.keys())
    model = MultiRelationalGraphSAGE(
        in_dim=features.size(1),
        hidden_dim=config.hidden_dim,
        out_dim=config.embedding_dim,
        relation_names=relation_names,
        num_layers=config.num_layers,
    ).to(device)
    features = features.to(device)
    edge_index = {name: edges.to(device) for name, edges in edge_index.items()}
    positives = _positive_edges(edge_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []

    model.train()
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(features, edge_index)
        loss = unsupervised_loss(
            embeddings=embeddings,
            positive_edges=positives,
            num_nodes=features.size(0),
            num_negatives=config.num_negatives,
            batch_size=min(config.batch_size, positives.size(1)),
            device=device,
        )
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(loss.detach().cpu().item())})

    model.eval()
    with torch.no_grad():
        final_embeddings = model(features, edge_index).detach().cpu().numpy()

    weights_path = output_directory / "graphsage_weights.pt"
    embeddings_path = output_directory / "entity_embeddings.parquet"
    history_path = output_directory / "training_history.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "relation_names": relation_names,
            "in_dim": features.size(1),
            "cutoff": cutoff,
        },
        weights_path,
    )
    embedding_frame = pd.DataFrame(final_embeddings, columns=[f"emb_{i}" for i in range(final_embeddings.shape[1])])
    embedding_frame.insert(0, "entity_id", entity_ids)
    embedding_frame.insert(1, "snapshot_cutoff", cutoff)
    embedding_frame.to_parquet(embeddings_path, index=False)
    atomic_write_json(history_path, history)
    metadata = {
        "created_at": utc_timestamp(),
        "device": str(device),
        "cuda_available": True,
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "epochs": config.epochs,
        "final_loss": history[-1]["loss"],
        "num_nodes": int(features.size(0)),
        "embedding_dim": config.embedding_dim,
        "cutoff": cutoff,
        "weights_path": str(weights_path),
        "embeddings_path": str(embeddings_path),
        "history_path": str(history_path),
        "config": asdict(config),
    }
    atomic_write_json(output_directory / "run_manifest.json", metadata)
    metadata["history"] = history
    metadata["embeddings"] = embedding_frame
    return metadata
