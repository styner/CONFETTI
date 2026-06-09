"""GNN architectures.

Three models share the same input/output convention:
  - Input: per-subject node features (shared graph topology per level)
  - Output: raw logits/scores of shape (batch, out_dim)
    * Classification: out_dim=1 + BCEWithLogitsLoss
    * Multi-output regression: out_dim=K + MSE/Huber loss

Each model implements the same `forward(batch)` API. `batch` is a small dict
holding all level tensors so that downstream training code is identical across
the three architectures:

    batch = {
        "x":           list of (B*N_L, F) tensors, one per level (or just L0 for
                       SingleLevelGCN),
        "edge_index":  list of (2, B*E_L) tensors,
        "edge_weight": list of (B*E_L,)  tensors,
        "batch":       list of (B*N_L,)  graph-id tensors (PyG convention),
        "pool_maps":   list of (N_{L+1},) tensors mapping coarser -> finer pos
                       (used by Graph U-Net only),
    }

A helper `make_batch()` builds such a dict from numpy arrays of shape
(B, N_L, F) per level.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class GCNBlock(nn.Module):
    """One GCNConv + (optional) BatchNorm1d + ReLU + dropout.

    We set add_self_loops=False because the batched edge_index already
    contains the per-graph self-loops we added in `make_level_batch`. Letting
    PyG add them dynamically at every forward triggered a subtle size-mismatch
    crash inside gcn_norm during the long permutation-importance loop.
    Preadding them makes the call deterministic and self-contained.

    BatchNorm1d is applied on the (B*N, F_out) flat node tensor after the
    conv, before the ReLU. It is cheap (one running mean/var per channel) and
    consistently helps in small-graph settings where the input feature
    distribution shifts between folds.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.conv = GCNConv(in_dim, out_dim, add_self_loops=False, normalize=True)
        self.bn = nn.BatchNorm1d(out_dim) if use_batchnorm else None
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_weight=edge_weight)
        if self.bn is not None:
            h = self.bn(h)
        h = F.relu(h)
        return F.dropout(h, p=self.dropout, training=self.training)


def _pool_concat(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """Global mean + max pool, concatenated."""
    return torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=-1)


# ---------------------------------------------------------------------------
# Batching helper for shared-topology graphs
# ---------------------------------------------------------------------------


def make_level_batch(
    x_BNF: np.ndarray | torch.Tensor,
    edge_index_2E: np.ndarray,
    edge_weight_E: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Stack B subjects sharing identical (edge_index, edge_weight) into a PyG-
    style batched representation, with per-graph self-loops pre-added.

      x_BNF: (B, N, F) numpy or torch tensor
      edge_index_2E: (2, E) numpy
      edge_weight_E: (E,)  numpy

    Returns dict with 'x', 'edge_index', 'edge_weight', 'batch'. The returned
    edge_index has shape (2, B*(E+N)) and edge_weight has shape (B*(E+N),) --
    the per-graph self-loop block is concatenated after the per-graph
    real-edge block.
    """
    if isinstance(x_BNF, np.ndarray):
        x_BNF = torch.from_numpy(x_BNF)
    B, N, _ = x_BNF.shape
    x = x_BNF.reshape(B * N, -1).to(device)
    e_t = torch.as_tensor(edge_index_2E, dtype=torch.long, device=device)
    E = e_t.size(1)
    # Replicate real edges B times with per-graph offset.
    offsets = (torch.arange(B, device=device).unsqueeze(0) * N).unsqueeze(0)  # (1, 1, B)
    edge_batched = e_t.unsqueeze(-1) + offsets                                # (2, E, B)
    edge_batched = edge_batched.permute(0, 2, 1).reshape(2, -1)               # (2, B*E)
    w = torch.as_tensor(edge_weight_E, dtype=torch.float32, device=device).repeat(B)

    # Pre-add self-loops (one per node per graph). GCNConv is configured with
    # add_self_loops=False so the only self-loops in the message-passing graph
    # are these.
    node_idx = torch.arange(B * N, device=device)
    self_loop_index = torch.stack([node_idx, node_idx], dim=0)               # (2, B*N)
    self_loop_weight = torch.ones(B * N, dtype=torch.float32, device=device)

    edge_full = torch.cat([edge_batched, self_loop_index], dim=1)
    weight_full = torch.cat([w, self_loop_weight], dim=0)

    batch = torch.arange(B, device=device).repeat_interleave(N)
    return {"x": x, "edge_index": edge_full, "edge_weight": weight_full, "batch": batch}


def make_pool_map_batch(
    pool_map: np.ndarray, n_fine: int, B: int, device: torch.device
) -> torch.Tensor:
    """Replicate a (N_coarse,) -> N_fine pool map across B graphs.

    Returns (B * N_coarse,) global indices into the fine-level flat tensor.
    """
    pm = torch.as_tensor(pool_map, dtype=torch.long, device=device)
    offsets = (torch.arange(B, device=device) * n_fine).unsqueeze(1)
    return (pm.unsqueeze(0) + offsets).reshape(-1)


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class SingleLevelGCN(nn.Module):
    """Two-layer GCN at a single resolution + global mean+max pool + MLP head.

    Constructor takes only the input dim, head dim, and dropout; the model is
    agnostic to which level it's used at (caller passes the corresponding
    batched tensors).

    If ``n_covariates > 0`` the head accepts an optional (B, n_covariates)
    tensor concatenated after the pooled graph embedding -- this is how
    subject-level covariates (sex, gestational age, num_DWI_artifact) enter
    the model. The covariate columns are assumed to be already z-scored by
    the trainer using training-fold stats.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 1,
        n_gcn_layers: int = 2,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        n_covariates: int = 0,
    ):
        super().__init__()
        d = in_dim
        layers: list[nn.Module] = []
        for _ in range(n_gcn_layers):
            layers.append(GCNBlock(d, hidden_dim, dropout, use_batchnorm=use_batchnorm))
            d = hidden_dim
        self.gcns = nn.ModuleList(layers)
        self.n_covariates = n_covariates
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim + n_covariates, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        batches: list[dict[str, torch.Tensor]],
        covariates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = batches[0]
        x = b["x"]
        for layer in self.gcns:
            x = layer(x, b["edge_index"], b["edge_weight"])
        pooled = _pool_concat(x, b["batch"])
        if self.n_covariates > 0 and covariates is not None:
            pooled = torch.cat([pooled, covariates], dim=-1)
        return self.head(pooled)


class MultiScaleConcat(nn.Module):
    """Per-level mini-GCN + global pool, concatenated across levels, MLP head."""

    def __init__(
        self,
        in_dim: int,
        n_levels: int,
        hidden_dim: int = 64,
        out_dim: int = 1,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        n_covariates: int = 0,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_covariates = n_covariates
        self.branches = nn.ModuleList(
            [
                nn.ModuleList([
                    GCNBlock(in_dim, hidden_dim, dropout, use_batchnorm=use_batchnorm),
                    GCNBlock(hidden_dim, hidden_dim, dropout, use_batchnorm=use_batchnorm),
                ])
                for _ in range(n_levels)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim * n_levels + n_covariates, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        batches: list[dict[str, torch.Tensor]],
        covariates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = []
        for L, b in enumerate(batches):
            x = b["x"]
            for layer in self.branches[L]:
                x = layer(x, b["edge_index"], b["edge_weight"])
            pooled.append(_pool_concat(x, b["batch"]))
        combined = torch.cat(pooled, dim=-1)
        if self.n_covariates > 0 and covariates is not None:
            combined = torch.cat([combined, covariates], dim=-1)
        return self.head(combined)


class HierarchicalGraphUNet(nn.Module):
    """Encoder + bottleneck + decoder over a predefined multi-resolution graph.

    Pooling between levels uses the precomputed `kept_orig` index map, so no
    learnable pooling parameters are needed (avoids the main instability of
    DiffPool/TopKPool at small N). Skip connections concatenate encoder
    features at each level with the corresponding decoder features.

    Forward expects `batches` of length n_levels (finest first, coarsest last)
    and `pool_maps` of length n_levels-1; pool_maps[L] is a (N_{L+1},)
    LongTensor of indices into the level-L flat tensor.

    The classification/regression head is applied to the global pool of the
    final L0 (finest) decoder output.
    """

    def __init__(
        self,
        in_dim: int,
        n_levels: int = 4,
        hidden_dim: int = 64,
        out_dim: int = 1,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
        n_covariates: int = 0,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_covariates = n_covariates
        self.enc = nn.ModuleList(
            [
                GCNBlock(
                    in_dim if L == 0 else hidden_dim,
                    hidden_dim, dropout, use_batchnorm=use_batchnorm,
                )
                for L in range(n_levels)
            ]
        )
        # Decoder takes (skip-feature || upsampled-feature) -> hidden
        self.dec = nn.ModuleList(
            [
                GCNBlock(2 * hidden_dim, hidden_dim, dropout, use_batchnorm=use_batchnorm)
                for _ in range(n_levels - 1)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim + n_covariates, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        batches: list[dict[str, torch.Tensor]],
        pool_maps: list[torch.Tensor],
        covariates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert len(batches) == self.n_levels
        assert len(pool_maps) == self.n_levels - 1

        # Encoder pass (record per-level features for skip).
        enc_feats: list[torch.Tensor] = []
        x = batches[0]["x"]
        for L in range(self.n_levels):
            x = self.enc[L](x, batches[L]["edge_index"], batches[L]["edge_weight"])
            enc_feats.append(x)
            if L < self.n_levels - 1:
                # Pool from level L to L+1 by indexing.
                x = x[pool_maps[L]]

        # Decoder: unpool then convolve, skip-concat at each level.
        for L in range(self.n_levels - 2, -1, -1):
            # Unpool from L+1 back to L (scatter to original positions, zero
            # elsewhere). pool_maps[L] are positions in the L-frame of L+1 nodes.
            n_L = enc_feats[L].shape[0]
            up = x.new_zeros(n_L, x.shape[-1])
            up[pool_maps[L]] = x
            skip = enc_feats[L]
            x = self.dec[L](
                torch.cat([up, skip], dim=-1),
                batches[L]["edge_index"],
                batches[L]["edge_weight"],
            )

        pooled = _pool_concat(x, batches[0]["batch"])
        if self.n_covariates > 0 and covariates is not None:
            pooled = torch.cat([pooled, covariates], dim=-1)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_model(
    name: str,
    in_dim: int,
    n_levels: int,
    out_dim: int,
    hidden_dim: int = 64,
    dropout: float = 0.3,
    use_batchnorm: bool = True,
    n_covariates: int = 0,
) -> nn.Module:
    name = name.lower()
    common = dict(hidden_dim=hidden_dim, out_dim=out_dim,
                  dropout=dropout, use_batchnorm=use_batchnorm,
                  n_covariates=n_covariates)
    if name in ("gcn", "single", "single_level_gcn"):
        return SingleLevelGCN(in_dim, **common)
    if name in ("multi", "multiscale", "multi_scale_concat"):
        return MultiScaleConcat(in_dim, n_levels, **common)
    if name in ("unet", "graph_unet", "hierarchical"):
        return HierarchicalGraphUNet(in_dim, n_levels, **common)
    raise ValueError(f"Unknown model name: {name}")
