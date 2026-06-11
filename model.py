import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax

from config import MODEL_CONFIG, GRAPH_CONFIG, ABLATION_CONFIG, FEATURE_CONFIG


def _as_long_tensor(x, device: torch.device) -> torch.Tensor:
    if x is None:
        return torch.zeros((0,), dtype=torch.long, device=device)
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.long)
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return torch.zeros((0,), dtype=torch.long, device=device)
        return torch.tensor(list(x), dtype=torch.long, device=device)
    return torch.zeros((0,), dtype=torch.long, device=device)

def _get_sample_pair_index(self, batch, key, sample_idx, device):
    val = batch.get(key, None)
    if val is None:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    pair_index = val[sample_idx]
    if pair_index is None:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    return pair_index.to(device=device, dtype=torch.long)

def _get_sample_pair_feat(self, batch, key, sample_idx, device):
    val = batch.get(key, None)
    if val is None:
        return torch.zeros((0, 5), dtype=torch.float32, device=device)
    feat = val[sample_idx]
    if feat is None:
        return torch.zeros((0, 5), dtype=torch.float32, device=device)
    return feat.to(device=device, dtype=torch.float32)

def _get_sample_long_idx(self, batch, key, sample_idx, device):
    val = batch.get(key, None)
    if val is None:
        return torch.zeros((0,), dtype=torch.long, device=device)
    idx = val[sample_idx]
    if idx is None:
        return torch.zeros((0,), dtype=torch.long, device=device)
    return idx.to(device=device, dtype=torch.long)

class GraphPooling(nn.Module):
    def __init__(self, in_channels: int, mode: str = "attention"):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode == "attention":
            self.gate = nn.Sequential(
                nn.Linear(in_channels, in_channels),
                nn.ReLU(),
                nn.Linear(in_channels, 1),
            )

    def forward(self, x: torch.Tensor, batch: torch.Tensor, size: Optional[int] = None) -> torch.Tensor:
        if size is None:
            size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        if x.numel() == 0:
            feat_dim = x.size(-1) if x.dim() == 2 else 0
            return torch.zeros((size, feat_dim), device=batch.device if batch.numel() > 0 else x.device)

        if self.mode == "mean":
            return global_mean_pool(x, batch, size=size)
        if self.mode == "sum":
            return global_add_pool(x, batch, size=size)
        if self.mode == "max":
            return global_max_pool(x, batch, size=size)
        if self.mode == "attention":
            alpha = softmax(self.gate(x).view(-1), batch, num_nodes=size)
            return global_add_pool(x * alpha.unsqueeze(-1), batch, size=size)
        raise ValueError(f"Unknown pooling mode: {self.mode}")


class GeometricEdgeBlock(MessagePassing):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float = 0.1):
        super().__init__(aggr="add", node_dim=0)
        geom_in_dim = 4  # unit vector (3) + distance (1)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim + geom_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.upd_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        pos: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return self.norm(x)
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, pos=pos)
        out = self.upd_mlp(torch.cat([x, out], dim=-1))
        out = self.norm(x + self.dropout(out))
        return out

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor,
        pos_i: torch.Tensor,
        pos_j: torch.Tensor,
    ) -> torch.Tensor:
        rel = pos_j - pos_i
        dist = torch.norm(rel, dim=-1, keepdim=True)
        unit = rel / (dist + 1e-8)
        geom = torch.cat([unit, dist], dim=-1)
        msg_in = torch.cat([x_i, x_j, edge_attr, geom], dim=-1)
        return self.msg_mlp(msg_in)


class JointGraphEncoder(nn.Module):
    def __init__(self, feature_dim: int, edge_dim: int, config: dict):
        super().__init__()
        self.hidden_dim = int(config["hidden_dim"])
        self.num_layers = int(config["num_gnn_layers"])
        self.dropout = float(config.get("dropout", 0.1))
        self.pooling_mode = str(config.get("pooling", "attention"))

        self.node_proj = nn.Sequential(
            nn.Linear(feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, edge_dim),
        )
        self.layers = nn.ModuleList([
            GeometricEdgeBlock(self.hidden_dim, edge_dim=edge_dim, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])
        self.pool = GraphPooling(self.hidden_dim, mode=self.pooling_mode)

    def forward(self, graph: Data) -> Dict[str, torch.Tensor]:
        x = getattr(graph, "x", None)
        if x is None or x.dim() != 2 or x.size(0) == 0:
            batch = getattr(graph, "batch", None)
            if batch is None or batch.numel() == 0:
                batch = torch.zeros((0,), dtype=torch.long, device=next(self.parameters()).device)
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else getattr(graph, "num_graphs", 1)
            hidden_dim = self.node_proj[-1].out_features
            return {
                "node_repr": torch.zeros((0, hidden_dim), dtype=torch.float32, device=next(self.parameters()).device),
                "batch": batch,
                "global_repr": torch.zeros((num_graphs, hidden_dim), dtype=torch.float32, device=next(self.parameters()).device),
            }

        edge_index = getattr(graph, "edge_index", None)
        if edge_index is None:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)

        edge_attr = getattr(graph, "edge_attr", None)
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), int(GRAPH_CONFIG.get("edge_feat_dim", 6))), dtype=torch.float32, device=x.device)

        pos = getattr(graph, "pos", None)
        if pos is None or pos.dim() != 2 or pos.size(0) != x.size(0):
            pos = torch.zeros((x.size(0), 3), dtype=torch.float32, device=x.device)

        batch = getattr(graph, "batch", None)
        if batch is None or batch.numel() == 0:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        for layer in self.layers:
            x = layer(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)

        global_repr = self.pool(x, batch, size=num_graphs)
        return {
            "node_repr": x,
            "batch": batch,
            "global_repr": global_repr,
        }


class IndexAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 or idx.numel() == 0:
            return torch.zeros((self.hidden_dim,), dtype=torch.float32, device=x.device)
        idx = idx.long().unique(sorted=True)
        idx = idx[(idx >= 0) & (idx < x.size(0))]
        if idx.numel() == 0:
            return torch.zeros((self.hidden_dim,), dtype=torch.float32, device=x.device)
        sub = x[idx]
        alpha = torch.softmax(self.gate(sub).view(-1), dim=0)
        return torch.sum(sub * alpha.unsqueeze(-1), dim=0)


class PairInteractionEncoder(nn.Module):
    def __init__(self, hidden_dim: int, pair_feat_dim: int = 5, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.pair_feat_dim = int(pair_feat_dim)
        in_dim = hidden_dim * 5 + self.pair_feat_dim
        self.pair_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        wt_x: torch.Tensor,
        mut_x: torch.Tensor,
        wt_idx: torch.Tensor,
        mut_idx: torch.Tensor,
        pair_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = wt_x.device
        if wt_idx.numel() == 0 or mut_idx.numel() == 0:
            return torch.zeros((self.hidden_dim,), dtype=torch.float32, device=device)

        wt_idx = wt_idx.long()
        mut_idx = mut_idx.long()
        valid = (
            (wt_idx >= 0) & (wt_idx < wt_x.size(0)) &
            (mut_idx >= 0) & (mut_idx < mut_x.size(0))
        )
        wt_idx = wt_idx[valid]
        mut_idx = mut_idx[valid]
        if wt_idx.numel() == 0:
            return torch.zeros((self.hidden_dim,), dtype=torch.float32, device=device)

        h_wt = wt_x[wt_idx]
        h_mut = mut_x[mut_idx]
        h_delta = h_mut - h_wt
        h_abs = torch.abs(h_delta)
        h_prod = h_mut * h_wt

        feat_dim = self.pair_feat_dim
        if pair_feat is None:
            pair_feat = torch.zeros((wt_idx.size(0), feat_dim), dtype=torch.float32, device=device)
        else:
            pair_feat = pair_feat.to(device=device, dtype=torch.float32)
            if pair_feat.dim() != 2:
                pair_feat = pair_feat.view(pair_feat.size(0), -1)
            if pair_feat.size(0) != wt_idx.size(0):
                n = min(pair_feat.size(0), wt_idx.size(0))
                pair_feat = pair_feat[:n]
                wt_idx = wt_idx[:n]
                mut_idx = mut_idx[:n]
                h_wt = h_wt[:n]
                h_mut = h_mut[:n]
                h_delta = h_delta[:n]
                h_abs = h_abs[:n]
                h_prod = h_prod[:n]
            if pair_feat.size(1) < feat_dim:
                pad = torch.zeros((pair_feat.size(0), feat_dim - pair_feat.size(1)), dtype=torch.float32, device=device)
                pair_feat = torch.cat([pair_feat, pad], dim=-1)
            elif pair_feat.size(1) > feat_dim:
                pair_feat = pair_feat[:, :feat_dim]

        pair_h = torch.cat([h_wt, h_mut, h_delta, h_abs, h_prod, pair_feat], dim=-1)
        pair_h = self.pair_mlp(pair_h)
        alpha = torch.softmax(self.gate(pair_h).view(-1), dim=0)
        return torch.sum(pair_h * alpha.unsqueeze(-1), dim=0)


class ContactDeltaEncoder(nn.Module):
    """显式 WT/MUT 抗体-抗原接触变化编码器。"""
    def __init__(self, hidden_dim: int, pair_feat_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pair_feat_dim = int(pair_feat_dim)
        in_dim = hidden_dim * 7 + self.pair_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        wt_x: torch.Tensor,
        mut_x: torch.Tensor,
        pair_index: Optional[torch.Tensor],
        pair_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = wt_x.device
        if pair_index is None or pair_index.numel() == 0:
            return torch.zeros((self.hidden_dim,), dtype=torch.float32, device=device)

        pair_index = pair_index.to(device=device, dtype=torch.long)
        if pair_index.dim() != 2 or pair_index.size(0) != 4:
            return torch.zeros((self.hidden_dim,), dtype=torch.float32, device=device)

        wt_ab_idx = pair_index[0]
        wt_ag_idx = pair_index[1]
        mut_ab_idx = pair_index[2]
        mut_ag_idx = pair_index[3]
        valid = (
            (wt_ab_idx >= 0) & (wt_ab_idx < wt_x.size(0)) &
            (wt_ag_idx >= 0) & (wt_ag_idx < wt_x.size(0)) &
            (mut_ab_idx >= 0) & (mut_ab_idx < mut_x.size(0)) &
            (mut_ag_idx >= 0) & (mut_ag_idx < mut_x.size(0))
        )
        if not valid.any():
            return torch.zeros((self.hidden_dim,), dtype=torch.float32, device=device)

        wt_ab_idx = wt_ab_idx[valid]
        wt_ag_idx = wt_ag_idx[valid]
        mut_ab_idx = mut_ab_idx[valid]
        mut_ag_idx = mut_ag_idx[valid]

        feat_dim = self.pair_feat_dim
        if pair_feat is None:
            pair_feat = torch.zeros((wt_ab_idx.size(0), feat_dim), dtype=torch.float32, device=device)
        else:
            pair_feat = pair_feat.to(device=device, dtype=torch.float32)
            if pair_feat.dim() != 2:
                pair_feat = pair_feat.view(pair_feat.size(0), -1)
            pair_feat = pair_feat[valid] if pair_feat.size(0) == valid.size(0) else pair_feat[:wt_ab_idx.size(0)]
            if pair_feat.size(0) != wt_ab_idx.size(0):
                n = min(pair_feat.size(0), wt_ab_idx.size(0))
                wt_ab_idx = wt_ab_idx[:n]
                wt_ag_idx = wt_ag_idx[:n]
                mut_ab_idx = mut_ab_idx[:n]
                mut_ag_idx = mut_ag_idx[:n]
                pair_feat = pair_feat[:n]
            if pair_feat.size(1) < feat_dim:
                pad = torch.zeros((pair_feat.size(0), feat_dim - pair_feat.size(1)), dtype=torch.float32, device=device)
                pair_feat = torch.cat([pair_feat, pad], dim=-1)
            elif pair_feat.size(1) > feat_dim:
                pair_feat = pair_feat[:, :feat_dim]

        if wt_ab_idx.numel() == 0:
            return torch.zeros((self.hidden_dim,), dtype=torch.float32, device=device)

        h_wt_ab = wt_x[wt_ab_idx]
        h_wt_ag = wt_x[wt_ag_idx]
        h_mut_ab = mut_x[mut_ab_idx]
        h_mut_ag = mut_x[mut_ag_idx]

        delta_ab = h_mut_ab - h_wt_ab
        delta_ag = h_mut_ag - h_wt_ag
        wt_pair = h_wt_ab * h_wt_ag
        mut_pair = h_mut_ab * h_mut_ag
        pair_delta = mut_pair - wt_pair

        h = torch.cat([
            h_wt_ab,
            h_wt_ag,
            h_mut_ab,
            h_mut_ag,
            delta_ab,
            delta_ag,
            pair_delta,
            pair_feat,
        ], dim=-1)
        h = self.mlp(h)
        alpha = torch.softmax(self.gate(h).view(-1), dim=0)
        return torch.sum(h * alpha.unsqueeze(-1), dim=0)


class DeltaRegressionHead(nn.Module):
    def __init__(self, input_dim: int, config: dict):
        super().__init__()
        hidden = int(config.get("delta_head_hidden_dim", 256))
        num_layers = int(config.get("delta_head_num_layers", 2))
        dropout = float(config.get("dropout", 0.1))
        out_dim = int(config.get("out_dim", 1))

        layers: List[nn.Module] = []
        cur = input_dim
        for _ in range(max(num_layers - 1, 1)):
            layers.extend([
                nn.Linear(cur, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            cur = hidden
        layers.append(nn.Linear(cur, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).view(-1)


class PLMOnlyBaseline(nn.Module):
    """
    PLM-only ablation baseline.

    This branch intentionally ignores GNN message passing, DSSP, amino-acid one-hot,
    physicochemical descriptors, edge features, interface/contact-delta encoders and
    pair encoders. It only reads the side-specific PLM slice from node features.

    Current node feature layout in pdb_graph.py:
      0:20  amino-acid one-hot
      20:23 physicochemical descriptors
      23    mutation flag
      24    antibody-side flag
      25    antigen-side flag
      26    interface flag
      27:31 DSSP
      31:159 side-specific PLM embedding
    """

    def __init__(self, config: dict = MODEL_CONFIG):
        super().__init__()
        self.plm_start_idx = int(config.get("plm_only_start_idx", 31))
        self.plm_dim = int(config.get("plm_only_dim", FEATURE_CONFIG.get("antiberty_dim", 128)))
        self.plm_end_idx = self.plm_start_idx + self.plm_dim
        self.summary_blocks = 5  # all_mean, all_max, mutation_mean, antibody_mean, antigen_mean
        dropout = float(config.get("dropout", 0.1))
        hidden = int(config.get("delta_head_hidden_dim", 256))
        in_dim = self.plm_dim * self.summary_blocks * 4  # WT, MUT, delta, abs(delta)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _num_graphs(graph: Data) -> int:
        batch = getattr(graph, "batch", None)
        if batch is not None and isinstance(batch, torch.Tensor) and batch.numel() > 0:
            return int(batch.max().item()) + 1
        if hasattr(graph, "num_graphs"):
            try:
                return int(graph.num_graphs)
            except Exception:
                pass
        return 1

    @staticmethod
    def _get_graph_ranges(graph: Data) -> List[Tuple[int, int]]:
        ptr = getattr(graph, "ptr", None)
        if ptr is not None and isinstance(ptr, torch.Tensor) and ptr.numel() >= 2:
            vals = ptr.detach().cpu().tolist()
            return [(int(vals[i]), int(vals[i + 1])) for i in range(len(vals) - 1)]
        n = int(getattr(graph, "num_nodes", 0))
        return [(0, n)]

    @staticmethod
    def _safe_mean(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros((x.size(-1),), dtype=torch.float32, device=x.device)
        return x.mean(dim=0)

    @staticmethod
    def _safe_max(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return torch.zeros((x.size(-1),), dtype=torch.float32, device=x.device)
        return x.max(dim=0).values

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 or mask.numel() == 0 or not bool(mask.any()):
            return torch.zeros((x.size(-1),), dtype=torch.float32, device=x.device)
        return x[mask].mean(dim=0)

    def _local_mut_mask(self, graph: Data, start: int, end: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros((max(end - start, 0),), dtype=torch.bool, device=device)
        idx = getattr(graph, "mut_idx", None)
        if idx is None:
            return mask
        if not isinstance(idx, torch.Tensor):
            try:
                idx = torch.tensor(idx, dtype=torch.long, device=device)
            except Exception:
                return mask
        idx = idx.to(device=device, dtype=torch.long)
        keep = idx[(idx >= start) & (idx < end)] - int(start)
        keep = keep[(keep >= 0) & (keep < mask.numel())]
        if keep.numel() > 0:
            mask[keep] = True
        return mask

    def _summary_for_sample(self, graph: Data, sample_idx: int) -> torch.Tensor:
        ranges = self._get_graph_ranges(graph)
        device = graph.x.device if hasattr(graph, "x") and isinstance(graph.x, torch.Tensor) else next(self.parameters()).device
        if sample_idx >= len(ranges) or not hasattr(graph, "x") or graph.x is None:
            return torch.zeros((self.plm_dim * self.summary_blocks,), dtype=torch.float32, device=device)

        start, end = ranges[sample_idx]
        x = graph.x[start:end]
        if x.numel() == 0 or x.dim() != 2 or x.size(1) < self.plm_end_idx:
            return torch.zeros((self.plm_dim * self.summary_blocks,), dtype=torch.float32, device=device)

        plm = x[:, self.plm_start_idx:self.plm_end_idx].float()
        mut_mask = self._local_mut_mask(graph, start, end, plm.device)
        ab_mask = x[:, 24].float() > 0.5 if x.size(1) > 24 else torch.zeros((x.size(0),), dtype=torch.bool, device=plm.device)
        ag_mask = x[:, 25].float() > 0.5 if x.size(1) > 25 else torch.zeros((x.size(0),), dtype=torch.bool, device=plm.device)

        all_mean = self._safe_mean(plm)
        all_max = self._safe_max(plm)
        mut_mean = self._masked_mean(plm, mut_mask)
        ab_mean = self._masked_mean(plm, ab_mask)
        ag_mean = self._masked_mean(plm, ag_mask)
        return torch.cat([all_mean, all_max, mut_mean, ab_mean, ag_mean], dim=-1)

    def forward(self, batch: Dict[str, Data]) -> torch.Tensor:
        if "wt_joint_graph" not in batch or "mut_joint_graph" not in batch:
            raise KeyError("Batch must contain 'wt_joint_graph' and 'mut_joint_graph'")

        wt_graph: Data = batch["wt_joint_graph"]
        mut_graph: Data = batch["mut_joint_graph"]
        num_graphs = min(self._num_graphs(wt_graph), self._num_graphs(mut_graph))
        if num_graphs <= 0:
            return torch.zeros((0,), dtype=torch.float32, device=next(self.parameters()).device)

        feats: List[torch.Tensor] = []
        for i in range(num_graphs):
            wt_s = self._summary_for_sample(wt_graph, i)
            mut_s = self._summary_for_sample(mut_graph, i)
            delta = mut_s - wt_s
            feats.append(torch.cat([wt_s, mut_s, delta, torch.abs(delta)], dim=-1))

        x = torch.stack(feats, dim=0)
        return self.mlp(x).view(-1)


class ddGModel(nn.Module):
    def __init__(
        self,
        feature_dim: Optional[int] = None,
        edge_dim: Optional[int] = None,
        config: dict = MODEL_CONFIG,
        use_local_subgraph: bool = True,
    ):
        super().__init__()
        self.use_local_subgraph = bool(use_local_subgraph)
        self.feature_dim = int(feature_dim) if feature_dim is not None else int(GRAPH_CONFIG.get("node_feat_dim", 159))
        self.edge_dim = int(edge_dim) if edge_dim is not None else int(GRAPH_CONFIG.get("edge_feat_dim", 6))
        self.hidden_dim = int(config.get("hidden_dim", 128))
        self.config = dict(config)

        # PLM-only baseline：直接从原始节点 PLM embedding 做 pooling + MLP，完全跳过 GNN/结构/界面分支。
        self.use_plm_only = bool(config.get("use_plm_only", ABLATION_CONFIG.get("use_plm_only", False)))
        self.plm_only_baseline = PLMOnlyBaseline(config=config) if self.use_plm_only else None

        # 消融开关：保持 head 输入维度不变；被消融的 readout / encoder 输出零向量。
        self.use_mutation_readout = bool(config.get(
            "use_mutation_readout",
            ABLATION_CONFIG.get("use_mutation_readout", True),
        ))
        self.use_interface_readout = bool(config.get(
            "use_interface_readout",
            ABLATION_CONFIG.get("use_interface_modeling", True),
        ))
        self.use_pair_encoder = bool(config.get(
            "use_pair_encoder",
            ABLATION_CONFIG.get("use_pair_encoder", True),
        ))

        self.encoder = JointGraphEncoder(
            feature_dim=self.feature_dim,
            edge_dim=self.edge_dim,
            config=config,
        )
        self.index_pool = IndexAttentionPool(self.hidden_dim, dropout=float(config.get("dropout", 0.1)))

        pair_feat_dim = int(config.get("pair_feat_dim", 5))
        self.pair_encoder = PairInteractionEncoder(
            self.hidden_dim,
            pair_feat_dim=pair_feat_dim,
            dropout=float(config.get("pair_dropout", config.get("dropout", 0.1))),
        )

        # 旧版 concat head：保留，方便在 config.py 中切回 "concat_head" 做对照实验。
        self.prediction_mode = str(config.get("prediction_mode", "concat_head")).lower()
        self.return_components = bool(config.get("return_components", False))
        self.use_local_mutation_effect = bool(config.get("use_local_mutation_effect", True))
        self.use_changed_interface_contact_effect = bool(config.get("use_changed_interface_contact_effect", True))
        self.use_global_calibration_bias = bool(config.get("use_global_calibration_bias", True))

        head_input_dim = self.hidden_dim * 10
        self.head = DeltaRegressionHead(head_input_dim, config=config)

        # interaction_delta_v0：不重建缓存的第一版三分量 additive head。
        # local: mutation_delta / abs(mutation_delta) / mutation_pair_repr
        # contact: interface_delta / abs(interface_delta) / interface_pair_repr
        # calibration: z_delta / abs(z_delta) / global_delta
        branch_input_dim = self.hidden_dim * 3
        self.local_effect_head = DeltaRegressionHead(branch_input_dim, config=config)
        self.contact_effect_head = DeltaRegressionHead(branch_input_dim, config=config)
        self.calibration_head = DeltaRegressionHead(branch_input_dim, config=config)

        # v1 contact-delta：显式抗体-抗原接触变化分支。
        self.use_contact_delta_encoder = bool(config.get("use_contact_delta_encoder", False))
        contact_pair_feat_dim = int(config.get("contact_pair_feat_dim", 16))
        self.contact_delta_encoder = ContactDeltaEncoder(
            hidden_dim=self.hidden_dim,
            pair_feat_dim=contact_pair_feat_dim,
            dropout=float(config.get("contact_delta_dropout", config.get("dropout", 0.1))),
        )
        self.contact_delta_effect_head = DeltaRegressionHead(self.hidden_dim, config=config)

        self.use_contact_delta_residual = bool(config.get("use_contact_delta_residual", False))
        self.contact_delta_residual_scale = float(config.get("contact_delta_residual_scale", 1.0))

    @staticmethod
    def _num_graphs(graph: Data) -> int:
        batch = getattr(graph, "batch", None)
        if batch is not None and batch.numel() > 0:
            return int(batch.max().item()) + 1
        if hasattr(graph, "num_graphs"):
            try:
                return int(graph.num_graphs)
            except Exception:
                pass
        return 1

    @staticmethod
    def _get_graph_ranges(graph: Data) -> List[Tuple[int, int]]:
        ptr = getattr(graph, "ptr", None)
        if ptr is not None and isinstance(ptr, torch.Tensor) and ptr.numel() >= 2:
            ptr = ptr.tolist()
            return [(int(ptr[i]), int(ptr[i + 1])) for i in range(len(ptr) - 1)]
        n = int(getattr(graph, "num_nodes", 0))
        return [(0, n)]

    @staticmethod
    def _slice_list_attr(values, start: int, end: int):
        if values is None:
            return []
        if isinstance(values, list):
            return values[start:end]
        try:
            return list(values)[start:end]
        except Exception:
            return []

    def _local_indices_for_sample(self, global_idx: Optional[torch.Tensor], start: int, end: int, device: torch.device) -> torch.Tensor:
        if global_idx is None:
            return torch.zeros((0,), dtype=torch.long, device=device)
        idx = _as_long_tensor(global_idx, device)
        if idx.numel() == 0:
            return idx
        mask = (idx >= start) & (idx < end)
        if not mask.any():
            return torch.zeros((0,), dtype=torch.long, device=device)
        return idx[mask] - int(start)

    def _normalize_residue_key_for_hash(self, k):
        """
        把 PyG Batch / torch.load 后可能出现的 residue key 统一转成可哈希的:
            (chain_id: str, resseq: int, icode: str)

        兼容:
            ('H', 100, '')
            ['H', 100, '']
            ['H', [100], '']
            ['H', tensor(100), '']
            ['H', tensor([100]), '']
        """

        def _scalar(x):
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu()
                if x.numel() == 0:
                    return ""
                if x.numel() == 1:
                    return x.view(-1)[0].item()
                return x.view(-1)[0].item()

            if isinstance(x, (list, tuple)):
                if len(x) == 0:
                    return ""
                return _scalar(x[0])

            return x

        if k is None:
            return None

        if isinstance(k, torch.Tensor):
            k = k.detach().cpu().tolist()

        if not isinstance(k, (list, tuple)):
            return None

        if len(k) < 3:
            return None

        chain = str(_scalar(k[0])).strip()
        resseq_raw = _scalar(k[1])
        icode = str(_scalar(k[2])).strip()

        if not chain:
            return None

        try:
            resseq = int(resseq_raw)
        except Exception:
            return None

        return (chain, resseq, icode)

    def _infer_aligned_pairs(
            self,
            wt_graph: Data,
            mut_graph: Data,
            sample_idx: int,
            device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        wt_ranges = self._get_graph_ranges(wt_graph)
        mut_ranges = self._get_graph_ranges(mut_graph)

        if sample_idx >= len(wt_ranges) or sample_idx >= len(mut_ranges):
            return (
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
            )

        wt_start, wt_end = wt_ranges[sample_idx]
        mut_start, mut_end = mut_ranges[sample_idx]

        wt_keys = self._slice_list_attr(
            getattr(wt_graph, "residue_keys", None),
            wt_start,
            wt_end,
        )
        mut_keys = self._slice_list_attr(
            getattr(mut_graph, "residue_keys", None),
            mut_start,
            mut_end,
        )

        if len(wt_keys) == 0 or len(mut_keys) == 0:
            return (
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
            )

        wt_map = {}
        for i, k in enumerate(wt_keys):
            nk = self._normalize_residue_key_for_hash(k)
            if nk is not None:
                wt_map[nk] = i

        mut_map = {}
        for i, k in enumerate(mut_keys):
            nk = self._normalize_residue_key_for_hash(k)
            if nk is not None:
                mut_map[nk] = i

        if len(wt_map) == 0 or len(mut_map) == 0:
            return (
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
            )

        common = [k for k in wt_map.keys() if k in mut_map]

        if len(common) == 0:
            return (
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
            )

        wt_idx = torch.tensor(
            [wt_map[k] for k in common],
            dtype=torch.long,
            device=device,
        )
        mut_idx = torch.tensor(
            [mut_map[k] for k in common],
            dtype=torch.long,
            device=device,
        )

        return wt_idx, mut_idx


    def _expand_shell(self, pos: torch.Tensor, center_idx: torch.Tensor, radius: float, topk: int) -> torch.Tensor:
        device = pos.device
        if pos.numel() == 0 or center_idx.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=device)
        center_idx = center_idx.long().unique(sorted=True)
        center_idx = center_idx[(center_idx >= 0) & (center_idx < pos.size(0))]
        if center_idx.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=device)
        center_pos = pos[center_idx]
        dmat = torch.cdist(pos, center_pos)
        min_dist = dmat.min(dim=1).values
        keep = torch.where(min_dist <= float(radius))[0]
        if keep.numel() == 0 and int(topk) > 0:
            k = min(int(topk), pos.size(0))
            keep = torch.topk(min_dist, k=k, largest=False).indices.sort().values
        return keep.long().unique(sorted=True)

    def _aligned_subset(self, aligned_wt_idx: torch.Tensor, aligned_mut_idx: torch.Tensor, allowed_wt: torch.Tensor, allowed_mut: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if aligned_wt_idx.numel() == 0 or aligned_mut_idx.numel() == 0:
            dev = aligned_wt_idx.device
            return torch.zeros((0,), dtype=torch.long, device=dev), torch.zeros((0,), dtype=torch.long, device=dev)
        aw = set(allowed_wt.tolist())
        am = set(allowed_mut.tolist())
        keep = [i for i, (w, m) in enumerate(zip(aligned_wt_idx.tolist(), aligned_mut_idx.tolist())) if w in aw and m in am]
        if len(keep) == 0:
            dev = aligned_wt_idx.device
            return torch.zeros((0,), dtype=torch.long, device=dev), torch.zeros((0,), dtype=torch.long, device=dev)
        keep_t = torch.tensor(keep, dtype=torch.long, device=aligned_wt_idx.device)
        return aligned_wt_idx[keep_t], aligned_mut_idx[keep_t]


    def _get_sample_tensor_from_batch(self, batch: Dict[str, Data], key: str, sample_idx: int, device: torch.device, default_shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        val = batch.get(key, None)
        if val is None:
            return torch.zeros(default_shape, dtype=dtype, device=device)
        try:
            obj = val[sample_idx]
        except Exception:
            obj = None
        if obj is None:
            return torch.zeros(default_shape, dtype=dtype, device=device)
        if not isinstance(obj, torch.Tensor):
            try:
                obj = torch.tensor(obj, dtype=dtype)
            except Exception:
                return torch.zeros(default_shape, dtype=dtype, device=device)
        return obj.to(device=device, dtype=dtype)

    def _build_pair_feat(
        self,
        wt_graph: Data,
        mut_graph: Data,
        sample_idx: int,
        wt_idx: torch.Tensor,
        mut_idx: torch.Tensor,
        local_wt_mut_idx: torch.Tensor,
        local_mut_mut_idx: torch.Tensor,
        local_wt_interface_idx: torch.Tensor,
        local_mut_interface_idx: torch.Tensor,
    ) -> torch.Tensor:
        if wt_idx.numel() == 0:
            return torch.zeros((0, 5), dtype=torch.float32, device=wt_idx.device)

        wt_start, wt_end = self._get_graph_ranges(wt_graph)[sample_idx]
        mut_start, mut_end = self._get_graph_ranges(mut_graph)[sample_idx]
        wt_pos = wt_graph.pos[wt_start:wt_end]
        mut_pos = mut_graph.pos[mut_start:mut_end]

        d = torch.norm(mut_pos[mut_idx] - wt_pos[wt_idx], dim=-1, keepdim=True)
        inv_d = 1.0 / (1.0 + d)

        wt_mut_set = set(local_wt_mut_idx.tolist())
        mut_mut_set = set(local_mut_mut_idx.tolist())
        wt_int_set = set(local_wt_interface_idx.tolist())
        mut_int_set = set(local_mut_interface_idx.tolist())

        is_mut_pair = torch.tensor(
            [[1.0] if (int(w) in wt_mut_set or int(m) in mut_mut_set) else [0.0] for w, m in zip(wt_idx.tolist(), mut_idx.tolist())],
            dtype=torch.float32,
            device=wt_idx.device,
        )
        is_interface_pair = torch.tensor(
            [[1.0] if (int(w) in wt_int_set or int(m) in mut_int_set) else [0.0] for w, m in zip(wt_idx.tolist(), mut_idx.tolist())],
            dtype=torch.float32,
            device=wt_idx.device,
        )

        wt_side = self._slice_list_attr(getattr(wt_graph, "partner_side", None), wt_start, wt_end)
        mut_side = self._slice_list_attr(getattr(mut_graph, "partner_side", None), mut_start, mut_end)
        same_side = torch.tensor(
            [[1.0] if (wt_side[int(w)] == mut_side[int(m)]) else [0.0] for w, m in zip(wt_idx.tolist(), mut_idx.tolist())],
            dtype=torch.float32,
            device=wt_idx.device,
        )
        cross_side = 1.0 - same_side
        return torch.cat([d, inv_d, is_mut_pair, is_interface_pair, cross_side], dim=-1)

    def _pool_delta(self, wt_x: torch.Tensor, mut_x: torch.Tensor, wt_idx: torch.Tensor, mut_idx: torch.Tensor) -> torch.Tensor:
        if wt_idx.numel() == 0 or mut_idx.numel() == 0:
            return torch.zeros((self.hidden_dim,), dtype=torch.float32, device=wt_x.device)
        delta = mut_x[mut_idx] - wt_x[wt_idx]
        idx = torch.arange(delta.size(0), dtype=torch.long, device=delta.device)
        return self.index_pool(delta, idx)

    def forward(self, batch: Dict[str, Data]) -> torch.Tensor:
        if "wt_joint_graph" not in batch or "mut_joint_graph" not in batch:
            raise KeyError("Batch must contain 'wt_joint_graph' and 'mut_joint_graph'")

        if self.use_plm_only:
            pred = self.plm_only_baseline(batch)
            if self.return_components:
                return {
                    "pred": pred,
                    "plm_only_effect": pred,
                }
            return pred

        wt_graph: Data = batch["wt_joint_graph"]
        mut_graph: Data = batch["mut_joint_graph"]

        wt_out = self.encoder(wt_graph)
        mut_out = self.encoder(mut_graph)

        wt_x = wt_out["node_repr"]
        mut_x = mut_out["node_repr"]
        wt_global = wt_out["global_repr"]
        mut_global = mut_out["global_repr"]

        num_graphs = min(self._num_graphs(wt_graph), self._num_graphs(mut_graph))
        radius_mut = float(GRAPH_CONFIG.get("local_radius", 12.0))
        radius_intf = float(GRAPH_CONFIG.get("interface_edge_threshold", 12.0))
        topk = int(GRAPH_CONFIG.get("local_topk_fallback", 20))

        outputs: List[torch.Tensor] = []
        local_terms: List[torch.Tensor] = []
        contact_terms: List[torch.Tensor] = []
        contact_delta_terms: List[torch.Tensor] = []
        calibration_terms: List[torch.Tensor] = []
        wt_ranges = self._get_graph_ranges(wt_graph)
        mut_ranges = self._get_graph_ranges(mut_graph)

        for i in range(num_graphs):
            wt_start, wt_end = wt_ranges[i]
            mut_start, mut_end = mut_ranges[i]
            wt_local_x = wt_x[wt_start:wt_end]
            mut_local_x = mut_x[mut_start:mut_end]
            wt_local_pos = wt_graph.pos[wt_start:wt_end] if getattr(wt_graph, "pos", None) is not None else torch.zeros((wt_local_x.size(0), 3), device=wt_local_x.device)
            mut_local_pos = mut_graph.pos[mut_start:mut_end] if getattr(mut_graph, "pos", None) is not None else torch.zeros((mut_local_x.size(0), 3), device=mut_local_x.device)

            wt_mut_idx = self._local_indices_for_sample(getattr(wt_graph, "mut_idx", None), wt_start, wt_end, wt_local_x.device)
            mut_mut_idx = self._local_indices_for_sample(getattr(mut_graph, "mut_idx", None), mut_start, mut_end, mut_local_x.device)
            wt_interface_idx = self._local_indices_for_sample(getattr(wt_graph, "interface_idx", None), wt_start, wt_end, wt_local_x.device)
            mut_interface_idx = self._local_indices_for_sample(getattr(mut_graph, "interface_idx", None), mut_start, mut_end, mut_local_x.device)

            aligned_wt_idx, aligned_mut_idx = self._infer_aligned_pairs(wt_graph, mut_graph, i, wt_local_x.device)

            wt_mut_shell = self._expand_shell(wt_local_pos, wt_mut_idx, radius_mut, topk)
            mut_mut_shell = self._expand_shell(mut_local_pos, mut_mut_idx, radius_mut, topk)
            wt_intf_shell = self._expand_shell(wt_local_pos, wt_interface_idx, radius_intf, topk)
            mut_intf_shell = self._expand_shell(mut_local_pos, mut_interface_idx, radius_intf, topk)

            mut_shell_wt_idx, mut_shell_mut_idx = self._aligned_subset(aligned_wt_idx, aligned_mut_idx, wt_mut_shell, mut_mut_shell)
            intf_shell_wt_idx, intf_shell_mut_idx = self._aligned_subset(aligned_wt_idx, aligned_mut_idx, wt_intf_shell, mut_intf_shell)

            global_delta = self._pool_delta(wt_local_x, mut_local_x, aligned_wt_idx, aligned_mut_idx)
            if self.use_mutation_readout:
                mutation_delta = self._pool_delta(wt_local_x, mut_local_x, mut_shell_wt_idx, mut_shell_mut_idx)
            else:
                mutation_delta = torch.zeros((self.hidden_dim,), dtype=torch.float32, device=wt_local_x.device)

            if self.use_interface_readout:
                interface_delta = self._pool_delta(wt_local_x, mut_local_x, intf_shell_wt_idx, intf_shell_mut_idx)
            else:
                interface_delta = torch.zeros((self.hidden_dim,), dtype=torch.float32, device=wt_local_x.device)

            pair_feat_mut = self._build_pair_feat(
                wt_graph, mut_graph, i,
                mut_shell_wt_idx, mut_shell_mut_idx,
                wt_mut_idx, mut_mut_idx,
                wt_interface_idx, mut_interface_idx,
            )
            pair_feat_intf = self._build_pair_feat(
                wt_graph, mut_graph, i,
                intf_shell_wt_idx, intf_shell_mut_idx,
                wt_mut_idx, mut_mut_idx,
                wt_interface_idx, mut_interface_idx,
            )
            if self.use_pair_encoder and self.use_mutation_readout:
                mutation_pair_repr = self.pair_encoder(
                    wt_local_x,
                    mut_local_x,
                    mut_shell_wt_idx,
                    mut_shell_mut_idx,
                    pair_feat_mut,
                )
            else:
                mutation_pair_repr = torch.zeros((self.hidden_dim,), dtype=torch.float32, device=wt_local_x.device)

            if self.use_pair_encoder and self.use_interface_readout:
                interface_pair_repr = self.pair_encoder(
                    wt_local_x,
                    mut_local_x,
                    intf_shell_wt_idx,
                    intf_shell_mut_idx,
                    pair_feat_intf,
                )
            else:
                interface_pair_repr = torch.zeros((self.hidden_dim,), dtype=torch.float32, device=wt_local_x.device)

            z_wt = wt_global[i]
            z_mut = mut_global[i]
            z_delta = z_mut - z_wt
            z_abs = torch.abs(z_delta)

            concat_feat = torch.cat([
                z_wt,
                z_mut,
                z_delta,
                z_abs,
                global_delta,
                mutation_delta,
                interface_delta,
                mutation_pair_repr,
                interface_pair_repr,
                mutation_pair_repr - interface_pair_repr,
            ], dim=-1)

            if self.prediction_mode in {"interaction_delta", "interaction_delta_v0", "interaction_delta_v1_contact", "additive_delta"}:
                zero = torch.zeros((), dtype=torch.float32, device=wt_local_x.device)

                local_feat = torch.cat([
                    mutation_delta,
                    torch.abs(mutation_delta),
                    mutation_pair_repr,
                ], dim=-1).unsqueeze(0)
                # ---------------------------------------------------------
                # V1.3 contact branch:
                #   old_contact_effect = Head([interface_delta, abs(interface_delta), interface_pair_repr])
                #   contact_delta_effect = Head(ContactDeltaEncoder(...))
                #   contact_effect = old_contact_effect + 0.15 * contact_delta_effect
                #   train.py 对 raw contact_delta_effect 额外施加 L2 约束。
                # ---------------------------------------------------------

                if self.use_changed_interface_contact_effect and self.use_interface_readout:
                    contact_feat = torch.cat([
                        interface_delta,
                        torch.abs(interface_delta),
                        interface_pair_repr,
                    ], dim=-1).unsqueeze(0)

                    old_contact_effect = self.contact_effect_head(contact_feat).view(())
                else:
                    old_contact_effect = zero

                contact_delta_effect = zero

                if (
                        self.prediction_mode == "interaction_delta_v1_contact"
                        and self.use_contact_delta_encoder
                        and self.use_changed_interface_contact_effect
                        and self.use_interface_readout
                ):
                    contact_pair_index = self._get_sample_tensor_from_batch(
                        batch,
                        "interface_contact_pair_index",
                        i,
                        wt_local_x.device,
                        default_shape=(4, 0),
                        dtype=torch.long,
                    )
                    contact_pair_feat = self._get_sample_tensor_from_batch(
                        batch,
                        "interface_contact_pair_feat",
                        i,
                        wt_local_x.device,
                        default_shape=(0, int(self.config.get("contact_pair_feat_dim", 16))),
                        dtype=torch.float32,
                    )

                    contact_delta_repr = self.contact_delta_encoder(
                        wt_x=wt_local_x,
                        mut_x=mut_local_x,
                        pair_index=contact_pair_index,
                        pair_feat=contact_pair_feat,
                    )

                    contact_delta_effect = self.contact_delta_effect_head(
                        contact_delta_repr.unsqueeze(0)
                    ).view(())

                if (
                        self.prediction_mode == "interaction_delta_v1_contact"
                        and self.use_contact_delta_encoder
                        and self.use_contact_delta_residual
                ):
                    contact_effect = old_contact_effect + self.contact_delta_residual_scale * contact_delta_effect
                elif (
                        self.prediction_mode == "interaction_delta_v1_contact"
                        and self.use_contact_delta_encoder
                ):
                    # 保留原 V1 行为，方便 ablation：
                    # contact_effect = contact_delta_effect
                    contact_effect = contact_delta_effect
                else:
                    # V0 / exp4 行为
                    contact_effect = old_contact_effect

                calibration_feat = torch.cat([
                    z_delta,
                    z_abs,
                    global_delta,
                ], dim=-1).unsqueeze(0)

                local_effect = self.local_effect_head(local_feat).view(()) if self.use_local_mutation_effect else zero
                calibration_bias = self.calibration_head(calibration_feat).view(()) if self.use_global_calibration_bias else zero
                pred = local_effect + contact_effect + calibration_bias

                outputs.append(pred)
                if self.return_components:
                    local_terms.append(local_effect)
                    contact_terms.append(contact_effect)
                    contact_delta_terms.append(contact_delta_effect)
                    calibration_terms.append(calibration_bias)
            else:
                outputs.append(concat_feat)

        if len(outputs) == 0:
            empty = torch.zeros((0,), dtype=torch.float32, device=next(self.parameters()).device)
            if self.return_components and self.prediction_mode in {"interaction_delta", "interaction_delta_v0", "interaction_delta_v1_contact", "additive_delta"}:
                return {
                    "pred": empty,
                    "local_mutation_effect": empty,
                    "changed_interface_contact_effect": empty,
                    "contact_delta_effect": empty,
                    "global_calibration_bias": empty,
                }
            return empty

        if self.prediction_mode in {"interaction_delta", "interaction_delta_v0", "interaction_delta_v1_contact", "additive_delta"}:
            pred_out = torch.stack(outputs, dim=0).view(-1)
            if self.return_components:
                return {
                    "pred": pred_out,
                    "local_mutation_effect": torch.stack(local_terms, dim=0).view(-1),
                    "changed_interface_contact_effect": torch.stack(contact_terms, dim=0).view(-1),
                    "contact_delta_effect": torch.stack(contact_delta_terms, dim=0).view(-1),
                    "global_calibration_bias": torch.stack(calibration_terms, dim=0).view(-1),
                }
            return pred_out

        out = torch.stack(outputs, dim=0)
        return self.head(out)
