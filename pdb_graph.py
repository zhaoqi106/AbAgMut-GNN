import logging
from typing import List, Dict, Tuple, Any, Optional, Set

import torch
from Bio.PDB import PDBParser
from torch_geometric.data import Data

from config import GRAPH_CONFIG, FEATURE_CONFIG, DSSP_CONFIG,ABLATION_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================================================
# 节点特征:
# 20维 one-hot + 3维理化 + 1维突变标志 +
# 1维 is_antibody + 1维 is_antigen + 1维 is_interface +
# 4维DSSP + 128维 side-specific PLM
# =========================================================
class ResidueFeaturizer:
    AA_PROPS = {
        "A": [0.62, -0.5, 1.0],
        "R": [-2.53, 1.0, 0.0],
        "N": [-0.78, 0.0, 0.0],
        "D": [-0.90, -1.0, 0.0],
        "C": [0.29, 0.0, 1.0],
        "E": [-0.74, -1.0, 0.0],
        "Q": [-0.85, 0.0, 0.0],
        "G": [0.48, 0.0, 0.0],
        "H": [-0.40, 0.5, 0.0],
        "I": [1.38, 0.0, 0.0],
        "L": [1.06, 0.0, 0.0],
        "K": [-1.50, 1.0, 0.0],
        "M": [0.64, 0.0, 0.0],
        "F": [1.19, 0.0, 0.0],
        "P": [0.12, 0.0, 0.0],
        "S": [-0.18, 0.0, 0.0],
        "T": [-0.05, 0.0, 0.0],
        "W": [0.81, 0.0, 0.0],
        "Y": [0.26, 0.0, 0.0],
        "V": [1.08, 0.0, 0.0],
        "X": [0.0, 0.0, 0.0],
    }

    AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

    def __init__(self):
        ab_dim = int(FEATURE_CONFIG.get("antiberty_dim", 128))
        ag_dim = int(FEATURE_CONFIG.get("esm2_dim", 128))
        if ab_dim != ag_dim:
            raise ValueError(f"antiberty_dim ({ab_dim}) != esm2_dim ({ag_dim}), current model expects same PLM dim")
        self.plm_dim = ab_dim

    @property
    def feature_dim(self) -> int:
        return 20 + 3 + 1 + 1 + 1 + 1 + 4 + self.plm_dim

    def residue_to_aa(self, resname: str) -> str:
        mapping = {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
            "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
            "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
            "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
        }
        return mapping.get(str(resname).upper(), "X")

    def __call__(
        self,
        aa: str,
        is_mutation: bool = False,
        is_antibody: bool = False,
        is_antigen: bool = False,
        is_interface: bool = False,
        dssp_feat: Optional[torch.Tensor] = None,
        plm_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        one_hot = torch.zeros(len(self.AA_LIST), dtype=torch.float32)
        if aa in self.AA_LIST:
            one_hot[self.AA_LIST.index(aa)] = 1.0

        phys = torch.tensor(self.AA_PROPS.get(aa, [0.0, 0.0, 0.0]), dtype=torch.float32)
        mut_flag = torch.tensor([1.0 if is_mutation else 0.0], dtype=torch.float32)
        ab_flag = torch.tensor([1.0 if is_antibody else 0.0], dtype=torch.float32)
        ag_flag = torch.tensor([1.0 if is_antigen else 0.0], dtype=torch.float32)
        intf_flag = torch.tensor([1.0 if is_interface else 0.0], dtype=torch.float32)

        if dssp_feat is None:
            dssp_feat = torch.zeros(4, dtype=torch.float32)
        else:
            dssp_feat = dssp_feat.float()

        if plm_emb is None:
            plm_emb = torch.zeros(self.plm_dim, dtype=torch.float32)
        else:
            plm_emb = plm_emb.float()
            if plm_emb.numel() != self.plm_dim:
                if plm_emb.numel() > self.plm_dim:
                    plm_emb = plm_emb[:self.plm_dim]
                else:
                    pad = torch.zeros(self.plm_dim - plm_emb.numel(), dtype=torch.float32)
                    plm_emb = torch.cat([plm_emb, pad], dim=0)

        feat = torch.cat([
            one_hot, phys, mut_flag, ab_flag, ag_flag, intf_flag, dssp_feat, plm_emb
        ], dim=0)
        return feat


class ComplexGraphBuilder:
    def __init__(
            self,
            contact_threshold: float = 8.0,
            interface_threshold: float = 12.0,
            add_sequential_edges: bool = True,
            add_intra_spatial_edges: bool = True,
            add_inter_partner_edges: bool = True,
            add_mutation_flag: bool = True,
            add_side_flags: bool = True,
            add_interface_flag: bool = True,
    ):
        self.contact_threshold = float(contact_threshold)
        self.interface_edge_threshold = float(interface_threshold)

        self.add_sequential_edges = bool(add_sequential_edges)
        self.add_intra_spatial_edges = bool(add_intra_spatial_edges)
        self.add_inter_partner_edges = bool(add_inter_partner_edges)

        self.add_mutation_flag = bool(add_mutation_flag)
        self.add_side_flags = bool(add_side_flags)
        self.add_interface_flag = bool(add_interface_flag)

        self.featurizer = ResidueFeaturizer()
        self.pdb_parser = PDBParser(QUIET=True)
        self.edge_feat_dim = int(GRAPH_CONFIG.get("edge_feat_dim", 6))

    @staticmethod
    def _norm_icode(icode: Any) -> str:
        if icode is None:
            return ""
        s = str(icode).strip()
        return s if s else ""

    @staticmethod
    def _mutation_key(mut: Dict[str, Any]) -> Tuple[str, int, str]:
        return (
            str(mut["chain"]),
            int(mut["resseq"]),
            ComplexGraphBuilder._norm_icode(mut.get("icode", "")),
        )

    def _empty_graph(self) -> Data:
        g = Data(
            x=torch.zeros((0, self.featurizer.feature_dim), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.zeros((0, self.edge_feat_dim), dtype=torch.float32),
        )
        g.pos = torch.zeros((0, 3), dtype=torch.float32)
        g.seq_idx = torch.zeros((0,), dtype=torch.long)
        g.residue_ids = []
        g.residue_keys = []
        g.chain_ids = []
        g.partner_side = []
        g.interface_mask = torch.zeros((0,), dtype=torch.bool)
        g.mut_idx = torch.tensor([], dtype=torch.long)
        g.interface_idx = torch.tensor([], dtype=torch.long)
        return g

    def _load_chain_residues(self, structure, chain_id: str, side_label: str) -> List[Dict[str, Any]]:
        model = next(structure.get_models())
        if chain_id not in model:
            raise KeyError(f"Chain '{chain_id}' not found in structure")

        chain = model[chain_id]
        residues: List[Dict[str, Any]] = []
        seq_pos = 0
        for res in chain:
            if res.id[0] != " " or "CA" not in res:
                continue
            aa = self.featurizer.residue_to_aa(res.get_resname())
            resseq = int(res.id[1])
            icode = self._norm_icode(res.id[2] if len(res.id) > 2 else "")
            residues.append({
                "chain_id": chain.id,
                "resseq": resseq,
                "icode": icode,
                "resname": res.get_resname(),
                "aa": aa,
                "coord": torch.tensor(res["CA"].get_coord(), dtype=torch.float32),
                "seq_pos": seq_pos,
                "side": side_label,
            })
            seq_pos += 1
        return residues

    def _detect_interface_keys(
        self,
        ab_residues: List[Dict[str, Any]],
        ag_residues: List[Dict[str, Any]],
    ) -> Set[Tuple[str, int, str]]:
        out: Set[Tuple[str, int, str]] = set()
        if len(ab_residues) == 0 or len(ag_residues) == 0:
            return out

        ab_coords = torch.stack([r["coord"] for r in ab_residues], dim=0)
        ag_coords = torch.stack([r["coord"] for r in ag_residues], dim=0)
        dist = torch.cdist(ab_coords, ag_coords)
        src, dst = torch.where(dist <= self.interface_edge_threshold)
        for i, j in zip(src.tolist(), dst.tolist()):
            a = ab_residues[i]
            b = ag_residues[j]
            out.add((a["chain_id"], a["resseq"], a["icode"]))
            out.add((b["chain_id"], b["resseq"], b["icode"]))
        return out

    def _build_plm_for_residue(
            self,
            res: Dict[str, Any],
            antiberty_embeddings: Optional[Dict[str, Any]],
            esm2_embeddings: Optional[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        chain_id = res["chain_id"]
        seq_pos = int(res["seq_pos"])
        key = (res["chain_id"], int(res["resseq"]), self._norm_icode(res.get("icode", "")))

        if res["side"] == "ab":
            emb_pack = antiberty_embeddings or {}
        else:
            emb_pack = esm2_embeddings or {}

        # 新格式：优先精确按 residue key 对齐
        residue_embeddings = emb_pack.get("residue_embeddings", {})
        if key in residue_embeddings:
            emb = residue_embeddings[key]
            if isinstance(emb, torch.Tensor) and emb.dim() == 1:
                return emb.float()

        # 兼容旧格式：退回 chain embedding + seq_pos
        chain_embeddings = emb_pack.get("chain_embeddings", {})
        emb_full = chain_embeddings.get(chain_id, None)
        if isinstance(emb_full, torch.Tensor) and emb_full.dim() == 2 and seq_pos < emb_full.size(0):
            return emb_full[seq_pos].float()

        return None

    def _build_edges(
        self,
        coords: torch.Tensor,
        chain_ids: List[str],
        seq_pos_list: List[int],
        partner_side: List[str],
        interface_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = coords.size(0)
        if num_nodes == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, self.edge_feat_dim), dtype=torch.float32),
            )

        edge_list: List[Tuple[int, int]] = []
        edge_attr_list: List[torch.Tensor] = []
        seen = set()

        dist = torch.cdist(coords, coords)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                same_chain = chain_ids[i] == chain_ids[j]
                same_partner = partner_side[i] == partner_side[j]
                d = float(dist[i, j].item())
                is_seq = same_chain and abs(seq_pos_list[i] - seq_pos_list[j]) == 1
                use_edge = False

                use_interface = bool(ABLATION_CONFIG.get("use_interface_modeling", True))

                if self.add_intra_spatial_edges and same_partner and d <= self.contact_threshold:
                    use_edge = True
                elif (
                    use_interface
                    and self.add_inter_partner_edges
                    and (not same_partner)
                    and d <= self.interface_edge_threshold
                ):
                    use_edge = True
                elif self.add_sequential_edges and is_seq:
                    use_edge = True

                if not use_edge:
                    continue
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                edge_list.append((i, j))

                seq_dist_norm = abs(seq_pos_list[i] - seq_pos_list[j]) / max(num_nodes - 1, 1)
                ca_dist_norm = min(d / max(self.interface_edge_threshold, 1e-6), 1.0)
                is_seq_f = 1.0 if is_seq else 0.0
                is_same_chain_f = 1.0 if same_chain else 0.0
                is_cross_partner_f = 0.0 if same_partner else 1.0
                is_interface_edge_f = 1.0 if ((not same_partner) and (d <= self.interface_edge_threshold)) else 0.0

                edge_attr_list.append(torch.tensor([
                    seq_dist_norm,
                    ca_dist_norm,
                    is_seq_f,
                    is_same_chain_f,
                    is_cross_partner_f,
                    is_interface_edge_f,
                ], dtype=torch.float32))

        if len(edge_list) == 0:
            return (
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros((0, self.edge_feat_dim), dtype=torch.float32),
            )

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attr_list, dim=0)
        return edge_index, edge_attr

    def build_joint_graph(
        self,
        pdb_path: str,
        ab_chains: List[str],
        ag_chains: List[str],
        dssp_feats: Optional[Dict[Tuple[str, int, str], torch.Tensor]] = None,
        antiberty_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        esm2_embeddings: Optional[Dict[str, torch.Tensor]] = None,
        mutations: Optional[List[Dict[str, Any]]] = None,
    ) -> Data:
        structure = self.pdb_parser.get_structure("complex", pdb_path)
        ab_residues: List[Dict[str, Any]] = []
        ag_residues: List[Dict[str, Any]] = []

        for ch in ab_chains or []:
            ab_residues.extend(self._load_chain_residues(structure, ch, "ab"))
        for ch in ag_chains or []:
            ag_residues.extend(self._load_chain_residues(structure, ch, "ag"))

        residues = ab_residues + ag_residues
        if len(residues) == 0:
            return self._empty_graph()

        if bool(ABLATION_CONFIG.get("use_interface_modeling", True)):
            interface_key_set = self._detect_interface_keys(ab_residues, ag_residues)
        else:
            interface_key_set = set()
        mut_key_set: Set[Tuple[str, int, str]] = set()
        if mutations:
            mut_key_set = {self._mutation_key(m) for m in mutations}

        x_list = []
        coords = []
        chain_ids = []
        residue_ids = []
        seq_pos_list = []
        residue_keys = []
        partner_side = []
        interface_mask_list = []

        for res in residues:
            key = (res["chain_id"], res["resseq"], res["icode"])
            is_mutation = key in mut_key_set
            is_interface = key in interface_key_set
            dssp_feat = dssp_feats.get(key) if dssp_feats else None
            plm_emb = self._build_plm_for_residue(res, antiberty_embeddings, esm2_embeddings)

            x_list.append(
                self.featurizer(
                    aa=res["aa"],
                    is_mutation=is_mutation,
                    is_antibody=(res["side"] == "ab"),
                    is_antigen=(res["side"] == "ag"),
                    is_interface=is_interface,
                    dssp_feat=dssp_feat,
                    plm_emb=plm_emb,
                )
            )
            coords.append(res["coord"])
            chain_ids.append(res["chain_id"])
            residue_ids.append(res["resseq"])
            seq_pos_list.append(int(res["seq_pos"]))
            residue_keys.append(key)
            partner_side.append(res["side"])
            interface_mask_list.append(is_interface)

        x = torch.stack(x_list, dim=0)
        coords_t = torch.stack(coords, dim=0)
        interface_mask = torch.tensor(interface_mask_list, dtype=torch.bool)
        edge_index, edge_attr = self._build_edges(
            coords_t,
            chain_ids,
            seq_pos_list,
            partner_side,
            interface_mask,
        )
        g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        g.pos = coords_t
        g.seq_idx = torch.arange(x.size(0), dtype=torch.long)
        g.residue_ids = residue_ids
        g.residue_keys = residue_keys
        g.chain_ids = chain_ids
        g.partner_side = partner_side
        g.interface_mask = interface_mask
        g.mut_idx = self.map_mutations_to_graph(g, mutations or [])
        g.interface_idx = torch.where(interface_mask)[0].long()
        return g

    def build_joint_local_graph(
        self,
        graph: Data,
        radius: float,
        topk_fallback: int = 20,
        include_cross_partner_context: bool = True,
    ) -> Data:
        if graph is None or graph.num_nodes == 0:
            return make_empty_subgraph_like(graph) if graph is not None else self._empty_graph()

        if getattr(graph, "mut_idx", None) is not None and graph.mut_idx.numel() > 0:
            center_idx = graph.mut_idx.long().unique(sorted=True)
            center_coords = graph.pos[center_idx]
            dmat = torch.cdist(graph.pos, center_coords)
            min_dist, _ = dmat.min(dim=1)
            selected = torch.where(min_dist <= float(radius))[0]

            if selected.numel() == 0 and int(topk_fallback) > 0:
                k = min(int(topk_fallback), graph.num_nodes)
                selected = torch.topk(min_dist, k=k, largest=False).indices.sort().values

            if include_cross_partner_context and center_idx.numel() > 0:
                center_sides = {graph.partner_side[int(i)] for i in center_idx.tolist()}
                opposite_nodes = []
                for side in center_sides:
                    target_side = "ag" if side == "ab" else "ab"
                    mask = torch.tensor([s == target_side for s in graph.partner_side], dtype=torch.bool)
                    if mask.any():
                        opp_idx = torch.where(mask)[0]
                        opp_dist = torch.cdist(graph.pos[opp_idx], center_coords)
                        opp_min, _ = opp_dist.min(dim=1)
                        opp_keep = opp_idx[opp_min <= float(radius)]
                        if opp_keep.numel() == 0 and int(topk_fallback) > 0:
                            k = min(int(topk_fallback), opp_idx.numel())
                            opp_keep = opp_idx[torch.topk(opp_min, k=k, largest=False).indices]
                        if opp_keep.numel() > 0:
                            opposite_nodes.append(opp_keep)
                if opposite_nodes:
                    selected = torch.cat([selected] + opposite_nodes, dim=0).unique(sorted=True)
        else:
            if getattr(graph, "interface_idx", None) is not None and graph.interface_idx.numel() > 0:
                selected = graph.interface_idx.long().unique(sorted=True)
            else:
                selected = torch.arange(graph.num_nodes, dtype=torch.long)

        return induce_subgraph(graph, selected)

    def map_mutations_to_graph(self, graph: Data, mutations: List[Dict[str, Any]]) -> torch.Tensor:
        if graph.num_nodes == 0 or not mutations:
            return torch.tensor([], dtype=torch.long)
        key_to_idx = {k: i for i, k in enumerate(graph.residue_keys)}
        found_idx = []
        for mut in mutations:
            key = self._mutation_key(mut)
            if key in key_to_idx:
                found_idx.append(key_to_idx[key])
        return torch.tensor(sorted(list(set(found_idx))), dtype=torch.long)


# =========================================================
# 子图工具
# =========================================================
def _get_graph_feature_dim(graph: Data) -> int:
    if hasattr(graph, "x") and graph.x is not None and graph.x.dim() == 2:
        return int(graph.x.size(-1))
    return int(GRAPH_CONFIG.get("node_feat_dim", 159))


def _get_edge_feature_dim(graph: Data) -> int:
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None and graph.edge_attr.dim() == 2:
        return int(graph.edge_attr.size(-1))
    return int(GRAPH_CONFIG.get("edge_feat_dim", 6))


def make_empty_subgraph_like(graph: Optional[Data]) -> Data:
    feat_dim = _get_graph_feature_dim(graph) if graph is not None else int(GRAPH_CONFIG.get("node_feat_dim", 159))
    edge_feat_dim = _get_edge_feature_dim(graph) if graph is not None else int(GRAPH_CONFIG.get("edge_feat_dim", 6))

    sub = Data(
        x=torch.zeros((0, feat_dim), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, edge_feat_dim), dtype=torch.float32),
    )
    sub.pos = torch.zeros((0, 3), dtype=torch.float32)
    sub.seq_idx = torch.zeros((0,), dtype=torch.long)
    sub.residue_ids = []
    sub.residue_keys = []
    sub.chain_ids = []
    sub.partner_side = []
    sub.interface_mask = torch.zeros((0,), dtype=torch.bool)
    sub.mut_idx = torch.tensor([], dtype=torch.long)
    sub.interface_idx = torch.tensor([], dtype=torch.long)
    return sub


def induce_subgraph(graph: Data, selected_idx: torch.Tensor) -> Data:
    if graph is None or graph.num_nodes == 0 or selected_idx.numel() == 0:
        return make_empty_subgraph_like(graph)

    selected_idx = selected_idx.long().unique(sorted=True)
    idx_map = -torch.ones(graph.num_nodes, dtype=torch.long)
    idx_map[selected_idx] = torch.arange(selected_idx.size(0), dtype=torch.long)

    if graph.edge_index is not None and graph.edge_index.numel() > 0:
        row, col = graph.edge_index
        mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        mask[selected_idx] = True
        edge_mask = mask[row] & mask[col]
        sub_edge_index = graph.edge_index[:, edge_mask]
        sub_edge_index = idx_map[sub_edge_index]
        sub_edge_attr = graph.edge_attr[edge_mask] if getattr(graph, "edge_attr", None) is not None else None
    else:
        sub_edge_index = torch.zeros((2, 0), dtype=torch.long)
        sub_edge_attr = torch.zeros((0, _get_edge_feature_dim(graph)), dtype=torch.float32)

    sub = Data(
        x=graph.x[selected_idx].clone(),
        edge_index=sub_edge_index.clone(),
        edge_attr=sub_edge_attr.clone() if sub_edge_attr is not None else torch.zeros((0, _get_edge_feature_dim(graph)), dtype=torch.float32),
    )
    sub.pos = graph.pos[selected_idx].clone() if getattr(graph, "pos", None) is not None else torch.zeros((selected_idx.numel(), 3), dtype=torch.float32)
    sub.seq_idx = torch.arange(selected_idx.numel(), dtype=torch.long)
    sub.residue_ids = [graph.residue_ids[i] for i in selected_idx.tolist()] if hasattr(graph, "residue_ids") else []
    sub.residue_keys = [graph.residue_keys[i] for i in selected_idx.tolist()] if hasattr(graph, "residue_keys") else []
    sub.chain_ids = [graph.chain_ids[i] for i in selected_idx.tolist()] if hasattr(graph, "chain_ids") else []
    sub.partner_side = [graph.partner_side[i] for i in selected_idx.tolist()] if hasattr(graph, "partner_side") else []
    sub.interface_mask = graph.interface_mask[selected_idx].clone() if hasattr(graph, "interface_mask") else torch.zeros((selected_idx.numel(),), dtype=torch.bool)

    if hasattr(graph, "mut_idx") and graph.mut_idx is not None and graph.mut_idx.numel() > 0:
        keep = idx_map[graph.mut_idx]
        keep = keep[keep >= 0]
        sub.mut_idx = keep.long().unique(sorted=True)
    else:
        sub.mut_idx = torch.tensor([], dtype=torch.long)

    if hasattr(graph, "interface_idx") and graph.interface_idx is not None and graph.interface_idx.numel() > 0:
        keep = idx_map[graph.interface_idx]
        keep = keep[keep >= 0]
        sub.interface_idx = keep.long().unique(sorted=True)
    else:
        sub.interface_idx = torch.where(sub.interface_mask)[0].long()

    return sub


if __name__ == "__main__":
    print("pdb_graph.py updated for joint WT/MUT interface graph construction.")
