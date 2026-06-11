import re
import math
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from config import (
    DATA_CONFIG,
    COLUMN_CONFIG,
    FILE_CONFIG,
    MUTATION_CONFIG,
    PARTNER_CONFIG,
    GRAPH_CONFIG,
    FEATURE_CONFIG,
    DSSP_CONFIG,
    RUN_CONFIG,
    DEBUG_CONFIG,
    ABLATION_CONFIG,
)

from pdb_graph import ComplexGraphBuilder, make_empty_subgraph_like

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.load.*weights_only=False.*"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================================================
# 基础工具
# =========================================================
def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()



def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        if isinstance(x, float) and math.isnan(x):
            return default
        return float(x)
    except Exception:
        return default



def _normalize_icode(x: Any) -> str:
    s = _safe_str(x)
    return s if s else ""



def _to_tensor_label(x: float) -> torch.Tensor:
    return torch.tensor(float(x), dtype=torch.float32)



def _make_empty_graph(feature_dim: Optional[int] = None) -> Data:
    if feature_dim is None:
        feature_dim = int(GRAPH_CONFIG["node_feat_dim"])

    edge_feat_dim = int(GRAPH_CONFIG.get("edge_feat_dim", 2))

    g = Data(
        x=torch.zeros((0, feature_dim), dtype=torch.float32),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        edge_attr=torch.zeros((0, edge_feat_dim), dtype=torch.float32),
    )
    g.pos = torch.zeros((0, 3), dtype=torch.float32)
    g.seq_idx = torch.zeros((0,), dtype=torch.long)
    g.residue_ids = []
    g.residue_keys = []
    g.chain_ids = []
    g.partner_side = []
    g.mut_idx = torch.tensor([], dtype=torch.long)
    g.interface_idx = torch.tensor([], dtype=torch.long)
    return g


# =========================================================
# 离线特征读取
# =========================================================
def get_dssp_cache_dir(side: str) -> Path:
    side = _safe_str(side).lower()
    if side not in {"wt", "mut"}:
        raise ValueError(f"Invalid DSSP side: {side}")
    return Path(DSSP_CONFIG[f"{side}_dir"])



def get_antiberty_cache_dir(side: str) -> Path:
    side = _safe_str(side).lower()
    if side not in {"wt", "mut"}:
        raise ValueError(f"Invalid AntiBERTy side: {side}")
    return Path(FEATURE_CONFIG[f"antiberty_{side}_dir"])



def get_esm2_cache_dir(side: str) -> Path:
    side = _safe_str(side).lower()
    if side not in {"wt", "mut"}:
        raise ValueError(f"Invalid ESM2 side: {side}")
    return Path(FEATURE_CONFIG[f"esm2_{side}_dir"])



def load_dssp_features_for_pdb(
    pdb_id: str,
    side: str,
) -> Dict[Tuple[str, int, str], torch.Tensor]:
    """
    返回:
        {
            (chain_id, resseq, icode): tensor([ss_H, ss_E, ss_C, rasa], dtype=float32)
        }
    """
    pdb_id = _safe_str(pdb_id)
    dssp_path = get_dssp_cache_dir(side) / f"{pdb_id}.csv"

    if not dssp_path.exists():
        logging.warning(f"DSSP feature file not found: {dssp_path}")
        return {}

    try:
        df = pd.read_csv(dssp_path)
    except Exception as e:
        logging.warning(f"Failed to read DSSP file: {dssp_path} | {e}")
        return {}

    required_cols = {"chain_id", "resseq", "icode", "ss_H", "ss_E", "ss_C"}
    missing = required_cols - set(df.columns)
    if missing:
        logging.warning(f"DSSP file missing columns {sorted(missing)}: {dssp_path}")
        return {}

    use_rasa = "RASA" in df.columns
    use_asa_complex = "ASA_complex" in df.columns

    feats: Dict[Tuple[str, int, str], torch.Tensor] = {}
    for _, row in df.iterrows():
        try:
            chain_id = _safe_str(row["chain_id"])
            resseq = int(row["resseq"])
            icode = _normalize_icode(row.get("icode", ""))

            ss_h = float(row["ss_H"])
            ss_e = float(row["ss_E"])
            ss_c = float(row["ss_C"])

            if use_rasa:
                sasa_val = float(row["RASA"])
            elif use_asa_complex:
                sasa_val = float(row["ASA_complex"])
            else:
                sasa_val = 0.0

            feats[(chain_id, resseq, icode)] = torch.tensor(
                [ss_h, ss_e, ss_c, sasa_val],
                dtype=torch.float32,
            )
        except Exception:
            continue

    return feats


def _parse_embedding_file(file_path: Path) -> Dict[str, Any]:
    """
    支持以下格式:

    旧格式:
    1) {chain_id: emb[L, D]}
    2) List[(chain_id, seq, emb)]

    新格式:
    {
        "chain_embeddings": {chain_id: emb[L, D]},
        "residue_keys": {chain_id: [(chain_id, resseq, icode), ...]},
        "residue_embeddings": {(chain_id, resseq, icode): emb[D]}
    }
    """
    if not file_path.exists():
        return {
            "chain_embeddings": {},
            "residue_keys": {},
            "residue_embeddings": {},
        }

    try:
        obj = torch.load(file_path, map_location="cpu")
    except Exception as e:
        logging.warning(f"Failed to load embedding file: {file_path} | {e}")
        return {
            "chain_embeddings": {},
            "residue_keys": {},
            "residue_embeddings": {},
        }

    out = {
        "chain_embeddings": {},
        "residue_keys": {},
        "residue_embeddings": {},
    }

    if isinstance(obj, dict):
        # 新格式
        if (
            "chain_embeddings" in obj
            or "residue_keys" in obj
            or "residue_embeddings" in obj
        ):
            chain_embeddings = obj.get("chain_embeddings", {}) or {}
            residue_keys = obj.get("residue_keys", {}) or {}
            residue_embeddings = obj.get("residue_embeddings", {}) or {}

            for k, v in chain_embeddings.items():
                chain_id = _safe_str(k)
                if chain_id and isinstance(v, torch.Tensor):
                    out["chain_embeddings"][chain_id] = v.float()

            for k, v in residue_keys.items():
                chain_id = _safe_str(k)
                if chain_id and isinstance(v, list):
                    norm_list = []
                    for item in v:
                        if isinstance(item, (list, tuple)) and len(item) >= 3:
                            norm_list.append((
                                _safe_str(item[0]),
                                int(item[1]),
                                _normalize_icode(item[2]),
                            ))
                    out["residue_keys"][chain_id] = norm_list

            for k, v in residue_embeddings.items():
                if (
                    isinstance(k, (list, tuple))
                    and len(k) >= 3
                    and isinstance(v, torch.Tensor)
                ):
                    key = (
                        _safe_str(k[0]),
                        int(k[1]),
                        _normalize_icode(k[2]),
                    )
                    out["residue_embeddings"][key] = v.float()

            return out

        # 旧格式1: {chain_id: emb}
        for k, v in obj.items():
            chain_id = _safe_str(k)
            if not chain_id:
                continue

            if isinstance(v, torch.Tensor):
                out["chain_embeddings"][chain_id] = v.float()
            elif isinstance(v, (list, tuple)) and len(v) >= 1 and isinstance(v[-1], torch.Tensor):
                out["chain_embeddings"][chain_id] = v[-1].float()

        return out

    # 旧格式2: List[(chain_id, seq, emb)]
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                chain_id = _safe_str(item[0])
                emb = item[2]
                if chain_id and isinstance(emb, torch.Tensor):
                    out["chain_embeddings"][chain_id] = emb.float()
        return out

    logging.warning(f"Unexpected embedding object type in {file_path}: {type(obj)}")
    return out

def load_antiberty_features_for_pdb(pdb_id: str, side: str) -> Dict[str, Any]:
    pdb_id = _safe_str(pdb_id)
    file_path = get_antiberty_cache_dir(side) / f"{pdb_id}.pt"
    out = _parse_embedding_file(file_path)

    if FEATURE_CONFIG.get("use_antiberty", True) and not (
        out["chain_embeddings"] or out["residue_embeddings"]
    ):
        logging.warning(
            f"AntiBERTy feature file not found or empty for pdb_id={pdb_id}, side={side} "
            f"(checked: {file_path.name})"
        )
    return out



def load_esm2_features_for_pdb(pdb_id: str, side: str) -> Dict[str, Any]:
    pdb_id = _safe_str(pdb_id)
    file_path = get_esm2_cache_dir(side) / f"{pdb_id}.pt"
    out = _parse_embedding_file(file_path)

    if FEATURE_CONFIG.get("use_esm2", True) and not (
        out["chain_embeddings"] or out["residue_embeddings"]
    ):
        logging.warning(
            f"ESM2 feature file not found or empty for pdb_id={pdb_id}, side={side} "
            f"(checked: {file_path.name})"
        )
    return out


# =========================================================
# Mutation 解析
# 格式: <wt_aa><chain><resseq><icode可选><mut_aa>
# 例如: WA33F, YH52A, WA33AF
# =========================================================
_MUT_PATTERN = re.compile(r"^([A-Z])([A-Za-z0-9])(\d+)([A-Za-z]?)([A-Z])$")



def parse_mutation_token(token: str) -> Dict[str, Any]:
    token = _safe_str(token).replace(" ", "")
    if not token:
        raise ValueError("Empty mutation token")

    m = _MUT_PATTERN.match(token)
    if m is None:
        raise ValueError(f"Invalid mutation token: {token}")

    wt_aa, chain_id, resseq, icode, mut_aa = m.groups()
    icode = _safe_str(icode).upper()

    allowed_aas = MUTATION_CONFIG["allowed_amino_acids"]
    if wt_aa not in allowed_aas:
        raise ValueError(f"Invalid wild-type amino acid '{wt_aa}' in token: {token}")
    if mut_aa not in allowed_aas:
        raise ValueError(f"Invalid mutant amino acid '{mut_aa}' in token: {token}")

    return {
        "raw": token,
        "wt_aa": wt_aa,
        "chain": chain_id,
        "resseq": int(resseq),
        "icode": icode,
        "mut_aa": mut_aa,
    }



def parse_mutation_string(mutation_str: str) -> List[Dict[str, Any]]:
    mutation_str = _safe_str(mutation_str)
    if not mutation_str:
        return []

    tokens = [x.strip() for x in re.split(r"[;,]", mutation_str) if x.strip()]
    mutations = [parse_mutation_token(tok) for tok in tokens]
    return mutations


# =========================================================
# Partners 解析
# 下划线左边 = 抗体链
# 下划线右边 = 抗原链
# =========================================================
def parse_partners_string(partners_str: str) -> Tuple[List[str], List[str]]:
    partners_str = _safe_str(partners_str).replace(" ", "")
    if not partners_str:
        raise ValueError("Empty Partners field")

    sep = PARTNER_CONFIG["group_sep"]
    if sep not in partners_str:
        raise ValueError(f"Partners field must contain '{sep}': {partners_str}")

    left, right = partners_str.split(sep, 1)
    if not left or not right:
        raise ValueError(f"Invalid Partners field: {partners_str}")

    ab_chains = list(left)
    ag_chains = list(right)
    return ab_chains, ag_chains



def split_mutations_by_partner_side(
    mutations: List[Dict[str, Any]],
    ab_chains: List[str],
    ag_chains: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ab_set = set(ab_chains)
    ag_set = set(ag_chains)

    ab_muts = []
    ag_muts = []

    for mut in mutations:
        ch = mut["chain"]
        if ch in ab_set:
            ab_muts.append(mut)
        elif ch in ag_set:
            ag_muts.append(mut)
        else:
            raise ValueError(
                f"Mutation chain '{ch}' not found in Partners definition. "
                f"ab_chains={ab_chains}, ag_chains={ag_chains}, mutation={mut['raw']}"
            )

    return ab_muts, ag_muts


# =========================================================
# 文件查找
# =========================================================
def resolve_pdb_path(
    pdb_id: str,
    pdb_dir: Path,
    suffix: str = ".pdb",
    case_sensitive: bool = False,
) -> Optional[Path]:
    pdb_id = _safe_str(pdb_id)
    if not pdb_id:
        return None

    direct_path = pdb_dir / f"{pdb_id}{suffix}"
    if direct_path.exists():
        return direct_path

    if case_sensitive:
        return None

    target = f"{pdb_id}{suffix}".lower()
    for p in pdb_dir.iterdir():
        if p.is_file() and p.name.lower() == target:
            return p

    return None


# =========================================================
# 行合法性检查
# =========================================================
def validate_row(row: pd.Series) -> Tuple[bool, str]:
    mut_pdb_col = COLUMN_CONFIG["mut_pdb_id"]
    wt_pdb_col = COLUMN_CONFIG["wt_pdb_id"]
    mutation_col = COLUMN_CONFIG["mutation"]
    partners_col = COLUMN_CONFIG["partners"]
    label_col = COLUMN_CONFIG["label"]

    mut_pdb_id = _safe_str(row.get(mut_pdb_col))
    wt_pdb_id = _safe_str(row.get(wt_pdb_col))
    mutation_str = _safe_str(row.get(mutation_col))
    partners_str = _safe_str(row.get(partners_col))
    ddg_val = _safe_float(row.get(label_col), default=None)

    if not mut_pdb_id:
        return False, f"Missing '{mut_pdb_col}'"
    if not wt_pdb_id:
        return False, f"Missing '{wt_pdb_col}'"
    if not mutation_str:
        return False, f"Missing '{mutation_col}'"
    if not partners_str:
        return False, f"Missing '{partners_col}'"
    if ddg_val is None:
        return False, f"Invalid '{label_col}'"

    try:
        muts = parse_mutation_string(mutation_str)
        ab_chains, ag_chains = parse_partners_string(partners_str)
        split_mutations_by_partner_side(muts, ab_chains, ag_chains)
    except Exception as e:
        return False, str(e)

    return True, "OK"


# =========================================================
# 缓存文件名
# =========================================================
def make_sample_cache_name(row: pd.Series, row_idx: int) -> str:
    mut_pdb_col = COLUMN_CONFIG["mut_pdb_id"]
    wt_pdb_col = COLUMN_CONFIG["wt_pdb_id"]
    mutation_col = COLUMN_CONFIG["mutation"]

    mut_id = _safe_str(row.get(mut_pdb_col))
    wt_id = _safe_str(row.get(wt_pdb_col))
    mutation_str = _safe_str(row.get(mutation_col))

    if not mut_id:
        raise ValueError(f"Missing '{mut_pdb_col}' when making cache file name")

    def _sanitize(s: str) -> str:
        return (
            str(s).strip()
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .replace("*", "_")
            .replace("?", "_")
            .replace('"', "_")
            .replace("<", "_")
            .replace(">", "_")
            .replace("|", "_")
            .replace(" ", "")
            .replace(";", "_")
            .replace(",", "_")
        )

    mut_id = _sanitize(mut_id)
    wt_id = _sanitize(wt_id) if wt_id else "WTUNK"
    mutation_str = _sanitize(mutation_str) if mutation_str else "MUTUNK"

    return f"{mut_id}.pt"

# =========================================================
# DataFrame 读取
# =========================================================
def load_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=DATA_CONFIG.get("csv_sep", ","))
    if DEBUG_CONFIG.get("debug", False) and DEBUG_CONFIG.get("max_samples") is not None:
        df = df.iloc[: DEBUG_CONFIG["max_samples"]].copy()

    logging.info(f"Loaded dataframe: {csv_path}, total rows={len(df)}")
    return df

def _mutation_key_from_dict(mut: Dict[str, Any]) -> Tuple[str, int, str]:
    return (
        str(mut["chain"]),
        int(mut["resseq"]),
        _normalize_icode(mut.get("icode", "")),
    )


def patch_wt_mut_idx_by_mut_graph_nn(
    wt_graph: Data,
    mut_graph: Data,
    mutations: List[Dict[str, Any]],
) -> torch.Tensor:
    """
    当 WT 侧按 (chain, resseq, icode) 精确映射失败时：
    - 若 MUT 侧该突变已成功命中
    - 则用 MUT 命中节点的坐标，在 WT 图中同链寻找最近邻节点补映射

    返回补丁后的 wt_mut_idx（torch.long）
    """
    if wt_graph is None or mut_graph is None:
        return torch.tensor([], dtype=torch.long)

    if getattr(wt_graph, "num_nodes", 0) == 0 or getattr(mut_graph, "num_nodes", 0) == 0:
        return getattr(wt_graph, "mut_idx", torch.tensor([], dtype=torch.long))

    # 原有 WT 精确命中
    wt_old = getattr(wt_graph, "mut_idx", None)
    if wt_old is None:
        wt_old = torch.tensor([], dtype=torch.long)
    wt_old = wt_old.long().unique(sorted=True)

    # MUT 精确命中
    mut_old = getattr(mut_graph, "mut_idx", None)
    if mut_old is None:
        mut_old = torch.tensor([], dtype=torch.long)
    mut_old = mut_old.long().unique(sorted=True)

    if len(mutations) == 0:
        return wt_old

    # key -> idx
    wt_key_to_idx = {}
    for i, k in enumerate(getattr(wt_graph, "residue_keys", [])):
        wt_key_to_idx[k] = i

    mut_key_to_idx = {}
    for i, k in enumerate(getattr(mut_graph, "residue_keys", [])):
        mut_key_to_idx[k] = i

    repaired = set(wt_old.tolist())

    # WT 候选信息
    wt_chain_ids = list(getattr(wt_graph, "chain_ids", []))
    wt_partner_side = list(getattr(wt_graph, "partner_side", []))
    wt_pos = getattr(wt_graph, "pos", None)

    mut_pos = getattr(mut_graph, "pos", None)

    if wt_pos is None or mut_pos is None:
        return wt_old

    for mut in mutations:
        key = _mutation_key_from_dict(mut)

        # 1) WT 精确命中：直接保留
        if key in wt_key_to_idx:
            repaired.add(wt_key_to_idx[key])
            continue

        # 2) WT 没命中，但 MUT 也没命中：没法补
        if key not in mut_key_to_idx:
            continue

        # 3) 用 MUT 命中的坐标，在 WT 图中找最近邻
        mut_idx_in_mut = mut_key_to_idx[key]
        ref_coord = mut_pos[mut_idx_in_mut]   # [3]

        target_chain = str(mut["chain"])

        # 优先：同链候选
        candidate_idx = [
            i for i, ch in enumerate(wt_chain_ids)
            if str(ch) == target_chain
        ]

        # 如果同链一个都没有，再退化到同 partner side
        if len(candidate_idx) == 0:
            if key in mut_key_to_idx:
                mut_side = mut_graph.partner_side[mut_idx_in_mut]
                candidate_idx = [
                    i for i, s in enumerate(wt_partner_side)
                    if s == mut_side
                ]

        # 还没有，就退到全图
        if len(candidate_idx) == 0:
            candidate_idx = list(range(wt_graph.num_nodes))

        cand_idx_t = torch.tensor(candidate_idx, dtype=torch.long)
        cand_pos = wt_pos[cand_idx_t]   # [N, 3]

        dists = torch.norm(cand_pos - ref_coord.unsqueeze(0), dim=1)
        nearest_local = int(torch.argmin(dists).item())
        nearest_dist = float(dists[nearest_local].item())
        nearest_global = int(cand_idx_t[nearest_local].item())

        if nearest_dist <= 12.0:
            repaired.add(nearest_global)

    return torch.tensor(sorted(repaired), dtype=torch.long)


# =========================================================
# WT-MUT 对齐 / pair 特征构建
# 这些字段会被 train.py / val.py 的 collate 透传给模型：
#   aligned_wt_idx / aligned_mut_idx
#   mutation_aligned_wt_idx / mutation_aligned_mut_idx
#   mutation_pair_index / mutation_pair_feat
#   interface_pair_index / interface_pair_feat
#   interface_contact_pair_index / interface_contact_pair_feat
# =========================================================
def _as_long_idx(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return torch.tensor([], dtype=torch.long)
        return x.detach().cpu().long().view(-1).unique(sorted=True)
    if isinstance(x, (list, tuple)) and len(x) > 0:
        try:
            return torch.tensor(list(x), dtype=torch.long).view(-1).unique(sorted=True)
        except Exception:
            return torch.tensor([], dtype=torch.long)
    return torch.tensor([], dtype=torch.long)


def _norm_residue_key(k: Any) -> Optional[Tuple[str, int, str]]:
    """统一 residue key，兼容 tuple/list/tensor/scalar 嵌套。"""
    def _scalar(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu()
            if x.numel() == 0:
                return ""
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
    if not isinstance(k, (list, tuple)) or len(k) < 3:
        return None

    chain = _safe_str(_scalar(k[0]))
    if not chain:
        return None
    try:
        resseq = int(_scalar(k[1]))
    except Exception:
        return None
    icode = _normalize_icode(_scalar(k[2]))
    return (chain, resseq, icode)


def _graph_residue_key_map(graph: Data) -> Dict[Tuple[str, int, str], int]:
    out: Dict[Tuple[str, int, str], int] = {}
    for i, k in enumerate(list(getattr(graph, "residue_keys", []) or [])):
        nk = _norm_residue_key(k)
        if nk is not None:
            out[nk] = int(i)
    return out


def _valid_idx(idx: torch.Tensor, n: int) -> torch.Tensor:
    idx = _as_long_idx(idx)
    if idx.numel() == 0:
        return idx
    return idx[(idx >= 0) & (idx < int(n))].long().unique(sorted=True)


def _indices_within_radius(graph: Data, center_idx: torch.Tensor, radius: float, topk: int = 0) -> torch.Tensor:
    if graph is None or getattr(graph, "num_nodes", 0) == 0:
        return torch.tensor([], dtype=torch.long)
    pos = getattr(graph, "pos", None)
    if not isinstance(pos, torch.Tensor) or pos.dim() != 2 or pos.size(0) == 0:
        return torch.tensor([], dtype=torch.long)

    center_idx = _valid_idx(center_idx, graph.num_nodes)
    if center_idx.numel() == 0:
        return torch.tensor([], dtype=torch.long)

    center_pos = pos[center_idx]
    dist = torch.cdist(pos.float(), center_pos.float()).min(dim=1).values
    keep = torch.where(dist <= float(radius))[0].long()
    if keep.numel() == 0 and int(topk) > 0:
        k = min(int(topk), int(graph.num_nodes))
        keep = torch.topk(dist, k=k, largest=False).indices.long()
    return keep.unique(sorted=True)


def build_aligned_indices_from_graphs(wt_graph: Data, mut_graph: Data) -> Tuple[torch.Tensor, torch.Tensor]:
    """按 residue key 精确对齐 WT/MUT 局部图节点。"""
    if wt_graph is None or mut_graph is None:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    wt_map = _graph_residue_key_map(wt_graph)
    mut_map = _graph_residue_key_map(mut_graph)
    common = sorted([k for k in wt_map.keys() if k in mut_map], key=lambda x: (x[0], x[1], x[2]))

    if not common:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    return (
        torch.tensor([wt_map[k] for k in common], dtype=torch.long),
        torch.tensor([mut_map[k] for k in common], dtype=torch.long),
    )


def _aligned_subset(
    aligned_wt_idx: torch.Tensor,
    aligned_mut_idx: torch.Tensor,
    allowed_wt_idx: torch.Tensor,
    allowed_mut_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    aligned_wt_idx = _as_long_idx(aligned_wt_idx)
    aligned_mut_idx = _as_long_idx(aligned_mut_idx)
    allowed_wt = set(_as_long_idx(allowed_wt_idx).tolist())
    allowed_mut = set(_as_long_idx(allowed_mut_idx).tolist())

    if aligned_wt_idx.numel() == 0 or aligned_mut_idx.numel() == 0:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

    n = min(aligned_wt_idx.numel(), aligned_mut_idx.numel())
    keep_wt: List[int] = []
    keep_mut: List[int] = []
    for w, m in zip(aligned_wt_idx[:n].tolist(), aligned_mut_idx[:n].tolist()):
        if int(w) in allowed_wt and int(m) in allowed_mut:
            keep_wt.append(int(w))
            keep_mut.append(int(m))

    if not keep_wt:
        return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
    return torch.tensor(keep_wt, dtype=torch.long), torch.tensor(keep_mut, dtype=torch.long)


def build_mutation_aligned_indices(
    wt_graph: Data,
    mut_graph: Data,
    aligned_wt_idx: torch.Tensor,
    aligned_mut_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """优先使用 residue-key 对齐；失败时退回 WT/MUT mut_idx 一一配对。"""
    wt_mut_idx = _valid_idx(getattr(wt_graph, "mut_idx", None), getattr(wt_graph, "num_nodes", 0))
    mut_mut_idx = _valid_idx(getattr(mut_graph, "mut_idx", None), getattr(mut_graph, "num_nodes", 0))

    mut_wt, mut_mut = _aligned_subset(
        aligned_wt_idx=aligned_wt_idx,
        aligned_mut_idx=aligned_mut_idx,
        allowed_wt_idx=wt_mut_idx,
        allowed_mut_idx=mut_mut_idx,
    )
    if mut_wt.numel() > 0 and mut_mut.numel() > 0:
        return mut_wt, mut_mut

    # 关键 fallback：SARS 这类外测样本即使没有全局 aligned pair，
    # wt_graph.mut_idx / mut_graph.mut_idx 通常已经命中突变中心。
    if wt_mut_idx.numel() > 0 and mut_mut_idx.numel() > 0:
        n = min(wt_mut_idx.numel(), mut_mut_idx.numel())
        return wt_mut_idx[:n].long(), mut_mut_idx[:n].long()

    return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)


def _make_pair_index_and_feat(
    wt_graph: Data,
    mut_graph: Data,
    wt_idx: torch.Tensor,
    mut_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成 2xN pair_index 和 5 维 pair_feat。"""
    wt_idx = _valid_idx(wt_idx, getattr(wt_graph, "num_nodes", 0))
    mut_idx = _valid_idx(mut_idx, getattr(mut_graph, "num_nodes", 0))
    n = min(wt_idx.numel(), mut_idx.numel())
    if n == 0:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 5), dtype=torch.float32)

    wt_idx = wt_idx[:n].long()
    mut_idx = mut_idx[:n].long()
    pair_index = torch.stack([wt_idx, mut_idx], dim=0).contiguous()

    wt_pos = getattr(wt_graph, "pos", None)
    mut_pos = getattr(mut_graph, "pos", None)
    if isinstance(wt_pos, torch.Tensor) and isinstance(mut_pos, torch.Tensor):
        d = torch.norm(wt_pos[wt_idx].float() - mut_pos[mut_idx].float(), dim=-1)
    else:
        d = torch.zeros((n,), dtype=torch.float32)

    inv_d = 1.0 / (d + 1e-6)
    wt_mut_set = set(_as_long_idx(getattr(wt_graph, "mut_idx", None)).tolist())
    mut_mut_set = set(_as_long_idx(getattr(mut_graph, "mut_idx", None)).tolist())
    wt_int_set = set(_as_long_idx(getattr(wt_graph, "interface_idx", None)).tolist())
    mut_int_set = set(_as_long_idx(getattr(mut_graph, "interface_idx", None)).tolist())
    wt_side = list(getattr(wt_graph, "partner_side", []) or [])
    mut_side = list(getattr(mut_graph, "partner_side", []) or [])

    is_mut_pair = []
    is_interface_pair = []
    cross_side = []
    for w, m in zip(wt_idx.tolist(), mut_idx.tolist()):
        is_mut_pair.append(1.0 if (w in wt_mut_set or m in mut_mut_set) else 0.0)
        is_interface_pair.append(1.0 if (w in wt_int_set or m in mut_int_set) else 0.0)
        sw = wt_side[w] if w < len(wt_side) else ""
        sm = mut_side[m] if m < len(mut_side) else ""
        cross_side.append(1.0 if sw and sm and sw != sm else 0.0)

    pair_feat = torch.stack(
        [
            d.float(),
            inv_d.float(),
            torch.tensor(is_mut_pair, dtype=torch.float32),
            torch.tensor(is_interface_pair, dtype=torch.float32),
            torch.tensor(cross_side, dtype=torch.float32),
        ],
        dim=-1,
    ).contiguous()
    return pair_index, pair_feat


def _contact_pairs_in_graph(graph: Data, threshold: float) -> Dict[Tuple[Tuple[str, int, str], Tuple[str, int, str]], Tuple[int, int, float]]:
    """返回 cross-partner contact pair: ((ab_key, ag_key) -> (ab_idx, ag_idx, distance))."""
    out: Dict[Tuple[Tuple[str, int, str], Tuple[str, int, str]], Tuple[int, int, float]] = {}
    if graph is None or getattr(graph, "num_nodes", 0) == 0:
        return out
    pos = getattr(graph, "pos", None)
    if not isinstance(pos, torch.Tensor) or pos.dim() != 2 or pos.size(0) == 0:
        return out

    partner_side = list(getattr(graph, "partner_side", []) or [])
    residue_keys_raw = list(getattr(graph, "residue_keys", []) or [])
    if len(partner_side) != graph.num_nodes or len(residue_keys_raw) != graph.num_nodes:
        return out

    keys = [_norm_residue_key(k) for k in residue_keys_raw]
    ab_idx = [i for i, s in enumerate(partner_side) if s == "ab" and keys[i] is not None]
    ag_idx = [i for i, s in enumerate(partner_side) if s == "ag" and keys[i] is not None]
    if not ab_idx or not ag_idx:
        return out

    ab_t = torch.tensor(ab_idx, dtype=torch.long)
    ag_t = torch.tensor(ag_idx, dtype=torch.long)
    dmat = torch.cdist(pos[ab_t].float(), pos[ag_t].float())
    src, dst = torch.where(dmat <= float(threshold))
    for a_local, g_local in zip(src.tolist(), dst.tolist()):
        ai = int(ab_t[a_local].item())
        gi = int(ag_t[g_local].item())
        ak = keys[ai]
        gk = keys[gi]
        if ak is None or gk is None:
            continue
        out[(ak, gk)] = (ai, gi, float(dmat[a_local, g_local].item()))
    return out


def build_interface_contact_pairs(
    wt_graph: Data,
    mut_graph: Data,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """生成 contact-delta 分支需要的 4xN index 和 16 维 pair feature。"""
    contact_threshold = float(GRAPH_CONFIG.get("contact_threshold", 8.0))
    interface_threshold = float(GRAPH_CONFIG.get("interface_contact_threshold", GRAPH_CONFIG.get("interface_edge_threshold", 12.0)))

    wt_pairs = _contact_pairs_in_graph(wt_graph, threshold=interface_threshold)
    mut_pairs = _contact_pairs_in_graph(mut_graph, threshold=interface_threshold)
    keys = sorted(set(wt_pairs.keys()) | set(mut_pairs.keys()), key=lambda x: (x[0], x[1]))

    wt_map = _graph_residue_key_map(wt_graph)
    mut_map = _graph_residue_key_map(mut_graph)

    pair_cols: List[List[int]] = []
    feat_rows: List[List[float]] = []
    wt_chain_ids = list(getattr(wt_graph, "chain_ids", []) or [])
    mut_chain_ids = list(getattr(mut_graph, "chain_ids", []) or [])

    for ab_key, ag_key in keys:
        if ab_key not in wt_map or ag_key not in wt_map or ab_key not in mut_map or ag_key not in mut_map:
            continue

        wt_ab = int(wt_map[ab_key])
        wt_ag = int(wt_map[ag_key])
        mut_ab = int(mut_map[ab_key])
        mut_ag = int(mut_map[ag_key])

        wt_d = float(torch.norm(wt_graph.pos[wt_ab].float() - wt_graph.pos[wt_ag].float()).item())
        mut_d = float(torch.norm(mut_graph.pos[mut_ab].float() - mut_graph.pos[mut_ag].float()).item())
        delta_d = mut_d - wt_d

        wt_contact = 1.0 if wt_d <= contact_threshold else 0.0
        mut_contact = 1.0 if mut_d <= contact_threshold else 0.0
        wt_interface = 1.0 if wt_d <= interface_threshold else 0.0
        mut_interface = 1.0 if mut_d <= interface_threshold else 0.0

        pair_cols.append([wt_ab, wt_ag, mut_ab, mut_ag])
        feat_rows.append([
            wt_d,
            mut_d,
            delta_d,
            abs(delta_d),
            wt_contact,
            mut_contact,
            1.0 if wt_contact == 0.0 and mut_contact == 1.0 else 0.0,
            1.0 if wt_contact == 1.0 and mut_contact == 0.0 else 0.0,
            wt_interface,
            mut_interface,
            1.0 if wt_interface == 0.0 and mut_interface == 1.0 else 0.0,
            1.0 if wt_interface == 1.0 and mut_interface == 0.0 else 0.0,
            1.0 / (wt_d + 1e-6),
            1.0 / (mut_d + 1e-6),
            1.0 if (wt_ab < len(wt_chain_ids) and mut_ab < len(mut_chain_ids) and wt_chain_ids[wt_ab] == mut_chain_ids[mut_ab]) else 0.0,
            1.0 if (wt_ag < len(wt_chain_ids) and mut_ag < len(mut_chain_ids) and wt_chain_ids[wt_ag] == mut_chain_ids[mut_ag]) else 0.0,
        ])

    if not pair_cols:
        return torch.zeros((4, 0), dtype=torch.long), torch.zeros((0, 16), dtype=torch.float32)

    return (
        torch.tensor(pair_cols, dtype=torch.long).t().contiguous(),
        torch.tensor(feat_rows, dtype=torch.float32).contiguous(),
    )


def enrich_sample_with_alignment_and_pairs(sample: Dict[str, Any]) -> Dict[str, Any]:
    """给 preprocess 产物补齐模型需要的 alignment / pair 字段。"""
    wt_graph = sample["wt_joint_graph"]
    mut_graph = sample["mut_joint_graph"]

    aligned_wt_idx, aligned_mut_idx = build_aligned_indices_from_graphs(wt_graph, mut_graph)
    mutation_aligned_wt_idx, mutation_aligned_mut_idx = build_mutation_aligned_indices(
        wt_graph=wt_graph,
        mut_graph=mut_graph,
        aligned_wt_idx=aligned_wt_idx,
        aligned_mut_idx=aligned_mut_idx,
    )

    local_radius = float(GRAPH_CONFIG.get("local_radius", 12.0))
    topk = int(GRAPH_CONFIG.get("local_topk_fallback", 20))
    wt_mut_shell = _indices_within_radius(wt_graph, getattr(wt_graph, "mut_idx", None), radius=local_radius, topk=topk)
    mut_mut_shell = _indices_within_radius(mut_graph, getattr(mut_graph, "mut_idx", None), radius=local_radius, topk=topk)
    mutation_shell_wt_idx, mutation_shell_mut_idx = _aligned_subset(
        aligned_wt_idx,
        aligned_mut_idx,
        wt_mut_shell,
        mut_mut_shell,
    )
    if mutation_shell_wt_idx.numel() == 0 or mutation_shell_mut_idx.numel() == 0:
        mutation_shell_wt_idx = mutation_aligned_wt_idx
        mutation_shell_mut_idx = mutation_aligned_mut_idx

    wt_interface_idx = _valid_idx(getattr(wt_graph, "interface_idx", None), getattr(wt_graph, "num_nodes", 0))
    mut_interface_idx = _valid_idx(getattr(mut_graph, "interface_idx", None), getattr(mut_graph, "num_nodes", 0))
    interface_aligned_wt_idx, interface_aligned_mut_idx = _aligned_subset(
        aligned_wt_idx,
        aligned_mut_idx,
        wt_interface_idx,
        mut_interface_idx,
    )

    # all_pair 使用所有 key-aligned 节点；若没有，则至少保留 mutation pair fallback。
    if aligned_wt_idx.numel() == 0 or aligned_mut_idx.numel() == 0:
        all_pair_index, all_pair_feat = _make_pair_index_and_feat(wt_graph, mut_graph, mutation_aligned_wt_idx, mutation_aligned_mut_idx)
    else:
        all_pair_index, all_pair_feat = _make_pair_index_and_feat(wt_graph, mut_graph, aligned_wt_idx, aligned_mut_idx)

    mutation_pair_index, mutation_pair_feat = _make_pair_index_and_feat(
        wt_graph,
        mut_graph,
        mutation_aligned_wt_idx,
        mutation_aligned_mut_idx,
    )
    interface_pair_index, interface_pair_feat = _make_pair_index_and_feat(
        wt_graph,
        mut_graph,
        interface_aligned_wt_idx,
        interface_aligned_mut_idx,
    )
    interface_contact_pair_index, interface_contact_pair_feat = build_interface_contact_pairs(wt_graph, mut_graph)

    sample.update({
        "aligned_wt_idx": aligned_wt_idx,
        "aligned_mut_idx": aligned_mut_idx,
        "mutation_shell_wt_idx": mutation_shell_wt_idx,
        "mutation_shell_mut_idx": mutation_shell_mut_idx,
        "interface_shell_wt_idx": interface_aligned_wt_idx,
        "interface_shell_mut_idx": interface_aligned_mut_idx,
        "mutation_aligned_wt_idx": mutation_aligned_wt_idx,
        "mutation_aligned_mut_idx": mutation_aligned_mut_idx,
        "interface_aligned_wt_idx": interface_aligned_wt_idx,
        "interface_aligned_mut_idx": interface_aligned_mut_idx,
        "all_pair_index": all_pair_index,
        "all_pair_feat": all_pair_feat,
        "mutation_pair_index": mutation_pair_index,
        "mutation_pair_feat": mutation_pair_feat,
        "interface_pair_index": interface_pair_index,
        "interface_pair_feat": interface_pair_feat,
        "interface_contact_pair_index": interface_contact_pair_index,
        "interface_contact_pair_feat": interface_contact_pair_feat,
    })
    return sample


# =========================================================
# 单样本构建（供 preprocess.py 使用）
# 新版本：
# - 只返回 WT/MUT 两张联合局部图
# - 抗体侧用 AntiBERTy
# - 抗原侧用 ESM2
# - 预期配合新版 pdb_graph.py 中的 joint graph API
# =========================================================
def build_sample_from_row(
    row: pd.Series,
    graph_builder: ComplexGraphBuilder,
    wt_pdb_dir: Path,
    mut_pdb_dir: Path,
    row_idx: Optional[int] = None,
) -> Dict[str, Any]:
    mut_pdb_col = COLUMN_CONFIG["mut_pdb_id"]
    wt_pdb_col = COLUMN_CONFIG["wt_pdb_id"]
    mutation_col = COLUMN_CONFIG["mutation"]
    partners_col = COLUMN_CONFIG["partners"]
    label_col = COLUMN_CONFIG["label"]

    pdb_suffix = FILE_CONFIG["pdb_suffix"]
    case_sensitive = FILE_CONFIG["case_sensitive"]

    mut_pdb_id = _safe_str(row[mut_pdb_col])
    wt_pdb_id = _safe_str(row[wt_pdb_col])
    mutation_str = _safe_str(row[mutation_col])
    partners_str = _safe_str(row[partners_col])
    ddg_val = _safe_float(row[label_col], default=None)
    sample_id = mut_pdb_id

    if ddg_val is None:
        raise ValueError(f"Invalid ddG value for row: {row.to_dict()}")

    mutations = parse_mutation_string(mutation_str)
    ab_chains, ag_chains = parse_partners_string(partners_str)
    ab_muts, ag_muts = split_mutations_by_partner_side(mutations, ab_chains, ag_chains)

    wt_path = resolve_pdb_path(
        pdb_id=wt_pdb_id,
        pdb_dir=wt_pdb_dir,
        suffix=pdb_suffix,
        case_sensitive=case_sensitive,
    )
    mut_path = resolve_pdb_path(
        pdb_id=mut_pdb_id,
        pdb_dir=mut_pdb_dir,
        suffix=pdb_suffix,
        case_sensitive=case_sensitive,
    )

    if wt_path is None:
        raise FileNotFoundError(f"WT pdb not found for '{wt_pdb_id}' under {wt_pdb_dir}")
    if mut_path is None:
        raise FileNotFoundError(f"MUT pdb not found for '{mut_pdb_id}' under {mut_pdb_dir}")

    # 读取 WT / MUT 对应的离线特征
    # 消融 PLM / DSSP 时不改变节点维度，只是不读取对应离线特征，后续 featurizer 会填 0。
    use_dssp = bool(ABLATION_CONFIG.get("use_dssp", True))
    use_plm = bool(ABLATION_CONFIG.get("use_plm", True))

    wt_dssp_feats = load_dssp_features_for_pdb(wt_pdb_id, side="wt") if use_dssp else {}
    mut_dssp_feats = load_dssp_features_for_pdb(mut_pdb_id, side="mut") if use_dssp else {}

    wt_antiberty_embeddings = load_antiberty_features_for_pdb(wt_pdb_id, side="wt") if use_plm else {}
    mut_antiberty_embeddings = load_antiberty_features_for_pdb(mut_pdb_id, side="mut") if use_plm else {}

    wt_esm2_embeddings = load_esm2_features_for_pdb(wt_pdb_id, side="wt") if use_plm else {}
    mut_esm2_embeddings = load_esm2_features_for_pdb(mut_pdb_id, side="mut") if use_plm else {}

    # 预期新版 pdb_graph.py 提供以下接口：
    # 1) build_joint_graph(...)
    # 2) build_joint_local_graph(...)
    if not hasattr(graph_builder, "build_joint_graph"):
        raise AttributeError(
            "ComplexGraphBuilder is expected to implement 'build_joint_graph' in the updated pdb_graph.py"
        )
    if not hasattr(graph_builder, "build_joint_local_graph"):
        raise AttributeError(
            "ComplexGraphBuilder is expected to implement 'build_joint_local_graph' in the updated pdb_graph.py"
        )

    wt_joint_full_graph = graph_builder.build_joint_graph(
        pdb_path=str(wt_path),
        ab_chains=ab_chains,
        ag_chains=ag_chains,
        dssp_feats=wt_dssp_feats,
        antiberty_embeddings=wt_antiberty_embeddings,
        esm2_embeddings=wt_esm2_embeddings,
        mutations=mutations,
    )

    mut_joint_full_graph = graph_builder.build_joint_graph(
        pdb_path=str(mut_path),
        ab_chains=ab_chains,
        ag_chains=ag_chains,
        dssp_feats=mut_dssp_feats,
        antiberty_embeddings=mut_antiberty_embeddings,
        esm2_embeddings=mut_esm2_embeddings,
        mutations=mutations,
    )

    # -------------------------------------------------
    # WT 映射失败时：使用 MUT 已命中的突变节点坐标做最近邻补映射
    # 仅修 WT，不改 MUT
    # -------------------------------------------------
    wt_old_mut_idx = getattr(wt_joint_full_graph, "mut_idx", torch.tensor([], dtype=torch.long))
    mut_old_mut_idx = getattr(mut_joint_full_graph, "mut_idx", torch.tensor([], dtype=torch.long))

    if (
        len(mutations) > 0
        and wt_old_mut_idx.numel() < len(mutations)
        and mut_old_mut_idx.numel() > 0
    ):
        wt_patched_mut_idx = patch_wt_mut_idx_by_mut_graph_nn(
            wt_graph=wt_joint_full_graph,
            mut_graph=mut_joint_full_graph,
            mutations=mutations,
        )
        wt_joint_full_graph.mut_idx = wt_patched_mut_idx

    local_radius = float(GRAPH_CONFIG.get("local_radius", 12.0))
    local_topk_fallback = int(GRAPH_CONFIG.get("local_topk_fallback", 20))
    include_cross_partner_context = bool(GRAPH_CONFIG.get("include_cross_partner_context", True))

    # A6: full graph instead of local graph
    use_local_subgraph = bool(ABLATION_CONFIG.get("use_local_subgraph", True))

    if use_local_subgraph:
        wt_joint_graph = graph_builder.build_joint_local_graph(
            graph=wt_joint_full_graph,
            radius=local_radius,
            topk_fallback=local_topk_fallback,
            include_cross_partner_context=include_cross_partner_context,
        )

        mut_joint_graph = graph_builder.build_joint_local_graph(
            graph=mut_joint_full_graph,
            radius=local_radius,
            topk_fallback=local_topk_fallback,
            include_cross_partner_context=include_cross_partner_context,
        )
    else:
        wt_joint_graph = wt_joint_full_graph
        mut_joint_graph = mut_joint_full_graph

    sample = {
        "sample_id": sample_id,
        "wt_pdb_id": wt_pdb_id,
        "mut_pdb_id": mut_pdb_id,
        "mutation_str": mutation_str,
        "partners_str": partners_str,
        "mutations": mutations,
        "ab_chains": ab_chains,
        "ag_chains": ag_chains,
        "ab_muts": ab_muts,
        "ag_muts": ag_muts,
        "wt_joint_graph": wt_joint_graph,
        "mut_joint_graph": mut_joint_graph,
        "ddg": _to_tensor_label(ddg_val),
        "graph_version": f"joint_wt_mut_v1__ablation_{ABLATION_CONFIG.get('tag', 'full_model')}",
    }

    # 关键修复：补齐 WT-MUT 对齐索引、mutation pair、interface/contact pair。
    # 原始版本只保存两张图，导致 val/train 阶段这些字段为 None，
    # local mutation effect 和 contact delta 分支会退化成常数。
    sample = enrich_sample_with_alignment_and_pairs(sample)
    return sample


# =========================================================
# 读取 preprocess.py 生成的缓存索引
# =========================================================
def load_cache_index(cache_index_path: Path) -> pd.DataFrame:
    if not cache_index_path.exists():
        raise FileNotFoundError(
            f"Cache index not found: {cache_index_path}. "
            f"Please run preprocess.py first."
        )

    df = pd.read_csv(cache_index_path)
    if DEBUG_CONFIG.get("debug", False) and DEBUG_CONFIG.get("max_samples") is not None:
        df = df.iloc[: DEBUG_CONFIG["max_samples"]].copy()

    required_cols = {
        "row_idx", "sample_id", "wt_pdb_id", "mut_pdb_id",
        "mutation_str", "partners_str", "cache_file", "cache_path", "ddg"
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Cache index missing columns: {sorted(missing_cols)} | path={cache_index_path}"
        )

    logging.info(f"Loaded cache index: {cache_index_path}, total rows={len(df)}")
    return df


# =========================================================
# 数据集类（训练阶段纯读缓存）
# =========================================================
class MutationDataset(Dataset):
    def __init__(
        self,
        cache_index_path: Optional[Path] = None,
        sample_cache_dir: Optional[Path] = None,
        preload_in_memory: bool = True,
    ):
        self.sample_cache_dir = (
            Path(sample_cache_dir)
            if sample_cache_dir is not None
            else Path(DATA_CONFIG["sample_cache_dir"])
        )
        self.cache_index_path = (
            Path(cache_index_path)
            if cache_index_path is not None
            else self.sample_cache_dir / "cache_index.csv"
        )
        self.preload_in_memory = bool(preload_in_memory)

        self.index_df = load_cache_index(self.cache_index_path)
        self.records: List[Dict[str, Any]] = self.index_df.to_dict("records")
        self.data_list: Optional[List[Dict[str, Any]]] = None

        if self.preload_in_memory:
            self.data_list = []
            for idx, rec in enumerate(self.records):
                sample = self._load_sample_from_record(rec)
                self.data_list.append(sample)
                if (idx + 1) % 100 == 0 or (idx + 1) == len(self.records):
                    logging.info(
                        f"Preloaded {idx + 1}/{len(self.records)} cached samples"
                    )

        logging.info(
            f"MutationDataset ready. samples={len(self.records)}, "
            f"preload_in_memory={self.preload_in_memory}"
        )

    def _resolve_cache_path(self, record: Dict[str, Any]) -> Path:
        cache_file = _safe_str(record.get("cache_file"))
        cache_path_str = _safe_str(record.get("cache_path"))

        if cache_file:
            p = self.sample_cache_dir / cache_file
            if p.exists():
                return p

        if cache_path_str:
            p = Path(cache_path_str)
            if p.exists():
                return p

        raise FileNotFoundError(
            f"Cached sample not found for record: sample_id={record.get('sample_id')}, "
            f"cache_file={cache_file}, cache_path={cache_path_str}"
        )

    def _load_sample_from_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        sample_path = self._resolve_cache_path(record)
        return torch.load(sample_path, map_location="cpu")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.data_list is not None:
            return self.data_list[idx]
        return self._load_sample_from_record(self.records[idx])


# =========================================================
# 批处理
# 新版：只打包 WT/MUT 两张联合图
# =========================================================
def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(samples) == 0:
        raise ValueError("Empty batch in collate_fn")

    batch = {
        "sample_id": [s["sample_id"] for s in samples],
        "wt_pdb_id": [s["wt_pdb_id"] for s in samples],
        "mut_pdb_id": [s["mut_pdb_id"] for s in samples],
        "mutation_str": [s["mutation_str"] for s in samples],
        "partners_str": [s["partners_str"] for s in samples],
        "mutations": [s["mutations"] for s in samples],
        "ab_chains": [s["ab_chains"] for s in samples],
        "ag_chains": [s["ag_chains"] for s in samples],
        "ab_muts": [s["ab_muts"] for s in samples],
        "ag_muts": [s["ag_muts"] for s in samples],
        "wt_joint_graph": Batch.from_data_list([s["wt_joint_graph"] for s in samples]),
        "mut_joint_graph": Batch.from_data_list([s["mut_joint_graph"] for s in samples]),
        "ddg": torch.stack([s["ddg"] for s in samples], dim=0),
    }

    optional_passthrough_keys = [
        "graph_version",
        "aligned_wt_idx",
        "aligned_mut_idx",
        "mutation_shell_wt_idx",
        "mutation_shell_mut_idx",
        "interface_shell_wt_idx",
        "interface_shell_mut_idx",
        "mutation_aligned_wt_idx",
        "mutation_aligned_mut_idx",
        "interface_aligned_wt_idx",
        "interface_aligned_mut_idx",
        "all_pair_index",
        "all_pair_feat",
        "mutation_pair_index",
        "mutation_pair_feat",
        "interface_pair_index",
        "interface_pair_feat",
        "interface_contact_pair_index",
        "interface_contact_pair_feat",
    ]
    for key in optional_passthrough_keys:
        if any(key in s for s in samples):
            batch[key] = [s.get(key, None) for s in samples]

    return batch


# =========================================================
# 五折划分工具：从 fold_i/train.csv 或 valid.csv 读取 sample_id
# =========================================================
def load_split_csv(split_csv_path: Path) -> pd.DataFrame:
    if not split_csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv_path}")

    df = pd.read_csv(split_csv_path)
    if "sample_id" not in df.columns and COLUMN_CONFIG["mut_pdb_id"] not in df.columns:
        raise ValueError(
            f"Split file must contain 'sample_id' or '{COLUMN_CONFIG['mut_pdb_id']}'. "
            f"Got columns: {df.columns.tolist()}"
        )
    return df


# =========================================================
# 简单测试入口
# =========================================================
if __name__ == "__main__":
    dataset = MutationDataset(
        preload_in_memory=False,
    )
    print(f"Dataset size: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample keys:")
        for k in sample.keys():
            print(" ", k)

        print("ddg:", sample["ddg"])
        print("ab_chains:", sample["ab_chains"])
        print("ag_chains:", sample["ag_chains"])
        print("mutation_str:", sample["mutation_str"])

        if "wt_joint_graph" in sample:
            print("wt_joint_graph.x.shape:", sample["wt_joint_graph"].x.shape)
        if "mut_joint_graph" in sample:
            print("mut_joint_graph.x.shape:", sample["mut_joint_graph"].x.shape)
