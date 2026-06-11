
from pathlib import Path
from typing import Dict, List, Tuple

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.load.*weights_only=False.*"
)

import pandas as pd
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser, PPBuilder

from config import DATA_CONFIG, FEATURE_CONFIG


# =========================================================
# 路径配置
# =========================================================
CSV_PATH = Path(DATA_CONFIG["csv_path"])

WT_PDB_DIR = Path(DATA_CONFIG["wt_pdb_dir"])
MUT_PDB_DIR = Path(DATA_CONFIG["mut_pdb_dir"])

WT_OUTPUT_DIR = Path(FEATURE_CONFIG["antiberty_wt_dir"])
MUT_OUTPUT_DIR = Path(FEATURE_CONFIG["antiberty_mut_dir"])

WT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MUT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# AntiBERTy 输出降维目标
# 最终每个残基保存 128 维
# =========================================================
TARGET_EMB_DIM = 128

# Lazy 初始化，避免导入阶段碰 CUDA
_ANTIBERTY_RUNNER = None


# =========================================================
# 基础工具
# =========================================================
def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()

def normalize_icode(icode) -> str:
    if icode is None:
        return ""
    s = str(icode).strip()
    return s if s else ""


def parse_resseq_icode_from_residue_id(res_id):
    """
    尽量稳妥地从 Bio.PDB residue.id 中解析 (resseq, icode)
    正常情况:
        (' ', 58, ' ')
    异常情况:
        解析失败则返回 (None, None)
    """
    try:
        if isinstance(res_id, tuple):
            if len(res_id) >= 3:
                _, resseq, icode = res_id[:3]
                return int(resseq), normalize_icode(icode)
            if len(res_id) == 2:
                _, resseq = res_id
                return int(resseq), ""
        return None, None
    except Exception:
        return None, None

def parse_partners_string(partners_str: str) -> Tuple[List[str], List[str]]:
    """
    Partners 规则：
        下划线左边 = 抗体链
        下划线右边 = 抗原链
    例如:
        HL_A   -> antibody=['H','L'], antigen=['A']
        AB_CD  -> antibody=['A','B'], antigen=['C','D']
    """
    partners_str = safe_str(partners_str).replace(" ", "")
    if not partners_str or "_" not in partners_str:
        return [], []

    left, right = partners_str.split("_", 1)
    ab_chains = list(left) if left else []
    ag_chains = list(right) if right else []
    return ab_chains, ag_chains


def build_partners_maps() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    返回两个映射：
        wt_ab_map[pdb_id]  = antibody_chains
        mut_ab_map[pdb_id] = antibody_chains
    """
    df = pd.read_csv(CSV_PATH)

    wt_map: Dict[str, List[str]] = {}
    mut_map: Dict[str, List[str]] = {}

    wt_pdb_col = "PDB"
    mut_pdb_col = "ID"
    partners_col = "Partners"

    for _, row in df.iterrows():
        wt_pdb_id = safe_str(row.get(wt_pdb_col))
        mut_pdb_id = safe_str(row.get(mut_pdb_col))
        partners_str = safe_str(row.get(partners_col))

        ab_chains, _ = parse_partners_string(partners_str)
        if not ab_chains:
            continue

        if wt_pdb_id and wt_pdb_id not in wt_map:
            wt_map[wt_pdb_id] = ab_chains

        if mut_pdb_id and mut_pdb_id not in mut_map:
            mut_map[mut_pdb_id] = ab_chains

    return wt_map, mut_map


def collect_pdb_files_from_dir(pdb_dir: Path) -> List[Path]:
    return sorted(pdb_dir.glob("*.pdb"))

def extract_chain_sequences_with_keys_from_pdb(pdb_path: Path) -> Dict[str, Dict[str, object]]:
    """
    直接按结构里的标准氨基酸残基顺序提取，保证和图构建一致。
    返回:
    {
        "H": {
            "sequence": "QVQL...",
            "residue_keys": [("H", 1, ""), ("H", 2, ""), ...]
        }
    }
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = structure[0]

    aa3_to_aa1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
    }

    out: Dict[str, Dict[str, object]] = {}

    for chain in model:
        seq_chars = []
        residue_keys = []

        for res in chain:
            if res.id[0] != " ":
                continue
            if "CA" not in res:
                continue

            aa = aa3_to_aa1.get(str(res.get_resname()).upper(), "X")
            if aa == "X":
                continue

            resseq, icode = parse_resseq_icode_from_residue_id(res.id)
            if resseq is None:
                print(
                    f"[WARN] Skip residue with bad id in {pdb_path.name} | "
                    f"chain={chain.id} res.id={res.id}",
                    flush=True
                )
                continue

            seq_chars.append(aa)
            residue_keys.append((chain.id, resseq, icode))

        seq = "".join(seq_chars)
        if seq:
            out[chain.id] = {
                "sequence": seq,
                "residue_keys": residue_keys,
            }

    return out

def reduce_embedding_dim(emb: torch.Tensor, target_dim: int = TARGET_EMB_DIM) -> torch.Tensor:
    """
    [L, D] -> [L, target_dim]
    使用自适应平均池化，保持和你当前流程一致。
    """
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb, dtype=torch.float32)

    emb = emb.float()

    if emb.dim() != 2:
        raise ValueError(f"Embedding tensor must be 2D [L, D], got shape={tuple(emb.shape)}")

    _, in_dim = emb.shape

    if in_dim == target_dim:
        return emb.contiguous()

    if in_dim < target_dim:
        raise ValueError(
            f"Embedding dim {in_dim} is smaller than target_dim {target_dim}, cannot down-project by pooling."
        )

    emb = emb.unsqueeze(1)                     # [L, 1, D]
    emb = F.adaptive_avg_pool1d(emb, target_dim)
    emb = emb.squeeze(1).contiguous()         # [L, target_dim]
    return emb


# =========================================================
# AntiBERTy 模型加载
# =========================================================
def get_antiberty_runner():
    global _ANTIBERTY_RUNNER
    if _ANTIBERTY_RUNNER is None:
        from antiberty import AntiBERTyRunner
        _ANTIBERTY_RUNNER = AntiBERTyRunner()
    return _ANTIBERTY_RUNNER


# =========================================================
# 单链 embedding
# =========================================================
def generate_embedding(sequence: str, max_model_len: int = 512) -> torch.Tensor:
    """
    对单条序列生成 AntiBERTy embedding，并在保存前降到 128 维
    输出形状:
        [L, 128]
    """
    if not sequence:
        raise ValueError("Empty input sequence.")

    runner = get_antiberty_runner()

    chunk_size = max_model_len - 2
    embeddings: List[torch.Tensor] = []

    for start in range(0, len(sequence), chunk_size):
        chunk = sequence[start:start + chunk_size]
        emb_chunk = runner.embed([chunk])[0]

        if not isinstance(emb_chunk, torch.Tensor):
            emb_chunk = torch.tensor(emb_chunk, dtype=torch.float32)

        emb_chunk = emb_chunk.detach().cpu().float()

        # AntiBERTy 某些版本返回 [L+2, D]，首尾是特殊 token
        if emb_chunk.dim() != 2:
            raise ValueError(f"Unexpected AntiBERTy embedding shape: {tuple(emb_chunk.shape)}")

        if emb_chunk.size(0) == len(chunk) + 2:
            emb_chunk = emb_chunk[1:-1]
        elif emb_chunk.size(0) != len(chunk):
            raise ValueError(
                f"AntiBERTy embedding length mismatch before pooling: "
                f"emb_len={emb_chunk.size(0)} seq_len={len(chunk)}"
            )

        emb_chunk = reduce_embedding_dim(emb_chunk, target_dim=TARGET_EMB_DIM)
        embeddings.append(emb_chunk)

    if len(embeddings) == 0:
        raise ValueError("Empty embedding list generated.")

    emb = torch.cat(embeddings, dim=0)

    if emb.dim() != 2 or emb.size(1) != TARGET_EMB_DIM:
        raise RuntimeError(
            f"Reduced embedding has wrong shape: {tuple(emb.shape)}, expected [L, {TARGET_EMB_DIM}]"
        )

    return emb


# =========================================================
# 保存一个 pdb 的 antibody 链特征
# =========================================================
def save_embeddings_for_one_pdb(
    pdb_id: str,
    chain_info_dict: Dict[str, Dict[str, object]],
    antibody_chains: List[str],
    out_path: Path,
) -> None:
    """
    保存格式:
    {
        "chain_embeddings": {chain_id: emb[L, 128]},
        "residue_keys": {chain_id: [(chain_id, resseq, icode), ...]},
        "residue_embeddings": {(chain_id, resseq, icode): emb[128]}
    }
    """
    chain_embeddings: Dict[str, torch.Tensor] = {}
    residue_keys_map: Dict[str, List[Tuple[str, int, str]]] = {}
    residue_embeddings: Dict[Tuple[str, int, str], torch.Tensor] = {}

    ab_set = set(antibody_chains)

    for chain_id, chain_info in chain_info_dict.items():
        if chain_id not in ab_set:
            continue

        seq = chain_info["sequence"]
        residue_keys = chain_info["residue_keys"]

        if not seq:
            print(f"[WARN] Empty sequence for antibody chain {chain_id} in PDB {pdb_id}", flush=True)
            continue

        try:
            emb = generate_embedding(seq)   # [L, 128]

            if emb.size(0) != len(residue_keys):
                print(
                    f"[WARN] Length mismatch in PDB {pdb_id} chain {chain_id}: "
                    f"emb_len={emb.size(0)} residue_keys={len(residue_keys)}",
                    flush=True
                )
                min_len = min(emb.size(0), len(residue_keys))
                emb = emb[:min_len]
                residue_keys = residue_keys[:min_len]

            chain_embeddings[chain_id] = emb
            residue_keys_map[chain_id] = residue_keys

            for i, key in enumerate(residue_keys):
                residue_embeddings[key] = emb[i]

            print(
                f"[OK] PDB {pdb_id} antibody chain {chain_id}: seq_len={len(seq)} emb_shape={tuple(emb.shape)}",
                flush=True
            )
        except Exception as e:
            print(f"[ERROR] Failed AntiBERTy embedding for chain {chain_id} in PDB {pdb_id}: {e}", flush=True)

    if chain_embeddings:
        torch.save(
            {
                "chain_embeddings": chain_embeddings,
                "residue_keys": residue_keys_map,
                "residue_embeddings": residue_embeddings,
            },
            out_path,
        )
        print(f"[OK] Saved {out_path}", flush=True)
    else:
        print(f"[WARN] No valid antibody-chain embeddings generated for {out_path}", flush=True)

# =========================================================
# 单个 PDB 处理
# =========================================================
def process_one_pdb(
    pdb_path: Path,
    output_dir: Path,
    partners_map: Dict[str, List[str]],
    idx: int,
    total: int,
    skip_existing: bool,
) -> None:
    pdb_path = Path(pdb_path)
    output_dir = Path(output_dir)
    pdb_id = pdb_path.stem

    out_path = output_dir / f"{pdb_id}.pt"

    if skip_existing and out_path.exists():
        print(f"[SKIP] {idx + 1}/{total} {pdb_id}", flush=True)
        return

    if not pdb_path.exists():
        print(f"[WARN] PDB not found: {pdb_path}", flush=True)
        return

    antibody_chains = partners_map.get(pdb_id, [])
    if not antibody_chains:
        print(f"[WARN] No antibody chain definition found in CSV for {pdb_id}, skip.", flush=True)
        return

    print(f"[AntiBERTy] {idx + 1}/{total} processing {pdb_path}", flush=True)

    try:
        chain_info_dict = extract_chain_sequences_with_keys_from_pdb(pdb_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse PDB {pdb_path}: {e}", flush=True)
        return

    if not chain_info_dict:
        print(f"[WARN] No valid protein chains found in {pdb_path}", flush=True)
        return

    save_embeddings_for_one_pdb(
        pdb_id=pdb_id,
        chain_info_dict=chain_info_dict,
        antibody_chains=antibody_chains,
        out_path=out_path,
    )

# =========================================================
# 批处理一侧
# =========================================================
def run_for_one_side(
    name: str,
    pdb_dir: Path,
    out_dir: Path,
    partners_map: Dict[str, List[str]],
    skip_existing: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = collect_pdb_files_from_dir(pdb_dir)
    total = len(pdb_files)

    print("=" * 80, flush=True)
    print(f"[{name}] csv={CSV_PATH}", flush=True)
    print(f"[{name}] pdb_dir={pdb_dir}", flush=True)
    print(f"[{name}] out_dir={out_dir}", flush=True)
    print(f"[{name}] total pdb files={total}", flush=True)
    print(f"[{name}] target embedding dim={TARGET_EMB_DIM}", flush=True)
    print("=" * 80, flush=True)

    if total == 0:
        print(f"[{name}] no pdb files found, skip.", flush=True)
        return

    for idx, pdb_path in enumerate(pdb_files):
        process_one_pdb(
            pdb_path=pdb_path,
            output_dir=out_dir,
            partners_map=partners_map,
            idx=idx,
            total=total,
            skip_existing=skip_existing,
        )


# =========================================================
# 主程序
# =========================================================
def main():
    wt_ab_map, mut_ab_map = build_partners_maps()

    # 默认建议重生成
    skip_existing = True

    run_for_one_side(
        name="WT",
        pdb_dir=WT_PDB_DIR,
        out_dir=WT_OUTPUT_DIR,
        partners_map=wt_ab_map,
        skip_existing=skip_existing,
    )

    run_for_one_side(
        name="MUT",
        pdb_dir=MUT_PDB_DIR,
        out_dir=MUT_OUTPUT_DIR,
        partners_map=mut_ab_map,
        skip_existing=skip_existing,
    )


if __name__ == "__main__":
    main()
