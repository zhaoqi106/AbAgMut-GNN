from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
import esm
from config import DATA_CONFIG, FEATURE_CONFIG


# =========================================================
# 路径配置
# =========================================================
CSV_PATH = Path(DATA_CONFIG["csv_path"])

WT_PDB_DIR = Path(DATA_CONFIG["wt_pdb_dir"])
MUT_PDB_DIR = Path(DATA_CONFIG["mut_pdb_dir"])

WT_OUTPUT_DIR = Path(FEATURE_CONFIG["esm2_wt_dir"])
MUT_OUTPUT_DIR = Path(FEATURE_CONFIG["esm2_mut_dir"])

WT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MUT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 本地 ESM2 权重
LOCAL_ESM2_MODEL_PATH = Path("/home/zhao/esm2_t33_650M_UR50D.pt")

# 输出降维目标
TARGET_EMB_DIM = 128

# lazy init
_ESM_MODEL = None
_ESM_ALPHABET = None
_ESM_BATCH_CONVERTER = None
_DEVICE = None


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

def build_partners_maps() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    返回两个映射：
        wt_ag_map[pdb_id]  = antigen_chains
        mut_ag_map[pdb_id] = antigen_chains
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

        _, ag_chains = parse_partners_string(partners_str)
        if not ag_chains:
            continue

        if wt_pdb_id and wt_pdb_id not in wt_map:
            wt_map[wt_pdb_id] = ag_chains

        if mut_pdb_id and mut_pdb_id not in mut_map:
            mut_map[mut_pdb_id] = ag_chains

    return wt_map, mut_map


def collect_pdb_files_from_dir(pdb_dir: Path) -> List[Path]:
    return sorted(
        p for p in pdb_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdb"
    )

def reduce_embedding_dim(emb: torch.Tensor, target_dim: int = TARGET_EMB_DIM) -> torch.Tensor:
    """
    [L, D] -> [L, target_dim]
    使用自适应平均池化，和你当前 AntiBERTy.py 保持一致风格。
    """
    if not isinstance(emb, torch.Tensor):
        emb = torch.tensor(emb, dtype=torch.float32)

    emb = emb.float()

    if emb.dim() != 2:
        raise ValueError(f"Embedding tensor must be 2D [L, D], got shape={tuple(emb.shape)}")

    seq_len, in_dim = emb.shape

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
# ESM2 模型加载
# =========================================================
def get_esm2_runner():
    """
    lazy 初始化 ESM2 模型
    使用本地权重：
        /home/zhao/esm2_t33_650M_UR50D.pt
    """
    global _ESM_MODEL, _ESM_ALPHABET, _ESM_BATCH_CONVERTER, _DEVICE

    if _ESM_MODEL is not None:
        return _ESM_MODEL, _ESM_ALPHABET, _ESM_BATCH_CONVERTER, _DEVICE

    if not LOCAL_ESM2_MODEL_PATH.exists():
        raise FileNotFoundError(f"Local ESM2 model not found: {LOCAL_ESM2_MODEL_PATH}")


    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(LOCAL_ESM2_MODEL_PATH))
    batch_converter = alphabet.get_batch_converter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)

    _ESM_MODEL = model
    _ESM_ALPHABET = alphabet
    _ESM_BATCH_CONVERTER = batch_converter
    _DEVICE = device

    print(f"[INFO] Loaded local ESM2 model from: {LOCAL_ESM2_MODEL_PATH}", flush=True)
    print(f"[INFO] ESM2 device: {device}", flush=True)

    return _ESM_MODEL, _ESM_ALPHABET, _ESM_BATCH_CONVERTER, _DEVICE


# =========================================================
# 单链 embedding
# =========================================================
@torch.no_grad()
def generate_embedding(sequence: str, repr_layer: int = 33) -> torch.Tensor:
    """
    对单条序列生成 ESM2 embedding，并降到 128 维
    输出形状:
        [L, 128]
    """
    if not sequence:
        raise ValueError("Empty input sequence.")

    model, alphabet, batch_converter, device = get_esm2_runner()

    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
    token_representations = results["representations"][repr_layer]   # [1, L+2, D]

    # 去掉 BOS / EOS
    emb = token_representations[0, 1: len(sequence) + 1].detach().cpu().float()   # [L, D]

    emb = reduce_embedding_dim(emb, target_dim=TARGET_EMB_DIM)

    if emb.dim() != 2 or emb.size(1) != TARGET_EMB_DIM:
        raise RuntimeError(
            f"Reduced embedding has wrong shape: {tuple(emb.shape)}, expected [L, {TARGET_EMB_DIM}]"
        )

    return emb


# =========================================================
# 保存一个 pdb 的 antigen 链特征
# =========================================================
def save_embeddings_for_one_pdb(
    pdb_id: str,
    chain_info_dict: Dict[str, Dict[str, object]],
    antigen_chains: List[str],
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

    ag_set = set(antigen_chains)

    for chain_id, chain_info in chain_info_dict.items():
        if chain_id not in ag_set:
            continue

        seq = chain_info["sequence"]
        residue_keys = chain_info["residue_keys"]

        if not seq:
            print(f"[WARN] Empty sequence for antigen chain {chain_id} in PDB {pdb_id}", flush=True)
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
                f"[OK] PDB {pdb_id} antigen chain {chain_id}: seq_len={len(seq)} emb_shape={tuple(emb.shape)}",
                flush=True
            )
        except Exception as e:
            print(f"[ERROR] Failed ESM2 embedding for chain {chain_id} in PDB {pdb_id}: {e}", flush=True)

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
        print(f"[WARN] No valid antigen-chain embeddings generated for {out_path}", flush=True)

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

    antigen_chains = partners_map.get(pdb_id, [])
    if not antigen_chains:
        print(f"[WARN] No antigen chain definition found in CSV for {pdb_id}, skip.", flush=True)
        return

    print(f"[ESM2] {idx + 1}/{total} processing {pdb_path}", flush=True)

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
        antigen_chains=antigen_chains,
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
    print(f"[{name}] local esm2 model={LOCAL_ESM2_MODEL_PATH}", flush=True)
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
    wt_ag_map, mut_ag_map = build_partners_maps()

    # 默认建议重生成
    skip_existing = True

    run_for_one_side(
        name="WT",
        pdb_dir=WT_PDB_DIR,
        out_dir=WT_OUTPUT_DIR,
        partners_map=wt_ag_map,
        skip_existing=skip_existing,
    )

    run_for_one_side(
        name="MUT",
        pdb_dir=MUT_PDB_DIR,
        out_dir=MUT_OUTPUT_DIR,
        partners_map=mut_ag_map,
        skip_existing=skip_existing,
    )


if __name__ == "__main__":
    main()