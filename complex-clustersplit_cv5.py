import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
from Bio.PDB import PDBParser

from config import DATA_CONFIG, COLUMN_CONFIG, TRAIN_CONFIG, PARTNER_CONFIG


# =====================================
# Config
# =====================================
CSV_PATH = Path(DATA_CONFIG["csv_path"])

# New split root. Prefer explicit config if you add it; otherwise default to dataset_dir / complexsplits.
OUTPUT_DIR = Path(DATA_CONFIG.get("complex_split_dir", Path(DATA_CONFIG["dataset_dir"]) / "complexsplits"))

WT_PDB_DIR = Path(DATA_CONFIG["wt_pdb_dir"])

PDB_COL = COLUMN_CONFIG["wt_pdb_id"]      # default "PDB"
ID_COL = COLUMN_CONFIG["mut_pdb_id"]      # default "ID"
LABEL_COL = COLUMN_CONFIG["label"]        # default "ddG"
MUT_COL = COLUMN_CONFIG["mutation"]       # default "Mutation"
PARTNERS_COL = COLUMN_CONFIG.get("partners", "Partners")

N_FOLDS = int(TRAIN_CONFIG.get("num_folds", 5))
SEED = int(TRAIN_CONFIG.get("seed", 42))

# Sequence identity thresholds for clustering.
# You can override these in TRAIN_CONFIG if needed.
AB_IDENTITY_THRESHOLD = float(TRAIN_CONFIG.get("complexsplit_ab_identity", 0.50))
AG_IDENTITY_THRESHOLD = float(TRAIN_CONFIG.get("complexsplit_ag_identity", 0.50))

# Search parameters, same style as split_dataset_cv5.py.
NUM_RESTARTS = int(TRAIN_CONFIG.get("complexsplit_num_restarts", 300))
MAX_MOVE_PASSES = int(TRAIN_CONFIG.get("complexsplit_max_move_passes", 40))
MAX_SWAP_PASSES = int(TRAIN_CONFIG.get("complexsplit_max_swap_passes", 25))

# Dominance control. Soft constraints, not hard constraints.
MAX_TOP1_RATIO = float(TRAIN_CONFIG.get("complexsplit_max_top1_ratio", 0.25))
MAX_TOP2_RATIO = float(TRAIN_CONFIG.get("complexsplit_max_top2_ratio", 0.45))
DOMINANCE_PENALTY_TOP1 = float(TRAIN_CONFIG.get("complexsplit_dominance_penalty_top1", 40.0))
DOMINANCE_PENALTY_TOP2 = float(TRAIN_CONFIG.get("complexsplit_dominance_penalty_top2", 30.0))
DOMINANCE_PENALTY_STD = float(TRAIN_CONFIG.get("complexsplit_dominance_penalty_std", 10.0))

GROUP_COL = "complex_cluster"
AB_CLUSTER_COL = "ab_cluster"
AG_CLUSTER_COL = "ag_cluster"
AB_SEQ_COL = "ab_sequence"
AG_SEQ_COL = "ag_sequence"


# =====================================
# Basic utilities
# =====================================
def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def count_mutations(mutation_str: str) -> int:
    if pd.isna(mutation_str):
        return 0
    s = str(mutation_str).strip()
    if not s:
        return 0
    return len([x.strip() for x in s.replace(";", ",").split(",") if x.strip()])


def zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    mean = float(series.mean())
    if std < 1e-12:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - mean) / std


def make_sample_id(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [f"{idx:06d}_{mid}" for idx, mid in zip(df.index.tolist(), df[ID_COL].astype(str).tolist())],
        index=df.index,
    )


def sanitize_cluster_text(x: Any) -> str:
    s = safe_str(x)
    if not s:
        return "UNK"
    for old, new in [
        ("/", "_"), ("\\", "_"), (":", "_"), ("*", "_"),
        ("?", "_"), ('"', "_"), ("<", "_"), (">", "_"),
        ("|", "_"), (" ", ""), (";", "_"), (",", "_"),
    ]:
        s = s.replace(old, new)
    return s or "UNK"


# =====================================
# PDB sequence extraction
# =====================================
AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def parse_partners_string(partners_str: str) -> Tuple[List[str], List[str]]:
    partners_str = safe_str(partners_str).replace(" ", "")
    if not partners_str:
        raise ValueError("Empty Partners field")

    sep = str(PARTNER_CONFIG.get("group_sep", "_"))
    if sep not in partners_str:
        raise ValueError(f"Partners field must contain '{sep}': {partners_str}")

    left, right = partners_str.split(sep, 1)
    if not left or not right:
        raise ValueError(f"Invalid Partners field: {partners_str}")

    return list(left), list(right)


def resolve_pdb_path(pdb_id: str, pdb_dir: Path) -> Optional[Path]:
    pdb_id = safe_str(pdb_id)
    if not pdb_id:
        return None

    direct = pdb_dir / f"{pdb_id}.pdb"
    if direct.exists():
        return direct

    target = f"{pdb_id}.pdb".lower()
    if pdb_dir.exists():
        for p in pdb_dir.iterdir():
            if p.is_file() and p.name.lower() == target:
                return p
    return None


def extract_chain_sequences_from_pdb(pdb_path: Path) -> Dict[str, str]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    model = structure[0]

    chain_to_seq: Dict[str, str] = {}
    for chain in model:
        chars: List[str] = []
        for res in chain:
            if res.id[0] != " ":
                continue
            if "CA" not in res:
                continue
            aa = AA3_TO_AA1.get(str(res.get_resname()).upper(), "X")
            if aa != "X":
                chars.append(aa)
        if chars:
            chain_to_seq[str(chain.id)] = "".join(chars)
    return chain_to_seq


def build_side_sequences_for_row(row: pd.Series, chain_seq_cache: Dict[str, Dict[str, str]]) -> Tuple[str, str]:
    pdb_id = safe_str(row[PDB_COL])
    partners = safe_str(row[PARTNERS_COL])
    ab_chains, ag_chains = parse_partners_string(partners)

    if pdb_id not in chain_seq_cache:
        pdb_path = resolve_pdb_path(pdb_id, WT_PDB_DIR)
        if pdb_path is None:
            raise FileNotFoundError(f"WT PDB not found for '{pdb_id}' under {WT_PDB_DIR}")
        chain_seq_cache[pdb_id] = extract_chain_sequences_from_pdb(pdb_path)

    seq_map = chain_seq_cache[pdb_id]
    missing_ab = [ch for ch in ab_chains if ch not in seq_map]
    missing_ag = [ch for ch in ag_chains if ch not in seq_map]
    if missing_ab or missing_ag:
        raise KeyError(
            f"Missing chains in WT PDB {pdb_id}: missing_ab={missing_ab}, missing_ag={missing_ag}, "
            f"available={sorted(seq_map.keys())}, Partners={partners}"
        )

    # Sort by chain order in Partners, not alphabetically, to keep deterministic biological side definition.
    ab_seq = "".join(seq_map[ch] for ch in ab_chains)
    ag_seq = "".join(seq_map[ch] for ch in ag_chains)
    if not ab_seq:
        raise ValueError(f"Empty antibody-side sequence for PDB={pdb_id}, Partners={partners}")
    if not ag_seq:
        raise ValueError(f"Empty antigen-side sequence for PDB={pdb_id}, Partners={partners}")
    return ab_seq, ag_seq


# =====================================
# Pure-Python sequence clustering
# =====================================
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def simple_sequence_identity(seq1: str, seq2: str) -> float:
    """
    Fast conservative identity without external alignment.
    For unequal lengths, denominator is max length and missing positions are mismatches.
    This prevents short fragments from matching long sequences too easily.
    """
    s1 = safe_str(seq1).upper()
    s2 = safe_str(seq2).upper()
    if not s1 or not s2:
        return 0.0
    denom = max(len(s1), len(s2))
    same = 0
    for a, b in zip(s1, s2):
        if a == b:
            same += 1
    return float(same) / float(denom)


def cluster_sequences_by_identity(seq_by_key: Dict[str, str], threshold: float, prefix: str) -> Dict[str, str]:
    keys = sorted(seq_by_key.keys())
    n = len(keys)
    if n == 0:
        return {}
    uf = UnionFind(n)

    for i in range(n):
        si = seq_by_key[keys[i]]
        for j in range(i + 1, n):
            sj = seq_by_key[keys[j]]
            if simple_sequence_identity(si, sj) >= threshold:
                uf.union(i, j)

    root_to_members: Dict[int, List[int]] = {}
    for i in range(n):
        root = uf.find(i)
        root_to_members.setdefault(root, []).append(i)

    # Stable cluster ids: larger clusters first, then lexicographic first key.
    clusters = sorted(root_to_members.values(), key=lambda idxs: (-len(idxs), keys[min(idxs)]))
    key_to_cluster: Dict[str, str] = {}
    for cid, idxs in enumerate(clusters, start=1):
        cluster_id = f"{prefix}{cid:04d}"
        for idx in idxs:
            key_to_cluster[keys[idx]] = cluster_id
    return key_to_cluster


# =====================================
# Build complex cluster table
# =====================================
def add_complex_clusters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    chain_seq_cache: Dict[str, Dict[str, str]] = {}

    # One biological side-sequence per row. Later rows sharing identical side sequence
    # naturally map to the same sequence key and cluster.
    ab_seqs: List[str] = []
    ag_seqs: List[str] = []
    failed_rows: List[Tuple[int, str]] = []

    for idx, row in df.iterrows():
        try:
            ab_seq, ag_seq = build_side_sequences_for_row(row, chain_seq_cache)
        except Exception as e:
            failed_rows.append((idx, str(e)))
            ab_seq, ag_seq = "", ""
        ab_seqs.append(ab_seq)
        ag_seqs.append(ag_seq)

    df[AB_SEQ_COL] = ab_seqs
    df[AG_SEQ_COL] = ag_seqs

    if failed_rows:
        failed_path = OUTPUT_DIR / "complex_cluster_failed_rows.csv"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failed_rows, columns=["row_idx", "error"]).to_csv(
            failed_path, index=False, encoding="utf-8-sig"
        )
        first = failed_rows[:10]
        raise RuntimeError(
            f"Failed to build side sequences for {len(failed_rows)} rows. "
            f"First errors: {first}. Saved: {failed_path}"
        )

    # Use sequence itself as cluster key source after deduplication.
    ab_unique = {f"ABSEQ_{i:06d}": seq for i, seq in enumerate(sorted(set(ab_seqs)), start=1)}
    ag_unique = {f"AGSEQ_{i:06d}": seq for i, seq in enumerate(sorted(set(ag_seqs)), start=1)}
    ab_seq_to_key = {seq: key for key, seq in ab_unique.items()}
    ag_seq_to_key = {seq: key for key, seq in ag_unique.items()}

    ab_key_to_cluster = cluster_sequences_by_identity(
        seq_by_key=ab_unique,
        threshold=AB_IDENTITY_THRESHOLD,
        prefix="AB",
    )
    ag_key_to_cluster = cluster_sequences_by_identity(
        seq_by_key=ag_unique,
        threshold=AG_IDENTITY_THRESHOLD,
        prefix="AG",
    )

    df[AB_CLUSTER_COL] = [ab_key_to_cluster[ab_seq_to_key[seq]] for seq in df[AB_SEQ_COL]]
    df[AG_CLUSTER_COL] = [ag_key_to_cluster[ag_seq_to_key[seq]] for seq in df[AG_SEQ_COL]]
    df[GROUP_COL] = df[AB_CLUSTER_COL] + "__" + df[AG_CLUSTER_COL]

    # Keep a compact table for inspection.
    cluster_table = (
        df[[PDB_COL, ID_COL, PARTNERS_COL, AB_CLUSTER_COL, AG_CLUSTER_COL, GROUP_COL, AB_SEQ_COL, AG_SEQ_COL]]
        .drop_duplicates()
        .sort_values([GROUP_COL, PDB_COL, ID_COL])
        .reset_index(drop=True)
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cluster_table.to_csv(OUTPUT_DIR / "complex_cluster_table.csv", index=False, encoding="utf-8-sig")

    return df


# =====================================
# Group statistics and fold optimizer
# =====================================
def build_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_id, sub_df in df.groupby(GROUP_COL):
        n_samples = len(sub_df)
        ddg_mean = float(sub_df[LABEL_COL].mean()) if n_samples > 0 else 0.0
        ddg_std = float(sub_df[LABEL_COL].std(ddof=0)) if n_samples > 0 else 0.0
        mut_counts = sub_df[MUT_COL].apply(count_mutations)
        single_ratio = float((mut_counts == 1).mean()) if n_samples > 0 else 0.0
        multi_ratio = float((mut_counts > 1).mean()) if n_samples > 0 else 0.0

        rows.append({
            GROUP_COL: str(group_id),
            "n_samples": int(n_samples),
            "n_pdbs": int(sub_df[PDB_COL].nunique()),
            "n_ab_clusters": int(sub_df[AB_CLUSTER_COL].nunique()),
            "n_ag_clusters": int(sub_df[AG_CLUSTER_COL].nunique()),
            "ddg_mean": ddg_mean,
            "ddg_std": ddg_std,
            "single_ratio": single_ratio,
            "multi_ratio": multi_ratio,
            "pdb_list": ";".join(sorted(sub_df[PDB_COL].astype(str).unique())),
        })

    stats_df = pd.DataFrame(rows)
    if len(stats_df) == 0:
        raise ValueError("No complex_cluster groups found in input CSV.")
    return stats_df


def make_empty_fold() -> Dict[str, Any]:
    return {
        "groups": [],
        "n_samples": 0,
        "n_groups": 0,
        "ddg_sum": 0.0,
        "single_sum": 0.0,
        "multi_sum": 0.0,
        "group_sizes": [],
    }


def add_group_to_fold(fold_state: Dict[str, Any], row: pd.Series) -> None:
    fold_state["groups"].append(str(row[GROUP_COL]))
    fold_state["n_samples"] += int(row["n_samples"])
    fold_state["n_groups"] += 1
    fold_state["ddg_sum"] += float(row["ddg_mean"])
    fold_state["single_sum"] += float(row["single_ratio"])
    fold_state["multi_sum"] += float(row["multi_ratio"])
    fold_state["group_sizes"].append(int(row["n_samples"]))


def remove_group_from_fold(fold_state: Dict[str, Any], row: pd.Series) -> None:
    group_id = str(row[GROUP_COL])
    group_size = int(row["n_samples"])
    fold_state["groups"].remove(group_id)
    fold_state["n_samples"] -= group_size
    fold_state["n_groups"] -= 1
    fold_state["ddg_sum"] -= float(row["ddg_mean"])
    fold_state["single_sum"] -= float(row["single_ratio"])
    fold_state["multi_sum"] -= float(row["multi_ratio"])
    fold_state["group_sizes"].remove(group_size)


def fold_avg(sum_value: float, n_groups: int) -> float:
    if n_groups <= 0:
        return 0.0
    return float(sum_value) / float(n_groups)


def compute_fold_dominance(fold_state: Dict[str, Any]) -> Dict[str, float]:
    n_samples = int(fold_state["n_samples"])
    sizes = sorted([int(x) for x in fold_state.get("group_sizes", [])], reverse=True)
    if n_samples <= 0 or not sizes:
        return {"top1_ratio": 0.0, "top2_ratio": 0.0, "group_size_std": 0.0}
    top1_ratio = float(sizes[0]) / float(n_samples)
    top2_ratio = float(sum(sizes[:2])) / float(n_samples)
    if len(sizes) <= 1:
        group_size_std = 0.0
    else:
        mean_size = float(sum(sizes)) / float(len(sizes))
        var = float(sum((x - mean_size) ** 2 for x in sizes)) / float(len(sizes))
        group_size_std = math.sqrt(var)
    return {"top1_ratio": top1_ratio, "top2_ratio": top2_ratio, "group_size_std": group_size_std}


def build_global_targets(stats_df: pd.DataFrame, n_folds: int) -> Dict[str, float]:
    return {
        "target_samples": float(stats_df["n_samples"].sum()) / float(n_folds),
        "target_groups": float(len(stats_df)) / float(n_folds),
        "target_ddg": float(stats_df["ddg_mean"].mean()),
        "target_single": float(stats_df["single_ratio"].mean()),
        "target_multi": float(stats_df["multi_ratio"].mean()),
        "max_top1_ratio": MAX_TOP1_RATIO,
        "max_top2_ratio": MAX_TOP2_RATIO,
        "largest_group": int(stats_df["n_samples"].max()),
    }


def compute_total_loss(folds: Dict[int, Dict[str, Any]], targets: Dict[str, float]) -> float:
    loss = 0.0
    top1_ratios = []
    top2_ratios = []

    for _, f in folds.items():
        n_groups = int(f["n_groups"])
        n_samples = int(f["n_samples"])
        if n_groups <= 0:
            return 1e18

        ddg_avg = fold_avg(f["ddg_sum"], n_groups)
        single_avg = fold_avg(f["single_sum"], n_groups)
        multi_avg = fold_avg(f["multi_sum"], n_groups)

        sample_dev = abs(n_samples - targets["target_samples"]) / max(targets["target_samples"], 1e-8)
        group_dev = abs(n_groups - targets["target_groups"]) / max(targets["target_groups"], 1e-8)
        ddg_dev = abs(ddg_avg - targets["target_ddg"])
        single_dev = abs(single_avg - targets["target_single"])
        multi_dev = abs(multi_avg - targets["target_multi"])

        dominance = compute_fold_dominance(f)
        top1_ratio = dominance["top1_ratio"]
        top2_ratio = dominance["top2_ratio"]
        top1_ratios.append(top1_ratio)
        top2_ratios.append(top2_ratio)

        top1_excess = max(0.0, top1_ratio - targets["max_top1_ratio"])
        top2_excess = max(0.0, top2_ratio - targets["max_top2_ratio"])

        fold_loss = (
            8.0 * sample_dev +
            3.0 * group_dev +
            1.2 * ddg_dev +
            0.8 * single_dev +
            0.8 * multi_dev +
            DOMINANCE_PENALTY_TOP1 * (top1_excess ** 2) +
            DOMINANCE_PENALTY_TOP2 * (top2_excess ** 2)
        )
        if n_groups == 1:
            fold_loss += 8.0
        elif n_groups == 2:
            fold_loss += 3.0
        loss += fold_loss

    if len(top1_ratios) > 1:
        loss += DOMINANCE_PENALTY_STD * float(pd.Series(top1_ratios).std(ddof=0))
    if len(top2_ratios) > 1:
        loss += (DOMINANCE_PENALTY_STD * 0.8) * float(pd.Series(top2_ratios).std(ddof=0))
    return float(loss)


def build_priority_stats(stats_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    work_df = stats_df.copy()
    work_df["n_samples_z"] = zscore(work_df["n_samples"]).abs()
    work_df["ddg_mean_z"] = zscore(work_df["ddg_mean"]).abs()
    work_df["ddg_std_z"] = zscore(work_df["ddg_std"]).abs()
    work_df["single_ratio_z"] = zscore(work_df["single_ratio"]).abs()
    work_df["multi_ratio_z"] = zscore(work_df["multi_ratio"]).abs()
    work_df["priority"] = (
        5.0 * work_df["n_samples_z"] +
        1.5 * work_df["ddg_mean_z"] +
        1.0 * work_df["ddg_std_z"] +
        1.0 * work_df["single_ratio_z"] +
        1.0 * work_df["multi_ratio_z"]
    )
    work_df = work_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return work_df.sort_values(by=["priority", "n_samples"], ascending=[False, False]).reset_index(drop=True)


def greedy_initialize_assignment(stats_df: pd.DataFrame, n_folds: int, seed: int) -> Dict[int, List[str]]:
    rng = random.Random(seed)
    work_df = build_priority_stats(stats_df, seed=seed)
    targets = build_global_targets(work_df, n_folds=n_folds)
    folds = {i: make_empty_fold() for i in range(n_folds)}
    rows = [row for _, row in work_df.iterrows()]
    if len(rows) < n_folds:
        raise ValueError(f"Unique complex clusters ({len(rows)}) is smaller than n_folds ({n_folds})")

    first_rows = sorted(rows[:n_folds], key=lambda r: int(r["n_samples"]), reverse=True)
    remaining_rows = rows[n_folds:]
    fold_order = list(range(n_folds))
    rng.shuffle(fold_order)
    for fold_idx, row in zip(fold_order, first_rows):
        add_group_to_fold(folds[fold_idx], row)

    for row in remaining_rows:
        candidate_scores: List[Tuple[float, float, float, float, int]] = []
        for fold_idx in range(n_folds):
            add_group_to_fold(folds[fold_idx], row)
            total_loss = compute_total_loss(folds, targets)
            dominance = compute_fold_dominance(folds[fold_idx])
            remove_group_from_fold(folds[fold_idx], row)
            candidate_scores.append((
                total_loss,
                dominance["top1_ratio"],
                folds[fold_idx]["n_samples"],
                folds[fold_idx]["n_groups"],
                fold_idx,
            ))
        candidate_scores.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        best_loss = candidate_scores[0][0]
        best_choices = [x for x in candidate_scores if abs(x[0] - best_loss) < 1e-12]
        chosen = rng.choice(best_choices)[4]
        add_group_to_fold(folds[chosen], row)

    return {i: list(folds[i]["groups"]) for i in range(n_folds)}


def assignment_to_folds(stats_df: pd.DataFrame, assignment: Dict[int, List[str]], n_folds: int) -> Dict[int, Dict[str, Any]]:
    row_map = {str(row[GROUP_COL]): row for _, row in stats_df.iterrows()}
    folds = {i: make_empty_fold() for i in range(n_folds)}
    for fold_idx, group_ids in assignment.items():
        for group_id in group_ids:
            add_group_to_fold(folds[fold_idx], row_map[group_id])
    return folds


def compute_assignment_loss(stats_df: pd.DataFrame, assignment: Dict[int, List[str]], n_folds: int) -> float:
    folds = assignment_to_folds(stats_df, assignment, n_folds)
    targets = build_global_targets(stats_df, n_folds)
    return compute_total_loss(folds, targets)


def try_improve_by_move(stats_df: pd.DataFrame, assignment: Dict[int, List[str]], n_folds: int) -> Tuple[Dict[int, List[str]], bool]:
    row_map = {str(row[GROUP_COL]): row for _, row in stats_df.iterrows()}
    targets = build_global_targets(stats_df, n_folds)
    folds = assignment_to_folds(stats_df, assignment, n_folds)
    best_loss = compute_total_loss(folds, targets)
    best_move = None

    all_groups_sorted = sorted(row_map.keys(), key=lambda x: row_map[x]["n_samples"], reverse=True)
    group_to_fold = {}
    for fold_idx, group_ids in assignment.items():
        for group_id in group_ids:
            group_to_fold[group_id] = fold_idx

    for group_id in all_groups_sorted:
        src = group_to_fold[group_id]
        row = row_map[group_id]
        if len(assignment[src]) <= 1:
            continue
        for dst in range(n_folds):
            if dst == src:
                continue
            remove_group_from_fold(folds[src], row)
            add_group_to_fold(folds[dst], row)
            new_loss = compute_total_loss(folds, targets)
            remove_group_from_fold(folds[dst], row)
            add_group_to_fold(folds[src], row)
            if new_loss + 1e-12 < best_loss:
                best_loss = new_loss
                best_move = (group_id, src, dst)

    if best_move is None:
        return assignment, False
    group_id, src, dst = best_move
    new_assignment = {k: list(v) for k, v in assignment.items()}
    new_assignment[src].remove(group_id)
    new_assignment[dst].append(group_id)
    return new_assignment, True


def try_improve_by_swap(stats_df: pd.DataFrame, assignment: Dict[int, List[str]], n_folds: int) -> Tuple[Dict[int, List[str]], bool]:
    row_map = {str(row[GROUP_COL]): row for _, row in stats_df.iterrows()}
    targets = build_global_targets(stats_df, n_folds)
    folds = assignment_to_folds(stats_df, assignment, n_folds)
    best_loss = compute_total_loss(folds, targets)
    best_swap = None

    fold_sorted_groups = {
        fold_idx: sorted(group_ids, key=lambda x: row_map[x]["n_samples"], reverse=True)
        for fold_idx, group_ids in assignment.items()
    }

    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            for gi in fold_sorted_groups[i]:
                row_i = row_map[gi]
                for gj in fold_sorted_groups[j]:
                    row_j = row_map[gj]
                    remove_group_from_fold(folds[i], row_i)
                    remove_group_from_fold(folds[j], row_j)
                    add_group_to_fold(folds[i], row_j)
                    add_group_to_fold(folds[j], row_i)
                    new_loss = compute_total_loss(folds, targets)
                    remove_group_from_fold(folds[i], row_j)
                    remove_group_from_fold(folds[j], row_i)
                    add_group_to_fold(folds[i], row_i)
                    add_group_to_fold(folds[j], row_j)
                    if new_loss + 1e-12 < best_loss:
                        best_loss = new_loss
                        best_swap = (gi, i, gj, j)

    if best_swap is None:
        return assignment, False
    gi, i, gj, j = best_swap
    new_assignment = {k: list(v) for k, v in assignment.items()}
    new_assignment[i].remove(gi)
    new_assignment[j].remove(gj)
    new_assignment[i].append(gj)
    new_assignment[j].append(gi)
    return new_assignment, True


def optimize_assignment(
    stats_df: pd.DataFrame,
    n_folds: int,
    seed: int,
    num_restarts: int = NUM_RESTARTS,
    max_move_passes: int = MAX_MOVE_PASSES,
    max_swap_passes: int = MAX_SWAP_PASSES,
) -> Dict[int, List[str]]:
    best_assignment = None
    best_loss = float("inf")
    for restart_idx in range(num_restarts):
        cur_seed = seed + restart_idx * 9973
        assignment = greedy_initialize_assignment(stats_df=stats_df, n_folds=n_folds, seed=cur_seed)
        for _ in range(max_move_passes):
            assignment, improved = try_improve_by_move(stats_df, assignment, n_folds)
            if not improved:
                break
        for _ in range(max_swap_passes):
            assignment, improved = try_improve_by_swap(stats_df, assignment, n_folds)
            if not improved:
                break
        for _ in range(max_move_passes):
            assignment, improved = try_improve_by_move(stats_df, assignment, n_folds)
            if not improved:
                break
        loss = compute_assignment_loss(stats_df, assignment, n_folds)
        if loss < best_loss:
            best_loss = loss
            best_assignment = assignment
        if (restart_idx + 1) % 20 == 0 or restart_idx == 0 or (restart_idx + 1) == num_restarts:
            print(f"[Search] restart {restart_idx + 1}/{num_restarts} | current_best_loss={best_loss:.6f}", flush=True)

    if best_assignment is None:
        raise RuntimeError("Failed to produce a valid fold assignment.")
    for fold_idx in range(n_folds):
        if len(best_assignment.get(fold_idx, [])) == 0:
            raise RuntimeError(f"Fold {fold_idx + 1} is empty after optimization.")
    return best_assignment


# =====================================
# Save outputs
# =====================================
def build_valid_fold_summary(valid_df: pd.DataFrame) -> Dict[str, Any]:
    if len(valid_df) == 0:
        return {
            "valid_samples": 0,
            "valid_complex_groups": 0,
            "valid_pdb_groups": 0,
            "valid_ab_clusters": 0,
            "valid_ag_clusters": 0,
            "valid_ddg_mean": 0.0,
            "valid_ddg_std": 0.0,
            "single_ratio": 0.0,
            "multi_ratio": 0.0,
            "top1_group": "",
            "top1_group_ratio": 0.0,
            "top2_group_ratio": 0.0,
            "top3_groups": "",
        }
    mut_counts = valid_df[MUT_COL].apply(count_mutations)
    group_counts = valid_df[GROUP_COL].astype(str).value_counts()
    top1_group = str(group_counts.index[0]) if len(group_counts) > 0 else ""
    top1_ratio = float(group_counts.iloc[0]) / float(len(valid_df)) if len(group_counts) > 0 else 0.0
    top2_ratio = float(group_counts.iloc[:2].sum()) / float(len(valid_df)) if len(group_counts) > 0 else 0.0
    top3_text = "; ".join([f"{k}:{v}" for k, v in group_counts.head(3).items()])
    return {
        "valid_samples": int(len(valid_df)),
        "valid_complex_groups": int(valid_df[GROUP_COL].nunique()),
        "valid_pdb_groups": int(valid_df[PDB_COL].nunique()),
        "valid_ab_clusters": int(valid_df[AB_CLUSTER_COL].nunique()),
        "valid_ag_clusters": int(valid_df[AG_CLUSTER_COL].nunique()),
        "valid_ddg_mean": float(valid_df[LABEL_COL].mean()),
        "valid_ddg_std": float(valid_df[LABEL_COL].std(ddof=0)) if len(valid_df) > 0 else 0.0,
        "single_ratio": float((mut_counts == 1).mean()),
        "multi_ratio": float((mut_counts > 1).mean()),
        "top1_group": top1_group,
        "top1_group_ratio": top1_ratio,
        "top2_group_ratio": top2_ratio,
        "top3_groups": top3_text,
    }


def save_fold_csvs(df: pd.DataFrame, fold_to_groups: Dict[int, List[str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_groups = set(df[GROUP_COL].astype(str).tolist())

    for fold_idx, valid_groups in fold_to_groups.items():
        valid_groups = list(valid_groups)
        train_groups = sorted(list(all_groups - set(valid_groups)))
        train_df = df[df[GROUP_COL].astype(str).isin(train_groups)].copy()
        valid_df = df[df[GROUP_COL].astype(str).isin(valid_groups)].copy()

        if "sample_id" not in train_df.columns:
            train_df["sample_id"] = make_sample_id(train_df)
        if "sample_id" not in valid_df.columns:
            valid_df["sample_id"] = make_sample_id(valid_df)

        fold_dir = output_dir / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(fold_dir / "train.csv", index=False, encoding="utf-8-sig")
        valid_df.to_csv(fold_dir / "valid.csv", index=False, encoding="utf-8-sig")
        train_df.to_csv(fold_dir / "train_split.csv", index=False, encoding="utf-8-sig")
        valid_df.to_csv(fold_dir / "valid_split.csv", index=False, encoding="utf-8-sig")

        summary = build_valid_fold_summary(valid_df)
        print("=" * 80)
        print(f"Fold {fold_idx + 1}")
        print(f"  Train samples: {len(train_df)}")
        print(f"  Valid samples: {summary['valid_samples']}")
        print(f"  Train complex groups: {train_df[GROUP_COL].nunique()}")
        print(f"  Valid complex groups: {summary['valid_complex_groups']}")
        print(f"  Valid PDB groups: {summary['valid_pdb_groups']}")
        print(f"  Valid AB clusters: {summary['valid_ab_clusters']}")
        print(f"  Valid AG clusters: {summary['valid_ag_clusters']}")
        print(f"  Valid complex groups: {sorted(valid_groups)}")
        print(f"  Valid ddG mean: {summary['valid_ddg_mean']:.4f}")
        print(f"  Valid ddG std:  {summary['valid_ddg_std']:.4f}")
        print(f"  Valid single-mutation ratio: {summary['single_ratio']:.4f}")
        print(f"  Valid multi-mutation ratio:  {summary['multi_ratio']:.4f}")
        print(f"  Top1 dominant complex group: {summary['top1_group']} | ratio={summary['top1_group_ratio']:.4f}")
        print(f"  Top2 complex group ratio: {summary['top2_group_ratio']:.4f}")
        print(f"  Top3 complex groups: {summary['top3_groups']}")
        print("=" * 80)


def save_fold_summary(df: pd.DataFrame, fold_to_groups: Dict[int, List[str]], output_dir: Path) -> None:
    rows = []
    for fold_idx, valid_groups in fold_to_groups.items():
        valid_df = df[df[GROUP_COL].astype(str).isin(valid_groups)].copy()
        summary = build_valid_fold_summary(valid_df)
        rows.append({
            "fold": fold_idx + 1,
            **summary,
            "valid_complex_group_list": ";".join(sorted(valid_groups)),
            "valid_pdb_list": ";".join(sorted(valid_df[PDB_COL].astype(str).unique())),
        })
    pd.DataFrame(rows).sort_values("fold").reset_index(drop=True).to_csv(
        output_dir / "fold_summary.csv", index=False, encoding="utf-8-sig"
    )


def save_group_assignment_csv(stats_df: pd.DataFrame, fold_to_groups: Dict[int, List[str]], output_dir: Path) -> None:
    group_to_fold = {}
    for fold_idx, groups in fold_to_groups.items():
        for group_id in groups:
            group_to_fold[group_id] = fold_idx + 1

    rows = []
    for _, row in stats_df.iterrows():
        group_id = str(row[GROUP_COL])
        rows.append({
            GROUP_COL: group_id,
            "assigned_fold": group_to_fold.get(group_id, -1),
            "n_samples": int(row["n_samples"]),
            "n_pdbs": int(row["n_pdbs"]),
            "n_ab_clusters": int(row["n_ab_clusters"]),
            "n_ag_clusters": int(row["n_ag_clusters"]),
            "ddg_mean": float(row["ddg_mean"]),
            "ddg_std": float(row["ddg_std"]),
            "single_ratio": float(row["single_ratio"]),
            "multi_ratio": float(row["multi_ratio"]),
            "pdb_list": row["pdb_list"],
        })
    pd.DataFrame(rows).sort_values(["assigned_fold", "n_samples"], ascending=[True, False]).reset_index(drop=True).to_csv(
        output_dir / "group_assignment.csv", index=False, encoding="utf-8-sig"
    )


def print_feasibility_hint(stats_df: pd.DataFrame) -> None:
    total_samples = int(stats_df["n_samples"].sum())
    target_fold_samples = float(total_samples) / float(N_FOLDS)
    largest_group = int(stats_df["n_samples"].max())
    theoretical_min_top1 = float(largest_group) / max(target_fold_samples, 1e-8)
    print("\n" + "-" * 80)
    print("Complex-cluster split settings")
    print(f"  AB identity threshold: {AB_IDENTITY_THRESHOLD:.2f}")
    print(f"  AG identity threshold: {AG_IDENTITY_THRESHOLD:.2f}")
    print(f"  Unique complex groups: {len(stats_df)}")
    print(f"  Requested max top1 ratio: {MAX_TOP1_RATIO:.4f}")
    print(f"  Requested max top2 ratio: {MAX_TOP2_RATIO:.4f}")
    print(f"  Largest complex group size: {largest_group}")
    print(f"  Ideal fold sample size: {target_fold_samples:.2f}")
    print(f"  Theoretical min top1 ratio from largest group alone: {theoretical_min_top1:.4f}")
    if theoretical_min_top1 > MAX_TOP1_RATIO:
        print(
            "  [WARN] 最大 complex group 已经超过目标上限；脚本会尽量压低大家族占比，"
            "但不保证能低于设定阈值。"
        )
    print("-" * 80)


def print_global_summary(df: pd.DataFrame, fold_to_groups: Dict[int, List[str]]) -> None:
    rows = []
    for fold_idx, valid_groups in fold_to_groups.items():
        valid_df = df[df[GROUP_COL].astype(str).isin(valid_groups)].copy()
        summary = build_valid_fold_summary(valid_df)
        rows.append({
            "fold": fold_idx + 1,
            "samples": summary["valid_samples"],
            "complex_groups": summary["valid_complex_groups"],
            "pdb_groups": summary["valid_pdb_groups"],
            "ab_clusters": summary["valid_ab_clusters"],
            "ag_clusters": summary["valid_ag_clusters"],
            "ddg_mean": summary["valid_ddg_mean"],
            "ddg_std": summary["valid_ddg_std"],
            "single_ratio": summary["single_ratio"],
            "multi_ratio": summary["multi_ratio"],
            "top1_group": summary["top1_group"],
            "top1_ratio": summary["top1_group_ratio"],
            "top2_ratio": summary["top2_group_ratio"],
        })
    summary_df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    print("\n" + "#" * 80)
    print("Global fold summary")
    print(summary_df.to_string(index=False))
    print("#" * 80)


# =====================================
# Main
# =====================================
def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, sep=DATA_CONFIG.get("csv_sep", ","))

    required_cols = {PDB_COL, ID_COL, LABEL_COL, MUT_COL, PARTNERS_COL}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {sorted(missing_cols)}")

    df = df.copy()
    df[PDB_COL] = df[PDB_COL].astype(str).str.strip()
    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    if df[PDB_COL].eq("").any():
        raise ValueError(f"Column '{PDB_COL}' contains empty values")
    if df[ID_COL].eq("").any():
        raise ValueError(f"Column '{ID_COL}' contains empty values")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("Start complex-cluster split")
    print(f"Input CSV: {CSV_PATH}")
    print(f"WT PDB dir: {WT_PDB_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"N_FOLDS: {N_FOLDS}")
    print(f"Seed: {SEED}")
    print("=" * 80)

    df = add_complex_clusters(df)
    n_unique_groups = df[GROUP_COL].nunique()
    if n_unique_groups < N_FOLDS:
        raise ValueError(f"Unique complex clusters ({n_unique_groups}) is smaller than n_folds ({N_FOLDS})")

    stats_df = build_group_stats(df)
    print_feasibility_hint(stats_df)

    fold_to_groups = optimize_assignment(
        stats_df=stats_df,
        n_folds=N_FOLDS,
        seed=SEED,
        num_restarts=NUM_RESTARTS,
        max_move_passes=MAX_MOVE_PASSES,
        max_swap_passes=MAX_SWAP_PASSES,
    )

    final_loss = compute_assignment_loss(stats_df, fold_to_groups, N_FOLDS)
    print(f"\n[Final] best assignment loss = {final_loss:.6f}")

    save_fold_csvs(df, fold_to_groups, OUTPUT_DIR)
    save_fold_summary(df, fold_to_groups, OUTPUT_DIR)
    save_group_assignment_csv(stats_df, fold_to_groups, OUTPUT_DIR)
    print_global_summary(df, fold_to_groups)

    print("\nDone.")
    print(f"Input CSV: {CSV_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Total samples: {len(df)}")
    print(f"Total unique PDB groups: {df[PDB_COL].nunique()}")
    print(f"Total unique AB clusters: {df[AB_CLUSTER_COL].nunique()}")
    print(f"Total unique AG clusters: {df[AG_CLUSTER_COL].nunique()}")
    print(f"Total unique complex clusters: {df[GROUP_COL].nunique()}")
    print(f"Search restarts: {NUM_RESTARTS}")


if __name__ == "__main__":
    main()
