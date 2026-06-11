import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch

from config import (
    DATA_CONFIG,
    GRAPH_CONFIG,
    LOG_CONFIG,
    RUN_CONFIG,
    COLUMN_CONFIG,
    ABLATION_CONFIG,
    GRAPH_CACHE_VERSION,
)
from dataset import (
    load_dataframe,
    validate_row,
    build_sample_from_row,
)
from pdb_graph import ComplexGraphBuilder


# =========================================================
# 当前图缓存版本
# 改了 sample 结构 / 特征体系 / 图结构后，务必升级版本号
# =========================================================



# =========================================================
# 日志
# =========================================================
def setup_logger() -> None:
    log_level = getattr(logging, str(LOG_CONFIG["log_level"]).upper(), logging.INFO)

    log_path = Path(LOG_CONFIG["log_file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(log_level)

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# =========================================================
# 保存失败日志
# =========================================================
def save_failed_rows(failed_rows: List[Tuple[int, str]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["row_idx", "error"])
        for row_idx, err in failed_rows:
            writer.writerow([row_idx, err])


# =========================================================
# 安全读取缓存样本
# =========================================================
def safe_load_cached_sample(sample_cache_path: Path) -> Optional[Dict[str, Any]]:
    try:
        sample = torch.load(sample_cache_path, map_location="cpu")
        if not isinstance(sample, dict):
            return None
        return sample
    except Exception:
        return None


# =========================================================
# 检查图对象是否合法
# =========================================================
def _is_valid_graph_object(graph: Any) -> bool:
    if graph is None:
        return False
    if not hasattr(graph, "x"):
        return False
    if graph.x is None:
        return False
    if not hasattr(graph, "edge_index"):
        return False
    if not hasattr(graph, "edge_attr"):
        return False
    if not hasattr(graph, "pos"):
        return False
    if not hasattr(graph, "mut_idx"):
        return False
    return True


# =========================================================
# 检查缓存样本是否可直接复用
# 当前要求：
# 1. 基础字段存在
# 2. wt_joint_graph / mut_joint_graph 存在
# 3. graph_version 一致
# 4. 节点 / 边特征维度一致
# =========================================================
def is_cache_sample_compatible(sample: Dict[str, Any]) -> bool:
    required_keys = [
        "sample_id",
        "wt_pdb_id",
        "mut_pdb_id",
        "mutation_str",
        "partners_str",
        "wt_joint_graph",
        "mut_joint_graph",
        "ddg",
        "graph_version",
        # v1 local/pair-cache fields. 这些字段缺失时说明是旧缓存，
        # 即使基础图能读，也不能直接复用，否则 local/pair/contact 分支会退化。
        "aligned_wt_idx",
        "aligned_mut_idx",
        "mutation_aligned_wt_idx",
        "mutation_aligned_mut_idx",
        "mutation_pair_index",
        "mutation_pair_feat",
        "interface_contact_pair_index",
        "interface_contact_pair_feat",
    ]
    for key in required_keys:
        if key not in sample:
            return False

    sample_graph_version = str(sample.get("graph_version", "")).strip()
    if sample_graph_version != GRAPH_CACHE_VERSION:
        return False

    graph_keys = ["wt_joint_graph", "mut_joint_graph"]
    detected_node_feat_dim = None
    detected_edge_feat_dim = None

    for key in graph_keys:
        graph = sample.get(key, None)
        if not _is_valid_graph_object(graph):
            return False

        if graph.x.dim() != 2:
            return False

        if graph.edge_index.dim() != 2 or graph.edge_index.size(0) != 2:
            return False

        if graph.edge_attr.dim() != 2:
            return False

        if graph.x.numel() > 0:
            detected_node_feat_dim = int(graph.x.size(-1))

        if graph.edge_attr.numel() > 0:
            detected_edge_feat_dim = int(graph.edge_attr.size(-1))

    config_node_feat_dim = int(GRAPH_CONFIG["node_feat_dim"])
    config_edge_feat_dim = int(GRAPH_CONFIG["edge_feat_dim"])

    if detected_node_feat_dim is not None and detected_node_feat_dim != config_node_feat_dim:
        return False

    if detected_edge_feat_dim is not None and detected_edge_feat_dim != config_edge_feat_dim:
        return False

    # 检查新增 pair/alignment 字段的基本形状，防止复用旧版坏缓存。
    for key in [
        "aligned_wt_idx",
        "aligned_mut_idx",
        "mutation_aligned_wt_idx",
        "mutation_aligned_mut_idx",
    ]:
        val = sample.get(key, None)
        if not isinstance(val, torch.Tensor) or val.dim() != 1:
            return False

    pair_shape_checks = {
        "mutation_pair_index": 2,
        "interface_contact_pair_index": 4,
    }
    for key, rows in pair_shape_checks.items():
        val = sample.get(key, None)
        if not isinstance(val, torch.Tensor) or val.dim() != 2 or val.size(0) != rows:
            return False

    feat_shape_checks = {
        "mutation_pair_feat": 5,
        "interface_contact_pair_feat": 16,
    }
    for key, cols in feat_shape_checks.items():
        val = sample.get(key, None)
        if not isinstance(val, torch.Tensor) or val.dim() != 2 or val.size(1) != cols:
            return False

    return True


# =========================================================
# 文件命名：直接使用原始 CSV 的 ID 列
# 生成: {ID}.pt
# =========================================================
def make_graph_cache_name_from_row(row) -> str:
    id_col = COLUMN_CONFIG["mut_pdb_id"]
    mut_id = str(row.get(id_col, "")).strip()
    if not mut_id:
        raise ValueError(f"Missing '{id_col}' when making graph cache file name")

    mut_id = (
        mut_id
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace('"', "_")
        .replace("<", "_")
        .replace(">", "_")
        .replace("|", "_")
        .strip()
    )
    return f"{mut_id}.pt"


# =========================================================
# 构建图构建器
# =========================================================
def build_graph_builder() -> ComplexGraphBuilder:
    return ComplexGraphBuilder(
        contact_threshold=float(GRAPH_CONFIG.get("intra_edge_threshold", 8.0)),
        interface_threshold=float(GRAPH_CONFIG.get("interface_edge_threshold", 12.0)),
        add_sequential_edges=bool(GRAPH_CONFIG.get("add_sequential_edges", True)),
        add_intra_spatial_edges=bool(GRAPH_CONFIG.get("add_intra_spatial_edges", True)),
        add_inter_partner_edges=bool(GRAPH_CONFIG.get("add_inter_partner_edges", True)),
        add_mutation_flag=bool(GRAPH_CONFIG.get("add_mutation_flag", True)),
        add_side_flags=bool(GRAPH_CONFIG.get("add_side_flags", True)),
        add_interface_flag=bool(GRAPH_CONFIG.get("add_interface_flag", True)),
    )


# =========================================================
# 主预处理流程
# 只生成图缓存 .pt，不再保存 cache_index.csv
# =========================================================
def preprocess_dataset() -> None:
    csv_path = Path(DATA_CONFIG["csv_path"])
    wt_pdb_dir = Path(DATA_CONFIG["wt_pdb_dir"])
    mut_pdb_dir = Path(DATA_CONFIG["mut_pdb_dir"])
    sample_cache_dir = Path(DATA_CONFIG["sample_cache_dir"])

    use_cache = bool(DATA_CONFIG.get("use_cache", True))
    rebuild_cache = bool(DATA_CONFIG.get("rebuild_cache", False))

    failed_rows_path = sample_cache_dir / "failed_rows.csv"

    sample_cache_dir.mkdir(parents=True, exist_ok=True)

    graph_builder = build_graph_builder()
    df = load_dataframe(csv_path)

    failed_rows: List[Tuple[int, str]] = []

    total_rows = len(df)
    built_count = 0
    reused_count = 0
    skipped_count = 0
    rebuilt_incompatible_count = 0

    logging.info("=" * 80)
    logging.info("Start preprocessing dataset")
    logging.info(f"Dataset: {DATA_CONFIG['dataset_name']}")
    logging.info(f"CSV path: {csv_path}")
    logging.info(f"WT pdb dir: {wt_pdb_dir}")
    logging.info(f"MUT pdb dir: {mut_pdb_dir}")
    logging.info(f"Sample cache dir: {sample_cache_dir}")
    logging.info(f"Graph naming rule: {{ID}}.pt")
    logging.info(f"use_cache={use_cache}, rebuild_cache={rebuild_cache}")
    logging.info(f"Ablation: {ABLATION_CONFIG.get('name', 'none')} | tag={ABLATION_CONFIG.get('tag', 'full_model')}")
    logging.info(
        "Graph cache policy: cache depends only on local/full graph; "
        "PLM/DSSP/interface/pair/mutation-readout ablations are applied dynamically during training."
    )
    logging.info(f"graph_cache_version={GRAPH_CACHE_VERSION}")
    logging.info(f"Total rows in CSV: {total_rows}")
    logging.info("=" * 80)

    seen_cache_names: Dict[str, int] = {}

    for idx, row in df.iterrows():
        ok, msg = validate_row(row)
        if not ok:
            failed_rows.append((idx, f"Validation failed: {msg}"))
            continue

        try:
            cache_name = make_graph_cache_name_from_row(row)
        except Exception as e:
            failed_rows.append((idx, str(e)))
            continue

        if cache_name in seen_cache_names:
            failed_rows.append(
                (
                    idx,
                    f"Duplicate cache name detected: {cache_name} | "
                    f"current_row={idx}, previous_row={seen_cache_names[cache_name]}"
                )
            )
            continue
        seen_cache_names[cache_name] = idx

        sample_cache_path = sample_cache_dir / cache_name
        source = "unknown"

        try:
            item = None
            cache_existed_before = sample_cache_path.exists()

            # -------------------------------------------------
            # 情况1：缓存存在，且不要求强制重建
            # -------------------------------------------------
            if use_cache and cache_existed_before and (not rebuild_cache):
                cached_item = safe_load_cached_sample(sample_cache_path)

                if cached_item is not None and is_cache_sample_compatible(cached_item):
                    item = cached_item
                    source = "cache_reuse"
                    reused_count += 1
                    skipped_count += 1
                else:
                    item = build_sample_from_row(
                        row=row,
                        graph_builder=graph_builder,
                        wt_pdb_dir=wt_pdb_dir,
                        mut_pdb_dir=mut_pdb_dir,
                        row_idx=idx,
                    )
                    item["graph_version"] = GRAPH_CACHE_VERSION
                    item["sample_id"] = str(row.get(COLUMN_CONFIG["mut_pdb_id"], "")).strip()

                    if use_cache:
                        torch.save(item, sample_cache_path)

                    source = "rebuild_incompatible_cache"
                    built_count += 1
                    rebuilt_incompatible_count += 1

            # -------------------------------------------------
            # 情况2：缓存不存在，或要求强制重建
            # -------------------------------------------------
            else:
                item = build_sample_from_row(
                    row=row,
                    graph_builder=graph_builder,
                    wt_pdb_dir=wt_pdb_dir,
                    mut_pdb_dir=mut_pdb_dir,
                    row_idx=idx,
                )
                item["graph_version"] = GRAPH_CACHE_VERSION
                item["sample_id"] = str(row.get(COLUMN_CONFIG["mut_pdb_id"], "")).strip()

                if use_cache:
                    torch.save(item, sample_cache_path)

                source = "rebuild_forced" if (rebuild_cache and cache_existed_before) else "built"
                built_count += 1

        except Exception as e:
            failed_rows.append((idx, str(e)))
            continue

        if (idx + 1) % 50 == 0 or (idx + 1) == total_rows:
            logging.info(
                f"Processed {idx + 1}/{total_rows} | "
                f"failed={len(failed_rows)} | "
                f"built={built_count} | reused={reused_count} | "
                f"rebuild_bad_cache={rebuilt_incompatible_count} | "
                f"last={source}"
            )

    save_failed_rows(failed_rows, failed_rows_path)

    logging.info(f"Saved failed rows to: {failed_rows_path}")
    logging.info(
        f"Preprocess done. total={total_rows}, failed={len(failed_rows)}, "
        f"built={built_count}, reused={reused_count}, rebuild_bad_cache={rebuilt_incompatible_count}"
    )


# =========================================================
# 检查当前 split 目录结构（可选提示）
# =========================================================
def check_cv_split_layout() -> None:
    split_dir = Path(DATA_CONFIG.get("split_dir", ""))
    if not split_dir.exists():
        logging.warning(f"Split dir not found: {split_dir}")
        return

    fold_prefix = str(RUN_CONFIG.get("cv_split_naming", "fold"))
    train_name = str(RUN_CONFIG.get("train_csv_name", "train.csv"))
    valid_name = str(RUN_CONFIG.get("valid_csv_name", "valid.csv"))

    fold_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith(fold_prefix)])
    if not fold_dirs:
        logging.warning(f"No CV fold directories found under: {split_dir}")
        return

    for fold_dir in fold_dirs:
        train_csv = fold_dir / train_name
        valid_csv = fold_dir / valid_name
        if not train_csv.exists() or not valid_csv.exists():
            logging.warning(
                f"Incomplete fold split: {fold_dir} | "
                f"train_exists={train_csv.exists()} valid_exists={valid_csv.exists()}"
            )


if __name__ == "__main__":
    setup_logger()
    check_cv_split_layout()
    preprocess_dataset()
