from pathlib import Path

# =========================================================
# 项目根目录
# =========================================================
PROJECT_ROOT = Path("/home/zhao/gwc/NEW3")

# =========================================================
# 当前使用的数据集名称
# 可选示例：
#   "SKEMPI"
#   "ABBIND"
# =========================================================
CURRENT_DATASET = "SKEMPI"

# =========================================================
# 划分方式配置
# 只需要修改这一行即可切换训练/验证使用哪一套 split。
# 可选：
#   "pdb"      -> data/{CURRENT_DATASET}/pdbsplits
#   "complex"  -> data/{CURRENT_DATASET}/complexsplits
# =========================================================
SPLITMODEL = "complex"

_ALLOWED_SPLITMODELS = {"pdb", "complex"}
_SPLITMODEL = str(SPLITMODEL).strip().lower()

if _SPLITMODEL not in _ALLOWED_SPLITMODELS:
    raise ValueError(
        f"Unsupported SPLITMODEL={SPLITMODEL!r}. "
        f"Allowed values: {sorted(_ALLOWED_SPLITMODELS)}"
    )

# =========================================================
# 消融实验配置
# 只需要修改这一行即可切换实验。
# 可选：
#   "none"
#   "w/o PLM"
#   "w/o DSSP"
#   "w/o mutation readout"
#   "w/o interface modeling"
#   "w/o pair encoder"
#   "full graph instead of local graph"
#   "PLM-only"                  -> 只使用 AntiBERTy/ESM2 节点嵌入做回归，不使用图结构/DSSP/界面/contact-delta
# =========================================================
ABLATION = "none"

_ALLOWED_ABLATIONS = {
    "none",
    "w/o plm",
    "w/o dssp",
    "w/o mutation readout",
    "w/o interface modeling",
    "w/o pair encoder",
    "full graph instead of local graph",
    "plm-only",
}

def _normalize_ablation_name(name: str) -> str:
    return str(name).strip().lower()


def _safe_ablation_tag(name: str) -> str:
    s = _normalize_ablation_name(name)
    if s in {"", "none", "full", "full model", "full_model"}:
        return "full_model"
    for old, new in [
        ("w/o", "wo"),
        ("/", "_"),
        (" ", "_"),
        ("-", "_"),
        ("__", "_"),
    ]:
        s = s.replace(old, new)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_") or "full_model"


_ABL = _normalize_ablation_name(ABLATION)
ABLATION_TAG = _safe_ablation_tag(ABLATION)

if _ABL not in _ALLOWED_ABLATIONS:
    raise ValueError(
        f"Unsupported ABLATION={ABLATION!r}. Allowed values: {sorted(_ALLOWED_ABLATIONS)}"
    )


ABLATION_CONFIG = {
    "name": ABLATION,
    "tag": ABLATION_TAG,

    # A0: PLM-only baseline。
    # 只使用节点特征中的 side-specific PLM 部分（AntiBERTy/ESM2，默认第 31:159 维），
    # 不经过 GNN、不使用 DSSP、边、界面建模、pair encoder 或 contact-delta。
    "use_plm_only": _ABL == "plm-only",

    # A1: 去掉 AntiBERTy / ESM2 PLM 特征；节点维度保持不变，PLM 位置填 0。
    # 注意：PLM-only 实验需要保留 PLM。
    "use_plm": _ABL != "w/o plm",

    # A2: 去掉 DSSP 二级结构 / 暴露度特征；节点维度保持不变，DSSP 位置填 0。
    # PLM-only 中也不使用 DSSP。
    "use_dssp": _ABL not in {"w/o dssp", "plm-only"},

    # A3: 去掉突变中心 readout。PLM-only 不使用 GNN readout。
    "use_mutation_readout": _ABL not in {"w/o mutation readout", "plm-only"},

    # A4: 去掉 interface modeling，包括 interface flag、跨 partner 边、interface readout。
    # PLM-only 不使用 interface flag/edge/readout，只用 PLM embedding。
    "use_interface_modeling": _ABL not in {"w/o interface modeling", "plm-only"},

    # A5: 去掉 WT-MUT pair interaction encoder。PLM-only 不使用 pair encoder。
    "use_pair_encoder": _ABL not in {"w/o pair encoder", "plm-only"},

    # A6: full graph instead of local graph。PLM-only 不依赖图结构，这个开关对它无实际影响。
    "use_local_subgraph": _ABL != "full graph instead of local graph",
}

# =========================================================
# Top-K mutation prioritization 配置
# =========================================================
TOPK_CONFIG = {
    # internal_cv: 使用当前数据集五折验证 out-of-fold prediction
    # external: 使用外部验证 ensemble prediction
    "mode": "internal_cv",

    # checkpoint 类型：只使用 best_composite
    "ckpt_kind": "best_composite",

    # Top-K 比例
    "top_fracs": [0.05, 0.10, 0.20],

    # 正 ddG：削弱结合
    "destabilizing_threshold": 1.0,

    # 负 ddG：增强结合
    # 如果增强样本太少，可先用 -0.5；补充材料可以再给 -1.0
    "stabilizing_threshold": -0.5,

    # 内部验证预测列
    "true_col": "y_true",
    "pred_col": "y_pred",

    # 外部验证时才用
    "external_train_dataset": "SKEMPI",
    "external_val_dataset": "SARS",
    "external_pred_col": "y_pred_ensemble_mean",
}

# =========================================================
# 基础目录
# =========================================================
DATA_ROOT = PROJECT_ROOT / "data"
FEATURE_ROOT = PROJECT_ROOT / "feature"
OUTPUT_ROOT = PROJECT_ROOT / "output"

DATASET_DIR = DATA_ROOT / CURRENT_DATASET
DATASET_FEATURE_DIR = FEATURE_ROOT / CURRENT_DATASET
DATASET_OUTPUT_DIR = OUTPUT_ROOT / CURRENT_DATASET
ABLATION_OUTPUT_DIR = DATASET_OUTPUT_DIR / ABLATION_TAG

# 原始数据目录
WT_PDB_DIR = DATASET_DIR / "wt"
MUT_PDB_DIR = DATASET_DIR / "mut"

# 五折划分目录
# 旧的 PDB-name split 根目录已经更名为 pdbsplits。
# 新的 complex-cluster split 根目录为 complexsplits。
PDB_SPLIT_DIR = DATASET_DIR / "pdbsplits"
COMPLEX_SPLIT_DIR = DATASET_DIR / "complexsplits"

if _SPLITMODEL == "pdb":
    SPLIT_DIR = PDB_SPLIT_DIR
elif _SPLITMODEL == "complex":
    SPLIT_DIR = COMPLEX_SPLIT_DIR
else:
    # 理论上不会走到这里，前面已经校验。
    raise ValueError(f"Unsupported SPLITMODEL={SPLITMODEL!r}")

# 特征目录
ESM_DIR = DATASET_FEATURE_DIR / "esm"
ANTIBERTY_DIR = DATASET_FEATURE_DIR / "antiberty"
DSSP_DIR = DATASET_FEATURE_DIR / "dssp"
GRAPH_DIR = DATASET_FEATURE_DIR / "graph"

# 图缓存只区分 local/full graph，不按 PLM/DSSP/interface 等消融区分。
# 这样 w/o PLM / w/o DSSP / w/o mutation readout / w/o interface modeling / w/o pair encoder
# 都可以复用同一套 full-feature local 图缓存。


# 训练输出目录（按数据集和消融实验隔离，避免互相覆盖）
CKPT_DIR = ABLATION_OUTPUT_DIR / "checkpoints"
LOG_DIR = ABLATION_OUTPUT_DIR / "logs"
PRED_DIR = ABLATION_OUTPUT_DIR / "predict"

# =========================================================
# 自动建目录
# =========================================================
for _dir in [
    DATA_ROOT,
    FEATURE_ROOT,
    OUTPUT_ROOT,
    DATASET_DIR,
    DATASET_FEATURE_DIR,
    DATASET_OUTPUT_DIR,
    ABLATION_OUTPUT_DIR,
    WT_PDB_DIR,
    MUT_PDB_DIR,
    PDB_SPLIT_DIR,
    COMPLEX_SPLIT_DIR,
    SPLIT_DIR,
    GRAPH_DIR,
    ESM_DIR / "wt",
    ESM_DIR / "mut",
    ANTIBERTY_DIR / "wt",
    ANTIBERTY_DIR / "mut",
    DSSP_DIR / "wt",
    DSSP_DIR / "mut",
    CKPT_DIR,
    LOG_DIR,
    PRED_DIR,
]:
    _dir.mkdir(parents=True, exist_ok=True)

# =========================================================
# 自动推断 csv 文件路径
# 优先使用 data/{数据集名}/{数据集名}.csv
# 若不存在，则退回 data/{数据集名}/dataset.csv
# =========================================================
DEFAULT_CSV_PATH = DATASET_DIR / f"{CURRENT_DATASET}.csv"
FALLBACK_CSV_PATH = DATASET_DIR / "dataset.csv"

if DEFAULT_CSV_PATH.exists():
    CSV_PATH = DEFAULT_CSV_PATH
else:
    CSV_PATH = FALLBACK_CSV_PATH

# =========================================================
# 当前项目数据配置
# =========================================================
DATA_CONFIG = {
    "dataset_name": CURRENT_DATASET,
    "dataset_dir": DATASET_DIR,
    "csv_sep": ",",
    "csv_path": CSV_PATH,

    # 原始 PDB
    "wt_pdb_dir": WT_PDB_DIR,
    "mut_pdb_dir": MUT_PDB_DIR,

    # 图缓存（预处理后 .pt）
    "sample_cache_dir": GRAPH_DIR,

    # 五折交叉验证切分目录
    # 训练代码只读取 split_dir；通过 SPLITMODEL 在 pdbsplits / complexsplits 间切换。
    "split_model": _SPLITMODEL,
    "split_dir": SPLIT_DIR,
    "pdb_split_dir": PDB_SPLIT_DIR,
    "complex_split_dir": COMPLEX_SPLIT_DIR,

    # 缓存控制
    "use_cache": True,
    "rebuild_cache": False,
}

# =========================================================
# 列名配置
# =========================================================
COLUMN_CONFIG = {
    "mut_pdb_id": "ID",         # 突变体 PDB 名 / 唯一 sample id
    "wt_pdb_id": "PDB",         # 野生型 PDB 名
    "mutation": "Mutation",     # 突变字符串
    "partners": "Partners",     # 链分组列
    "label": "ddG",             # 监督标签
}

# =========================================================
# 文件配置
# =========================================================
FILE_CONFIG = {
    "pdb_suffix": ".pdb",
    "case_sensitive": False,
}

# =========================================================
# 突变配置
# =========================================================
MUTATION_CONFIG = {
    "multi_mut_sep": ",",
    "allowed_amino_acids": set("ACDEFGHIKLMNPQRSTVWY"),
}

# =========================================================
# Partners 配置
# 下划线左边 = 抗体链
# 下划线右边 = 抗原链
# =========================================================
PARTNER_CONFIG = {
    "group_sep": "_",
}

# =========================================================
# 特征配置
# =========================================================
FEATURE_CONFIG = {

    "use_antiberty": True,
    "use_esm2": True,

    # AntiBERTy
    "antiberty_wt_dir": ANTIBERTY_DIR / "wt",
    "antiberty_mut_dir": ANTIBERTY_DIR / "mut",
    "antiberty_dim": 128,

    # ESM2
    "esm2_wt_dir": ESM_DIR / "wt",
    "esm2_mut_dir": ESM_DIR / "mut",
    "esm2_dim": 128,

    # 是否使用 side-specific PLM；预处理阶段始终保留。
    "use_side_specific_plm": True,
}

# =========================================================
# DSSP 配置
# =========================================================
DSSP_CONFIG = {
    # 预处理阶段始终读取 DSSP；消融在训练时动态置零。
    "use_dssp": True,
    "wt_dir": DSSP_DIR / "wt",
    "mut_dir": DSSP_DIR / "mut",
    "dim": 4,
}

# =========================================================
# 节点特征维度
# 20 one-hot
# + 3 理化
# + 1 mutation flag
# + 1 is_antibody
# + 1 is_antigen
# + 1 is_interface
# + 4 DSSP
# + 128 PLM
# = 159
# 注意：消融 PLM/DSSP 时不改变维度，只把对应位置填 0。
# =========================================================
NODE_FEAT_DIM = 159

# =========================================================
# 边特征维度
# 1) seq_dist_norm
# 2) ca_dist_norm
# 3) is_sequential_edge
# 4) is_same_chain
# 5) is_cross_partner
# 6) is_interface_edge
# = 6
# 注意：消融 interface modeling 时不改变维度，只关闭跨 partner 边并将相关边特征置 0。
# =========================================================
EDGE_FEAT_DIM = 6

# =========================================================
# 图构建参数
# =========================================================
GRAPH_CONFIG = {
    # 维度
    "node_feat_dim": NODE_FEAT_DIM,
    "edge_feat_dim": EDGE_FEAT_DIM,

    # 建边
    "intra_edge_threshold": 8.0,
    "interface_edge_threshold": 12.0,
    # v1 contact-delta 显式接触对阈值。
    "contact_threshold": 8.0,
    "interface_contact_threshold": 12.0,
    "add_sequential_edges": True,
    "add_intra_spatial_edges": True,
    # 预处理阶段始终保留 interface/cross-partner 信息；消融在训练时动态置零。
    "add_inter_partner_edges": True,

    # 联合局部图
    "graph_mode": "joint_wt_mut",
    "use_joint_graph": True,
    "use_local_subgraph": ABLATION_CONFIG["use_local_subgraph"],
    "local_radius": 12.0,
    "local_topk_fallback": 20,
    "include_cross_partner_context": True,
    "merge_multi_mut_centers": True,

    # 节点标签
    "add_mutation_flag": True,
    "add_side_flags": True,
    "add_interface_flag": True,

    # 几何信息
    "use_pos": True,
    "use_edge_vector": True,
    "use_ca_coord": True,
}
# =========================================================
# 图缓存版本
# 只区分 local/full graph。
# PLM/DSSP/interface/pair/mutation-readout 等消融在训练阶段动态处理，
# 不应该单独生成不同图缓存。
# =========================================================
GRAPH_CACHE_VERSION = (
    "graph_v1_full"
    if not GRAPH_CONFIG.get("use_local_subgraph", True)
    else "graph_v1_local"
)
# =========================================================
# 模型参数
# =========================================================
MODEL_CONFIG = {
    "gnn_type": "gvp",

    "hidden_dim": 128,
    "dropout": 0.1,
    "num_gnn_layers": 3,

    "node_scalar_dim": NODE_FEAT_DIM,
    "node_vector_dim": 0,
    "edge_scalar_dim": EDGE_FEAT_DIM,
    "edge_vector_dim": 1,

    "pooling": "attention",
    "out_dim": 1,

    "use_delta_head": True,
    "delta_head_hidden_dim": 256,
    "delta_head_num_layers": 2,

    "use_mutation_readout": ABLATION_CONFIG["use_mutation_readout"],
    "use_global_readout": True,
    "use_interface_readout": ABLATION_CONFIG["use_interface_modeling"],
    "use_pair_encoder": ABLATION_CONFIG["use_pair_encoder"],

    # PLM-only baseline 开关：只使用节点 PLM embedding 做 WT/MUT pooling-difference 回归。
    "use_plm_only": ABLATION_CONFIG["use_plm_only"],
    "plm_start_idx": 31,
    "plm_end_idx": 159,

    # =====================================================
    # Prediction mode
    #   "concat_head"：旧版，所有表示拼接后单 head 回归。
    #   "interaction_delta_v0"：第一版 interaction-delta，不重建图缓存。
    #   "interaction_delta_v1_contact"：显式 contact-delta 分支，需要重建图缓存。
    #       pred = local_mutation_effect
    #            + changed_interface_contact_effect
    #            + global_calibration_bias
    # =====================================================
    "prediction_mode": "interaction_delta_v1_contact",
    "return_components": True,
    "use_local_mutation_effect": True,
    "use_changed_interface_contact_effect": True,
    "use_global_calibration_bias": True,

    "require_cached_alignment": True,
    "pair_encoder_type": "aligned_pair_mlp",
    # 当前 _build_pair_feat 实际生成 5 维：d, inv_d, is_mut_pair, is_interface_pair, cross_side。
    # 第一版不改缓存，因此这里必须和现有 pair_feat 对齐。
    "pair_feat_dim": 5,
    "pair_num_heads": 4,
    "pair_num_layers": 2,
    "pair_dropout": 0.1,

    # v1 contact-delta branch：显式抗体-抗原接触变化特征。
    "use_contact_delta_encoder": True,
    "contact_pair_feat_dim": 16,
    "contact_pair_hidden_dim": 128,
    "contact_delta_dropout": 0.1,
    "contact_threshold": 8.0,
    "interface_contact_threshold": 12.0,

    # V1.3: old_contact_effect + 0.15 * contact_delta_effect
    "use_contact_delta_residual": True,
    "contact_delta_residual_scale": 0.15,
}

# =========================================================
# 训练参数
# =========================================================
TRAIN_CONFIG = {
    "seed": 42,

    # 多 GPU 并行折训练
    "parallel_folds": True,
    "gpu_list": [5, 4, 6, 7, 3],

    # CV
    "num_folds": 5,

    # complex-cluster split 参数。
    # complex-clustersplit_cv5.py 会优先读取这里的配置；没有配置时使用脚本默认值。
    "complexsplit_ab_identity": 0.50,
    "complexsplit_ag_identity": 0.50,
    "complexsplit_num_restarts": 300,
    "complexsplit_max_move_passes": 40,
    "complexsplit_max_swap_passes": 25,

    # DataLoader
    "batch_size": 16,
    "num_workers": 0,
    "shuffle": True,

    # 优化
    "epochs": 100,
    "lr": 3e-4,
    "weight_decay": 1e-4,

    # loss
    "loss_type": "mse",
    "huber_delta": 1.0,
    "grad_clip_norm": 1.0,

    # Exp4 baseline: 不默认启用 calibration L2。
    "use_calibration_l2": False,
    "calibration_l2_lambda": 0.0,
    "calibration_l2_component_name": "global_calibration_bias",

    # V1.3: 对 contact_delta_effect 分支输出加 L2，防止显式 contact 分支过拟合。
    # 注意：model.py 返回的 raw contact-delta 分量名是 "contact_delta_effect"。
    "use_contact_delta_l2": True,
    "contact_delta_l2_lambda": 1e-3,
    "contact_delta_l2_component_name": "contact_delta_effect",

    # early stopping
    "use_early_stopping": True,
    "early_stopping_patience": 30,
    "save_checkpoints": False,

    # 监控指标
    "monitor_metric": "composite",
    "monitor_mode": "min",
    "composite_pearson_weight": 0.5,

    # 设备
    "use_cuda": True,

    # 可复现性
    "deterministic": False,
    "seed_per_worker": True,
}

# =========================================================
# 运行配置
# =========================================================
RUN_CONFIG = {
    "cv_split_naming": "fold",      # fold_1, fold_2, ...
    "train_csv_name": "train.csv",
    "valid_csv_name": "valid.csv",
}

# =========================================================
# 日志配置
# =========================================================
LOG_CONFIG = {
    "log_level": "INFO",
    "log_file": LOG_DIR / f"{CURRENT_DATASET.lower()}_{ABLATION_TAG}_train.log",
}

# =========================================================
# 指标配置
# =========================================================
METRIC_CONFIG = {
    "primary_metric": "rmse",
    "metrics": ["rmse", "mae", "pearson"],
}

# =========================================================
# 调试配置
# =========================================================
DEBUG_CONFIG = {
    "debug": False,
    "max_samples": None,
    "verbose": True,
}

# =========================================================
# 汇总配置
# =========================================================
CONFIG = {
    "project_root": PROJECT_ROOT,
    "current_dataset": CURRENT_DATASET,
    "split_model": _SPLITMODEL,
    "ablation": ABLATION_CONFIG,
    "data": DATA_CONFIG,
    "columns": COLUMN_CONFIG,
    "files": FILE_CONFIG,
    "mutation": MUTATION_CONFIG,
    "partners": PARTNER_CONFIG,
    "feature": FEATURE_CONFIG,
    "dssp": DSSP_CONFIG,
    "graph": GRAPH_CONFIG,
    "model": MODEL_CONFIG,
    "train": TRAIN_CONFIG,
    "run": RUN_CONFIG,
    "log": LOG_CONFIG,
    "metric": METRIC_CONFIG,
    "debug": DEBUG_CONFIG,
}

