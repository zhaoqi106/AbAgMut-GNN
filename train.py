import copy
import logging
import math
import multiprocessing as mp
import os
import queue
import signal
import ctypes
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from config import (
    TRAIN_CONFIG,
    DATA_CONFIG,
    LOG_CONFIG,
    RUN_CONFIG,
    CKPT_DIR,
    PRED_DIR,
    COLUMN_CONFIG,
    ABLATION_CONFIG,
)
from model import ddGModel

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.load.*weights_only=False.*"
)


# =========================================================
# 进程清理 / 父进程死亡联动
# =========================================================
_CHILD_PROCESSES: List[mp.Process] = []
_MAIN_SHUTTING_DOWN = False


def set_parent_death_signal(sig: int = signal.SIGTERM) -> None:
    """
    Linux-only: 让当前进程在父进程死亡时自动收到 sig。

    这对 multiprocessing spawn 出来的 worker 很关键：
    即使主进程被 kill -9，worker 也会因为父进程死亡而收到 SIGTERM。
    worker 再退出后，它自己创建的子进程也会因为同样机制继续退出。
    """
    if os.name != "posix":
        return

    try:
        libc = ctypes.CDLL("libc.so.6")
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, int(sig), 0, 0, 0)

        # 防止极小概率竞态：设置 PDEATHSIG 前父进程已经死了。
        if os.getppid() == 1:
            os.kill(os.getpid(), int(sig))
    except Exception:
        # 不让清理机制影响正常训练启动。
        pass


def terminate_child_processes(processes: Optional[List[mp.Process]] = None, grace_seconds: float = 8.0) -> None:
    """先 TERM 再 KILL 清理当前 train.py 直接创建的 multiprocessing worker。"""
    if processes is None:
        processes = _CHILD_PROCESSES

    alive = [p for p in processes if p is not None and p.pid is not None and p.is_alive()]
    if not alive:
        return

    logging.warning("Terminating child worker processes: %s", [p.pid for p in alive])

    for p in alive:
        try:
            p.terminate()
        except Exception:
            pass

    deadline = time.time() + float(grace_seconds)
    for p in alive:
        remaining = max(0.0, deadline - time.time())
        try:
            p.join(timeout=remaining)
        except Exception:
            pass

    still_alive = [p for p in alive if p.is_alive()]
    if still_alive:
        logging.warning("Force killing child worker processes: %s", [p.pid for p in still_alive])
        for p in still_alive:
            try:
                p.kill()
            except Exception:
                pass
        for p in still_alive:
            try:
                p.join(timeout=2.0)
            except Exception:
                pass


def install_main_signal_handlers() -> None:
    """主进程收到 kill/ctrl-c 时，先清理所有 worker，再退出。"""
    def _handler(signum, frame):
        global _MAIN_SHUTTING_DOWN
        if _MAIN_SHUTTING_DOWN:
            raise SystemExit(128 + int(signum))
        _MAIN_SHUTTING_DOWN = True
        logging.warning("Main process received signal %s; cleaning child processes...", signum)
        terminate_child_processes()
        raise SystemExit(128 + int(signum))

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass


# =========================================================
# CPU 资源限制
# =========================================================
def configure_cpu_limit(cpu_fraction: Optional[float] = None) -> int:
    """
    将训练进程限制在最多约 cpu_fraction 比例的 CPU 核心上，并同步限制常见数值库线程数。

    默认读取 TRAIN_CONFIG["max_cpu_fraction"]，未配置时为 0.70。
    注意：这是按可用 CPU 核心数做限制；GPU 计算不受这里限制。
    """
    if cpu_fraction is None:
        cpu_fraction = float(TRAIN_CONFIG.get("max_cpu_fraction", 0.70))

    cpu_fraction = max(0.01, min(float(cpu_fraction), 1.0))
    total_cpus = os.cpu_count() or 1
    max_cpus = max(1, int(math.floor(total_cpus * cpu_fraction)))

    try:
        current_affinity = sorted(os.sched_getaffinity(0))
        max_cpus = min(max_cpus, len(current_affinity))
        allowed_cpus = set(current_affinity[:max_cpus])
        os.sched_setaffinity(0, allowed_cpus)
    except Exception as e:
        logging.warning(f"CPU affinity limit not applied: {e}")

    thread_env_keys = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ]
    for key in thread_env_keys:
        os.environ[key] = str(max_cpus)

    try:
        torch.set_num_threads(max_cpus)
        torch.set_num_interop_threads(max(1, min(max_cpus, 4)))
    except Exception as e:
        logging.warning(f"PyTorch CPU thread limit not fully applied: {e}")

    logging.info(
        f"CPU limit configured: max_cpu_fraction={cpu_fraction:.2f}, "
        f"allowed_cpu_threads={max_cpus}/{total_cpus}"
    )
    return max_cpus


# =========================================================
# 日志
# =========================================================
def setup_logger() -> None:
    log_level = getattr(logging, str(LOG_CONFIG.get("log_level", "INFO")).upper(), logging.INFO)
    log_file = Path(LOG_CONFIG["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(log_level)
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# =========================================================
# 随机种子
# =========================================================
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def seed_worker(worker_id: int) -> None:
    set_parent_death_signal(signal.SIGTERM)
    base_seed = int(TRAIN_CONFIG.get("seed", 42))
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# =========================================================
# 基础工具
# =========================================================
def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def get_id_col() -> str:
    return str(COLUMN_CONFIG.get("mut_pdb_id", "ID"))


def get_label_col() -> str:
    return str(COLUMN_CONFIG.get("label", "ddG"))


def filter_records_by_ddg_range(
        records: List[Dict[str, Any]],
        min_ddg_exclusive: float = -8.0,
        max_ddg_exclusive: float = 8.0,
) -> List[Dict[str, Any]]:
    """
    只保留 -8 < ddG < 8 的样本，即去掉 ddG <= -8 和 ddG >= 8。
    """
    label_col = get_label_col()
    kept = []

    for rec in records:
        try:
            ddg = float(rec[label_col])
        except Exception:
            continue

        if min_ddg_exclusive < ddg < max_ddg_exclusive:
            kept.append(rec)

    return kept


def get_GRAPH_DIR() -> Path:
    graph_dir = DATA_CONFIG.get("sample_cache_dir", None)
    if graph_dir:
        return Path(graph_dir)
    return Path("/home/zhao/gwc/NEW3/output/graph")


def normalize_split_key(x: Any) -> str:
    s = safe_str(x)
    if not s:
        return ""
    m = re.match(r"^\d{6}_(.+)$", s)
    if m:
        return m.group(1)
    return s


# =========================================================
# 指标
# =========================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if len(y_true) == 0:
        return {"rmse": 0.0, "mae": 0.0, "pearson": 0.0}

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    if len(y_true) > 1 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = 0.0

    return {"rmse": rmse, "mae": mae, "pearson": pearson}


# =========================================================
# ddG 标签标准化
# =========================================================
class LabelScaler:
    """
    Per-fold ddG 标准化器。

    训练 loss 在标准化后的 z-ddG 尺度上计算；
    日志、验证指标、预测 CSV 全部反标准化回原始 ddG 尺度。
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0, enabled: bool = True):
        self.mean = float(mean)
        self.std = float(std)
        self.enabled = bool(enabled)
        if not np.isfinite(self.mean):
            self.mean = 0.0
        if (not np.isfinite(self.std)) or self.std < 1e-8:
            self.std = 1.0

    def transform_tensor(self, y: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return y
        return (y - self.mean) / self.std

    def inverse_tensor(self, y: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return y
        return y * self.std + self.mean

    def inverse_numpy(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        if not self.enabled:
            return y
        return y * self.std + self.mean

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "mean": float(self.mean),
            "std": float(self.std),
        }


def build_label_scaler_from_records(records: List[Dict[str, Any]]) -> LabelScaler:
    use_label_standardization = bool(TRAIN_CONFIG.get("use_label_standardization", True))
    label_col = get_label_col()

    values = []
    for rec in records:
        try:
            values.append(float(rec[label_col]))
        except Exception:
            continue

    if len(values) == 0:
        raise RuntimeError("Cannot build label scaler: no valid ddG values in training records.")

    arr = np.asarray(values, dtype=np.float32)
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    scaler = LabelScaler(mean=mean, std=std, enabled=use_label_standardization)
    return scaler


def label_scaler_from_checkpoint(ckpt: Dict[str, Any]) -> LabelScaler:
    """
    从 checkpoint 恢复标签标准化参数。
    兼容旧 checkpoint：如果没有 label_scaler 字段，则认为未标准化。
    """
    state = ckpt.get("label_scaler", None)
    if isinstance(state, dict):
        return LabelScaler(
            mean=float(state.get("mean", 0.0)),
            std=float(state.get("std", 1.0)),
            enabled=bool(state.get("enabled", True)),
        )

    if "label_mean" in ckpt or "label_std" in ckpt:
        return LabelScaler(
            mean=float(ckpt.get("label_mean", 0.0)),
            std=float(ckpt.get("label_std", 1.0)),
            enabled=bool(ckpt.get("label_normalized", True)),
        )

    return LabelScaler(mean=0.0, std=1.0, enabled=False)


# =========================================================
# 保存 checkpoint
# =========================================================
def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        path: Path,
        label_scaler: Optional[LabelScaler] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if label_scaler is None:
        label_scaler = LabelScaler(enabled=False)

    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": dict(metrics),
            "label_scaler": label_scaler.state_dict(),
            "label_mean": float(label_scaler.mean),
            "label_std": float(label_scaler.std),
            "label_normalized": bool(label_scaler.enabled),
        },
        path,
    )


# =========================================================
# 读取原始 CSV
# =========================================================
def load_full_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Original CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=DATA_CONFIG.get("csv_sep", ","))
    id_col = get_id_col()
    label_col = get_label_col()

    required_cols = {id_col, label_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Original CSV missing required columns: {sorted(missing_cols)} | path={csv_path}"
        )

    df = df.copy()
    df[id_col] = df[id_col].astype(str).str.strip()
    df["_norm_id"] = df[id_col].map(normalize_split_key)

    if df["_norm_id"].eq("").any():
        bad_n = int(df["_norm_id"].eq("").sum())
        raise ValueError(f"Original CSV contains {bad_n} empty ID values in column '{id_col}'")

    if df["_norm_id"].duplicated().any():
        dup_ids = df.loc[df["_norm_id"].duplicated(), "_norm_id"].tolist()
        raise ValueError(
            f"Original CSV has duplicated IDs, cannot map uniquely to ID.pt. "
            f"First 10 duplicates: {dup_ids[:10]}"
        )

    logging.info(f"Loaded original dataframe: {csv_path}, total rows={len(df)}")
    return df


# =========================================================
# 图缓存数据集
# =========================================================
class GraphCacheDatasetByID(Dataset):
    def __init__(
            self,
            records: List[Dict[str, Any]],
            GRAPH_DIR: Path,
            preload_in_memory: bool = False,
    ):
        self.records = records
        self.GRAPH_DIR = Path(GRAPH_DIR)
        self.preload_in_memory = bool(preload_in_memory)
        self.data_list: Optional[List[Dict[str, Any]]] = None
        self.id_col = get_id_col()
        self.label_col = get_label_col()

        if self.preload_in_memory:
            self.data_list = []
            for idx, rec in enumerate(self.records):
                sample = self._load_sample_from_record(rec)
                self.data_list.append(sample)
                if (idx + 1) % 100 == 0 or (idx + 1) == len(self.records):
                    logging.info(f"Preloaded split samples: {idx + 1}/{len(self.records)}")

        logging.info(
            f"GraphCacheDatasetByID ready. samples={len(self.records)}, "
            f"GRAPH_DIR={self.GRAPH_DIR}, "
            f"preload_in_memory={self.preload_in_memory}"
        )

    def _resolve_pt_path(self, record: Dict[str, Any]) -> Path:
        sample_id = normalize_split_key(record.get(self.id_col, ""))
        if not sample_id:
            raise ValueError(f"Missing ID value in record: {record}")

        pt_path = self.GRAPH_DIR / f"{sample_id}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Graph cache not found: {pt_path}")
        return pt_path

    def _load_sample_from_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        pt_path = self._resolve_pt_path(record)
        sample = torch.load(pt_path, map_location="cpu")
        if not isinstance(sample, dict):
            raise TypeError(f"Loaded object is not a dict: {pt_path}")

        sample = dict(sample)
        mut_id = normalize_split_key(record.get(self.id_col, ""))
        label_val = float(record[self.label_col])

        sample["sample_id"] = mut_id
        sample["mut_pdb_id"] = mut_id
        sample["ddg"] = torch.tensor(label_val, dtype=torch.float32)
        return sample

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.data_list is not None:
            return self.data_list[idx]
        return self._load_sample_from_record(self.records[idx])


# =========================================================
# collate
# =========================================================
def collate_fn_joint(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(samples) == 0:
        raise ValueError("Empty batch in collate_fn_joint")

    batch = {
        "sample_id": [s.get("sample_id", "") for s in samples],
        "wt_pdb_id": [s.get("wt_pdb_id", "") for s in samples],
        "mut_pdb_id": [s.get("mut_pdb_id", "") for s in samples],
        "mutation_str": [s.get("mutation_str", "") for s in samples],
        "partners_str": [s.get("partners_str", "") for s in samples],
        "wt_joint_graph": Batch.from_data_list([s["wt_joint_graph"] for s in samples]),
        "mut_joint_graph": Batch.from_data_list([s["mut_joint_graph"] for s in samples]),
        "ddg": torch.stack([s["ddg"] for s in samples], dim=0),
    }

    optional_passthrough_keys = [
        "mutations",
        "ab_chains",
        "ag_chains",
        "ab_muts",
        "ag_muts",
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
# DataLoader
# =========================================================
def build_dataloader(dataset: Dataset, shuffle: bool, seed: Optional[int] = None) -> DataLoader:
    configured_workers = int(TRAIN_CONFIG.get("num_workers", 0))
    cpu_fraction = float(TRAIN_CONFIG.get("max_cpu_fraction", 0.70))
    max_cpu_workers = max(1, int(math.floor((os.cpu_count() or 1) * max(0.01, min(cpu_fraction, 1.0)))))
    num_workers = min(configured_workers, max_cpu_workers)
    pin_memory = torch.cuda.is_available()

    generator = torch.Generator()
    if seed is None:
        seed = int(TRAIN_CONFIG.get("seed", 42))
    generator.manual_seed(int(seed))

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(TRAIN_CONFIG["batch_size"]),
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": collate_fn_joint,
        "drop_last": False,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(**loader_kwargs)


# =========================================================
# batch 搬运到设备
# =========================================================
def _move_nested_to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    if isinstance(x, list):
        return [_move_nested_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(_move_nested_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: _move_nested_to_device(v, device) for k, v in x.items()}
    return x


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    graph_keys = ["wt_joint_graph", "mut_joint_graph"]
    for key in graph_keys:
        if key in batch and hasattr(batch[key], "to"):
            batch[key] = batch[key].to(device, non_blocking=True)

    for key, value in list(batch.items()):
        if key in graph_keys:
            continue
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
        elif isinstance(value, (list, tuple, dict)):
            batch[key] = _move_nested_to_device(value, device)

    return batch


# =========================================================
# 训练时动态消融
# =========================================================
def apply_ablation_to_graph(graph: Any, ablation_config: Dict[str, Any]) -> Any:
    if graph is None or not hasattr(graph, "x") or graph.x is None:
        return graph

    use_plm = bool(ablation_config.get("use_plm", True))
    use_dssp = bool(ablation_config.get("use_dssp", True))
    use_interface = bool(ablation_config.get("use_interface_modeling", True))
    use_plm_only = bool(ablation_config.get("use_plm_only", False))

    if isinstance(graph.x, torch.Tensor):
        graph.x = graph.x.clone()

        if use_plm_only and graph.x.dim() == 2:
            # PLM-only baseline：只保留 side-specific PLM 数值输入。
            # mutation / antibody / antigen flag 不作为 MLP 输入；它们只在 model.py 中用于选择 pooling 子集。
            # 因此这里保留 23:26 三个选择用 flag 和 31:159 PLM，其余置零，便于调试和防止意外特征泄漏。
            if graph.x.size(1) >= 31:
                keep = torch.zeros_like(graph.x)
                keep[:, 23:26] = graph.x[:, 23:26]
                keep[:, 31:159] = graph.x[:, 31:159]
                graph.x = keep
        else:
            if (not use_plm) and graph.x.dim() == 2 and graph.x.size(1) >= 159:
                graph.x[:, 31:159] = 0.0

            if (not use_dssp) and graph.x.dim() == 2 and graph.x.size(1) >= 31:
                graph.x[:, 27:31] = 0.0

            if (not use_interface) and graph.x.dim() == 2 and graph.x.size(1) >= 27:
                graph.x[:, 26] = 0.0

    if use_plm_only:
        # PLM-only 不使用任何边信息；保留 edge_index 形状以兼容 PyG Batch，但边特征置零。
        if hasattr(graph, "edge_attr") and isinstance(graph.edge_attr, torch.Tensor):
            graph.edge_attr = torch.zeros_like(graph.edge_attr)

    if not use_interface:
        if hasattr(graph, "edge_attr") and isinstance(graph.edge_attr, torch.Tensor):
            graph.edge_attr = graph.edge_attr.clone()
            if graph.edge_attr.dim() == 2 and graph.edge_attr.size(1) >= 6:
                graph.edge_attr[:, 4] = 0.0
                graph.edge_attr[:, 5] = 0.0

        if hasattr(graph, "interface_idx"):
            device = graph.x.device if isinstance(getattr(graph, "x", None), torch.Tensor) else torch.device("cpu")
            graph.interface_idx = torch.zeros((0,), dtype=torch.long, device=device)

        if hasattr(graph, "interface_mask") and isinstance(graph.interface_mask, torch.Tensor):
            graph.interface_mask = torch.zeros_like(graph.interface_mask, dtype=torch.bool)

    return graph


def apply_ablation_to_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    for key in ["wt_joint_graph", "mut_joint_graph"]:
        if key in batch:
            batch[key] = apply_ablation_to_graph(batch[key], ABLATION_CONFIG)
    return batch


# =========================================================
# Loss
# =========================================================
def build_criterion() -> nn.Module:
    loss_type = str(TRAIN_CONFIG.get("loss_type", "mse")).lower()
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "huber":
        return nn.HuberLoss(delta=float(TRAIN_CONFIG.get("huber_delta", 1.0)))
    if loss_type in {"smoothl1", "smooth_l1"}:
        return nn.SmoothL1Loss(beta=float(TRAIN_CONFIG.get("huber_delta", 1.0)))
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def build_contact_delta_l2_loss(
        components: Dict[str, torch.Tensor],
        device: torch.device,
) -> torch.Tensor:
    """V1.3: 对 raw contact_delta_effect 加 L2 约束。

    注意：这里正则的是 model.py 返回的 raw ``contact_delta_effect``，
    而不是已经与 old_contact_effect 合成后的 ``changed_interface_contact_effect``。
    """
    if not bool(TRAIN_CONFIG.get("use_contact_delta_l2", False)):
        return torch.zeros((), dtype=torch.float32, device=device)

    lam = float(TRAIN_CONFIG.get("contact_delta_l2_lambda", 0.0))
    if lam <= 0:
        return torch.zeros((), dtype=torch.float32, device=device)

    name = str(TRAIN_CONFIG.get(
        "contact_delta_l2_component_name",
        "contact_delta_effect",
    ))

    comp = components.get(name, None)
    if comp is None or not isinstance(comp, torch.Tensor) or comp.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=device)

    comp = comp.view(-1).float()
    return lam * torch.mean(comp ** 2)


# =========================================================
# 模型输出兼容
# =========================================================
def extract_model_prediction(model_out: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """兼容旧模型 Tensor 输出和新版 dict 输出。"""
    if isinstance(model_out, dict):
        if "pred" not in model_out:
            raise KeyError("Model output dict must contain key 'pred'.")
        pred = model_out["pred"]
        components = {
            k: v for k, v in model_out.items()
            if k != "pred" and isinstance(v, torch.Tensor)
        }
        return pred, components
    return model_out, {}


# =========================================================
# 训练 / 验证
# =========================================================
def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        label_scaler: Optional[LabelScaler] = None,
) -> Dict[str, float]:
    model.train()
    if label_scaler is None:
        label_scaler = LabelScaler(enabled=False)

    total_loss = 0.0
    total_task_loss = 0.0
    total_contact_delta_l2 = 0.0
    all_y_true = []
    all_y_pred = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        batch = apply_ablation_to_batch(batch)

        optimizer.zero_grad()
        y_true_raw = batch["ddg"].view(-1)
        y_true_train = label_scaler.transform_tensor(y_true_raw)
        model_out = model(batch)
        y_pred_train, components = extract_model_prediction(model_out)
        y_pred_train = y_pred_train.view(-1)

        task_loss = criterion(y_pred_train, y_true_train)
        contact_delta_l2 = build_contact_delta_l2_loss(
            components=components,
            device=y_pred_train.device,
        )
        loss = task_loss + contact_delta_l2
        loss.backward()

        grad_clip_norm = TRAIN_CONFIG.get("grad_clip_norm", None)
        if grad_clip_norm is not None and float(grad_clip_norm) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))

        optimizer.step()

        y_pred_raw = label_scaler.inverse_tensor(y_pred_train)

        batch_size = y_true_raw.size(0)
        total_loss += float(loss.item()) * batch_size
        total_task_loss += float(task_loss.item()) * batch_size
        total_contact_delta_l2 += float(contact_delta_l2.item()) * batch_size
        all_y_true.append(y_true_raw.detach().cpu().numpy())
        all_y_pred.append(y_pred_raw.detach().cpu().numpy())

    y_true_arr = np.concatenate(all_y_true) if all_y_true else np.array([])
    y_pred_arr = np.concatenate(all_y_pred) if all_y_pred else np.array([])
    metrics = compute_metrics(y_true_arr, y_pred_arr)
    denom = max(len(dataloader.dataset), 1)
    metrics["loss"] = total_loss / denom
    metrics["task_loss"] = total_task_loss / denom
    metrics["contact_delta_l2"] = total_contact_delta_l2 / denom
    metrics["label_normalized"] = bool(label_scaler.enabled)
    metrics["label_mean"] = float(label_scaler.mean)
    metrics["label_std"] = float(label_scaler.std)
    return metrics


@torch.no_grad()
def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        label_scaler: Optional[LabelScaler] = None,
):
    model.eval()
    if label_scaler is None:
        label_scaler = LabelScaler(enabled=False)

    total_loss = 0.0
    all_y_true = []
    all_y_pred = []
    all_ids = []
    all_components: Dict[str, List[np.ndarray]] = {}

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        batch = apply_ablation_to_batch(batch)
        y_true_raw = batch["ddg"].view(-1)
        y_true_eval = label_scaler.transform_tensor(y_true_raw)
        model_out = model(batch)
        y_pred_eval, components = extract_model_prediction(model_out)
        y_pred_eval = y_pred_eval.view(-1)

        loss = criterion(y_pred_eval, y_true_eval)
        y_pred_raw = label_scaler.inverse_tensor(y_pred_eval)

        for comp_name, comp_tensor in components.items():
            comp = comp_tensor.view(-1)
            if label_scaler.enabled:
                comp = comp * float(label_scaler.std)
            all_components.setdefault(comp_name, []).append(comp.detach().cpu().numpy())
        total_loss += float(loss.item()) * y_true_raw.size(0)

        all_y_true.append(y_true_raw.detach().cpu().numpy())
        all_y_pred.append(y_pred_raw.detach().cpu().numpy())
        all_ids.extend(batch["sample_id"])

    y_true_arr = np.concatenate(all_y_true) if all_y_true else np.array([])
    y_pred_arr = np.concatenate(all_y_pred) if all_y_pred else np.array([])

    metrics = compute_metrics(y_true_arr, y_pred_arr)
    metrics["loss"] = total_loss / max(len(dataloader.dataset), 1)
    metrics["label_normalized"] = bool(label_scaler.enabled)
    metrics["label_mean"] = float(label_scaler.mean)
    metrics["label_std"] = float(label_scaler.std)
    component_arrays = {
        name: np.concatenate(values) if values else np.array([])
        for name, values in all_components.items()
    }
    return metrics, y_true_arr, y_pred_arr, all_ids, component_arrays


# =========================================================
# split -> records
# =========================================================
def build_fold_datasets(
        full_df: pd.DataFrame,
        GRAPH_DIR: Path,
        train_csv_path: Path,
        valid_csv_path: Path,
        preload_split_in_memory: bool = False,
):
    train_df = pd.read_csv(train_csv_path)
    valid_df = pd.read_csv(valid_csv_path)
    id_col = get_id_col()

    record_by_id = {}
    for rec in full_df.to_dict("records"):
        norm_id = normalize_split_key(rec.get(id_col, ""))
        if norm_id:
            record_by_id[norm_id] = rec

    def extract_keys(split_df: pd.DataFrame, split_name: str) -> List[str]:
        if id_col in split_df.columns:
            raw_keys = split_df[id_col].tolist()
        elif "ID" in split_df.columns:
            raw_keys = split_df["ID"].tolist()
        elif "sample_id" in split_df.columns:
            raw_keys = split_df["sample_id"].tolist()
        else:
            raise ValueError(
                f"{split_name} split file must contain '{id_col}', 'ID', or 'sample_id'. "
                f"Got columns: {split_df.columns.tolist()}"
            )
        return [normalize_split_key(x) for x in raw_keys if normalize_split_key(x)]

    def collect_records(split_df: pd.DataFrame, split_name: str) -> List[Dict[str, Any]]:
        keys = extract_keys(split_df, split_name)
        selected = []
        missed = []
        for k in keys:
            rec = record_by_id.get(k, None)
            if rec is not None:
                selected.append(rec)
            else:
                missed.append(k)
        if missed:
            raise ValueError(
                f"{split_name}: {len(missed)} samples not found in original CSV. "
                f"First 10 missed keys: {missed[:10]}"
            )
        return selected

    train_records = collect_records(train_df, "train")
    valid_records = collect_records(valid_df, "valid")

    raw_train_n = len(train_records)
    raw_valid_n = len(valid_records)

    train_records = filter_records_by_ddg_range(
        train_records,
        min_ddg_exclusive=-8.0,
        max_ddg_exclusive=8.0,
    )
    valid_records = filter_records_by_ddg_range(
        valid_records,
        min_ddg_exclusive=-8.0,
        max_ddg_exclusive=8.0,
    )

    logging.info(
        f"ddG filter applied: keep -8 < ddG < 8 | "
        f"train {raw_train_n} -> {len(train_records)}, "
        f"valid {raw_valid_n} -> {len(valid_records)}"
    )

    if len(train_records) == 0:
        raise RuntimeError(f"Train subset is empty after ddG filtering: {train_csv_path}")
    if len(valid_records) == 0:
        raise RuntimeError(f"Valid subset is empty after ddG filtering: {valid_csv_path}")

    train_dataset = GraphCacheDatasetByID(
        records=train_records,
        GRAPH_DIR=GRAPH_DIR,
        preload_in_memory=preload_split_in_memory,
    )
    valid_dataset = GraphCacheDatasetByID(
        records=valid_records,
        GRAPH_DIR=GRAPH_DIR,
        preload_in_memory=preload_split_in_memory,
    )
    return train_dataset, valid_dataset, train_records, valid_records


# =========================================================
# 监控比较逻辑
# =========================================================
def is_better(current: float, best: float, mode: str) -> bool:
    if mode == "min":
        return current < best
    if mode == "max":
        return current > best
    raise ValueError(f"Unsupported monitor mode: {mode}")


def compute_composite_score(metrics: Dict[str, float]) -> float:
    """
    Exp4: checkpoint selection by composite score.

    composite = rmse - alpha * pearson

    alpha 读取 TRAIN_CONFIG["composite_pearson_weight"]，默认 0.5。
    越小越好。
    """
    rmse = float(metrics.get("rmse", 0.0))
    pearson = float(metrics.get("pearson", 0.0))
    alpha = float(TRAIN_CONFIG.get("composite_pearson_weight", 0.5))
    return rmse - alpha * pearson


def build_checkpoint_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    out = dict(metrics)
    out["composite"] = compute_composite_score(metrics)
    out["composite_pearson_weight"] = float(TRAIN_CONFIG.get("composite_pearson_weight", 0.5))
    return out


def checkpoint_is_better(kind: str, metrics: Dict[str, float], best_value: float) -> bool:
    kind = str(kind).lower().strip()

    if kind == "best_composite":
        return is_better(float(metrics["composite"]), best_value, "min")

    raise ValueError(f"Unsupported checkpoint kind: {kind}. This version only saves best_composite.")


# =========================================================
# 预测落盘
# =========================================================
def save_predictions(
        fold_idx: int,
        epoch: int,
        sample_ids: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        components: Optional[Dict[str, np.ndarray]] = None,
        tag: str = "best",
) -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    tag = str(tag).strip()
    if tag:
        out_path = PRED_DIR / f"fold_{fold_idx:02d}_{tag}_epoch_{epoch:03d}_valid_predictions.csv"
    else:
        out_path = PRED_DIR / f"fold_{fold_idx:02d}_epoch_{epoch:03d}_valid_predictions.csv"

    df = pd.DataFrame({
        "sample_id": sample_ids,
        "y_true": y_true.reshape(-1),
        "y_pred": y_pred.reshape(-1),
    })
    if components:
        for name, arr in components.items():
            arr = np.asarray(arr).reshape(-1)
            if arr.shape[0] == len(df):
                df[name] = arr
    df.to_csv(out_path, index=False, encoding="utf-8-sig")


# =========================================================
# 单折训练
# =========================================================
def train_one_fold(
        fold_idx: int,
        device: torch.device,
        full_df: pd.DataFrame,
        GRAPH_DIR: Path,
) -> Dict[str, Any]:
    fold_seed = int(TRAIN_CONFIG.get("seed", 42)) + int(fold_idx)
    deterministic = bool(TRAIN_CONFIG.get("deterministic", True))
    set_seed(fold_seed, deterministic=deterministic)

    logging.info(
        f"[Fold {fold_idx}] Using fold-specific seed: {fold_seed} "
        f"(deterministic={deterministic})"
    )

    fold_prefix = str(RUN_CONFIG.get("cv_split_naming", "fold"))
    train_name = str(RUN_CONFIG.get("train_csv_name", "train.csv"))
    valid_name = str(RUN_CONFIG.get("valid_csv_name", "valid.csv"))

    split_dir = Path(DATA_CONFIG["split_dir"]) / f"{fold_prefix}_{fold_idx}"
    train_csv_path = split_dir / train_name
    valid_csv_path = split_dir / valid_name
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Train split file not found: {train_csv_path}")
    if not valid_csv_path.exists():
        raise FileNotFoundError(f"Valid split file not found: {valid_csv_path}")

    preload_split_in_memory = bool(TRAIN_CONFIG.get("preload_split_in_memory", False))
    train_dataset, valid_dataset, train_records, _valid_records = build_fold_datasets(
        full_df=full_df,
        GRAPH_DIR=GRAPH_DIR,
        train_csv_path=train_csv_path,
        valid_csv_path=valid_csv_path,
        preload_split_in_memory=preload_split_in_memory,
    )
    logging.info(
        f"[Fold {fold_idx}] Train samples: {len(train_dataset)}, Valid samples: {len(valid_dataset)}"
    )

    label_scaler = build_label_scaler_from_records(train_records)
    logging.info(
        f"[Fold {fold_idx}] Label standardization: enabled={label_scaler.enabled} | "
        f"mean={label_scaler.mean:.6f}, std={label_scaler.std:.6f}"
    )

    train_loader = build_dataloader(train_dataset, shuffle=bool(TRAIN_CONFIG.get("shuffle", True)), seed=fold_seed)
    valid_loader = build_dataloader(valid_dataset, shuffle=False, seed=fold_seed)

    model = ddGModel().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(TRAIN_CONFIG.get("lr", 1e-3)),
        weight_decay=float(TRAIN_CONFIG.get("weight_decay", 1e-4)),
    )
    criterion = build_criterion()

    epochs = int(TRAIN_CONFIG.get("epochs", 100))
    # 只保留 best_composite：early stopping / summary / checkpoint 统一按 composite 选择。
    monitor_metric = "composite"
    monitor_mode = "min"
    use_early_stopping = bool(TRAIN_CONFIG.get("use_early_stopping", True))
    patience = int(TRAIN_CONFIG.get("early_stopping_patience", TRAIN_CONFIG.get("early_stopping_patience", 20)))

    best_monitor = float("inf") if monitor_mode == "min" else float("-inf")
    best_epoch = 0
    best_valid_metrics: Optional[Dict[str, float]] = None
    best_pred = None
    bad_epochs = 0

    # 只维护 best_composite checkpoint；不再保存 best_rmse / best_pearson。
    checkpoint_best_values = {
        "best_composite": float("inf"),
    }
    checkpoint_best_epochs = {
        "best_composite": 0,
    }
    checkpoint_best_metrics: Dict[str, Optional[Dict[str, float]]] = {
        "best_composite": None,
    }

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        _ = train_one_epoch(model, train_loader, optimizer, criterion, device, label_scaler=label_scaler)
        valid_metrics, y_true, y_pred, sample_ids, components = evaluate(
            model, valid_loader, criterion, device, label_scaler=label_scaler
        )
        valid_metrics = build_checkpoint_metrics(valid_metrics)
        epoch_time = time.time() - epoch_start

        current_monitor = float(valid_metrics.get(monitor_metric, valid_metrics.get("rmse", 0.0)))
        logging.info(
            f"[Fold {fold_idx} | Epoch {epoch:03d}] "
            f"Valid Loss={valid_metrics['loss']:.4f} "
            f"RMSE={valid_metrics['rmse']:.4f} "
            f"MAE={valid_metrics['mae']:.4f} "
            f"Pearson={valid_metrics['pearson']:.4f} "
            f"Composite={valid_metrics['composite']:.4f} | "
            f"Time={epoch_time:.2f}s"
        )

        # 1) 保留原来的 early stopping / legacy best 逻辑
        if is_better(current_monitor, best_monitor, monitor_mode):
            best_monitor = current_monitor
            best_epoch = epoch
            best_valid_metrics = copy.deepcopy(valid_metrics)
            best_pred = (sample_ids, y_true.copy(), y_pred.copy())
            bad_epochs = 0

            # 不再保存 legacy fold_xx_best.pt / fold_xx_best_*_valid_predictions.csv，
            # 避免除 best_composite 之外的额外输出。
        else:
            bad_epochs += 1

        # 2) 只保存 best_composite
        for ckpt_kind in ["best_composite"]:
            if checkpoint_is_better(ckpt_kind, valid_metrics, checkpoint_best_values[ckpt_kind]):
                checkpoint_best_values[ckpt_kind] = float(valid_metrics["composite"])

                checkpoint_best_epochs[ckpt_kind] = epoch
                checkpoint_best_metrics[ckpt_kind] = copy.deepcopy(valid_metrics)

                ckpt_path = CKPT_DIR / f"fold_{fold_idx:02d}_{ckpt_kind}.pt"
                save_checkpoint(model, optimizer, epoch, valid_metrics, ckpt_path, label_scaler=label_scaler)
                save_predictions(
                    fold_idx,
                    epoch,
                    sample_ids,
                    y_true,
                    y_pred,
                    components=components,
                    tag=ckpt_kind,
                )

                logging.info(
                    f"[Fold {fold_idx} | Epoch {epoch:03d}] "
                    f"Saved {ckpt_kind} checkpoint | "
                    f"RMSE={valid_metrics['rmse']:.4f} "
                    f"Pearson={valid_metrics['pearson']:.4f} "
                    f"Composite={valid_metrics['composite']:.4f} | "
                    f"path={ckpt_path}"
                )

        if use_early_stopping and bad_epochs >= patience:
            logging.info(
                f"[Fold {fold_idx}] Early stopping at epoch {epoch}. "
                f"Best monitor epoch={best_epoch}, best_{monitor_metric}={best_monitor:.4f} | "
                f"best_composite_epoch={checkpoint_best_epochs['best_composite']}"
            )
            break

    if best_valid_metrics is None:
        raise RuntimeError(f"[Fold {fold_idx}] No valid metrics were produced.")

    return {
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_metrics": best_valid_metrics,
        "best_monitor": best_monitor,
        "best_pred": best_pred,
        "checkpoint_best_epochs": checkpoint_best_epochs,
        "checkpoint_best_metrics": checkpoint_best_metrics,
    }


# =========================================================
# 多 GPU worker
# =========================================================
def gpu_worker(
        worker_gpu: int,
        fold_queue: mp.Queue,
        result_queue: mp.Queue,
        full_df: pd.DataFrame,
        GRAPH_DIR: Path,
) -> None:
    set_parent_death_signal(signal.SIGTERM)
    try:
        setup_logger()
        configure_cpu_limit()

        if torch.cuda.is_available():
            torch.cuda.set_device(worker_gpu)
            device = torch.device(f"cuda:{worker_gpu}")
        else:
            device = torch.device("cpu")

        logging.info(f"[Worker GPU {worker_gpu}] started on device={device}")

        while True:
            try:
                fold_idx = fold_queue.get_nowait()
            except queue.Empty:
                break

            try:
                logging.info(f"[Worker GPU {worker_gpu}] start Fold {fold_idx}")
                result = train_one_fold(fold_idx, device, full_df, GRAPH_DIR)
                result_queue.put(("ok", result))
                logging.info(f"[Worker GPU {worker_gpu}] done Fold {fold_idx}")
            except Exception as e:
                result_queue.put(("error", {"fold": fold_idx, "gpu": worker_gpu, "error": repr(e)}))
                logging.exception(f"[Worker GPU {worker_gpu}] Fold {fold_idx} failed")
    except Exception as e:
        result_queue.put(("error", {"fold": None, "gpu": worker_gpu, "error": repr(e)}))
        logging.exception(f"[Worker GPU {worker_gpu}] worker failed to start")


# =========================================================
# 汇总
# =========================================================
def summarize_cv_results(results: List[Dict[str, Any]]) -> None:
    if len(results) == 0:
        logging.warning("No CV results to summarize.")
        return

    rows = []
    for r in sorted(results, key=lambda x: x["fold"]):
        m = r["best_metrics"]
        row = {
            "fold": r["fold"],
            "best_epoch": r["best_epoch"],
            "rmse": float(m["rmse"]),
            "mae": float(m["mae"]),
            "pearson": float(m["pearson"]),
            "loss": float(m["loss"]),
            "composite": float(m.get("composite", compute_composite_score(m))),
        }

        ckpt_epochs = r.get("checkpoint_best_epochs", {}) or {}
        ckpt_metrics = r.get("checkpoint_best_metrics", {}) or {}

        for kind in ["best_composite"]:
            km = ckpt_metrics.get(kind, None)
            row[f"{kind}_epoch"] = ckpt_epochs.get(kind, None)

            if isinstance(km, dict):
                row[f"{kind}_rmse"] = float(km.get("rmse", 0.0))
                row[f"{kind}_mae"] = float(km.get("mae", 0.0))
                row[f"{kind}_pearson"] = float(km.get("pearson", 0.0))
                row[f"{kind}_loss"] = float(km.get("loss", 0.0))
                row[f"{kind}_composite"] = float(km.get("composite", compute_composite_score(km)))

        rows.append(row)

    df = pd.DataFrame(rows)
    summary_path = PRED_DIR / "cv_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    logging.info("=" * 80)
    logging.info("Cross-validation summary")
    for _, row in df.iterrows():
        logging.info(
            f"Fold {int(row['fold'])}: best_epoch={int(row['best_epoch'])} | "
            f"RMSE={row['rmse']:.4f} MAE={row['mae']:.4f} "
            f"Pearson={row['pearson']:.4f} Composite={row['composite']:.4f}"
        )

    logging.info(
        f"Mean ± Std | RMSE={df['rmse'].mean():.4f} ± {df['rmse'].std(ddof=0):.4f} | "
        f"MAE={df['mae'].mean():.4f} ± {df['mae'].std(ddof=0):.4f} | "
        f"Pearson={df['pearson'].mean():.4f} ± {df['pearson'].std(ddof=0):.4f} | "
        f"Composite={df['composite'].mean():.4f} ± {df['composite'].std(ddof=0):.4f}"
    )

    if "best_composite_rmse" in df.columns:
        logging.info("-" * 80)
        logging.info("Checkpoint selector summary")
        for kind in ["best_composite"]:
            rmse_col = f"{kind}_rmse"
            mae_col = f"{kind}_mae"
            pearson_col = f"{kind}_pearson"
            comp_col = f"{kind}_composite"
            epoch_col = f"{kind}_epoch"

            if rmse_col not in df.columns:
                continue

            logging.info(
                f"{kind}: "
                f"epoch_mean={df[epoch_col].mean():.2f} | "
                f"RMSE={df[rmse_col].mean():.4f} ± {df[rmse_col].std(ddof=0):.4f} | "
                f"MAE={df[mae_col].mean():.4f} ± {df[mae_col].std(ddof=0):.4f} | "
                f"Pearson={df[pearson_col].mean():.4f} ± {df[pearson_col].std(ddof=0):.4f} | "
                f"Composite={df[comp_col].mean():.4f} ± {df[comp_col].std(ddof=0):.4f}"
            )

    logging.info(f"Saved CV summary to: {summary_path}")
    logging.info("=" * 80)


# =========================================================
# main
# =========================================================
def main() -> None:
    setup_logger()
    install_main_signal_handlers()
    configure_cpu_limit()

    csv_path = Path(DATA_CONFIG["csv_path"])
    split_dir = Path(DATA_CONFIG["split_dir"])
    GRAPH_DIR = get_GRAPH_DIR()

    full_df = load_full_dataframe(csv_path)

    logging.info("=" * 80)
    logging.info("Start training ddG model (V1.3: V1.1 residual contact-delta scale=0.15 + contact_delta_effect L2)")
    logging.info(f"Seed: {TRAIN_CONFIG.get('seed', 42)}")
    logging.info(f"Deterministic: {TRAIN_CONFIG.get('deterministic', True)}")
    logging.info(f"Original CSV: {csv_path}")
    logging.info(f"Split dir: {split_dir}")
    logging.info(f"Graph cache dir: {GRAPH_DIR}")
    logging.info(f"Ablation: {ABLATION_CONFIG.get('name', 'none')} | tag={ABLATION_CONFIG.get('tag', 'full_model')}")
    logging.info(f"Ablation config: {ABLATION_CONFIG}")
    logging.info(f"Train config: {TRAIN_CONFIG}")
    logging.info(f"Composite selector: rmse - {TRAIN_CONFIG.get('composite_pearson_weight', 0.5)} * pearson")
    logging.info("=" * 80)

    num_folds = int(TRAIN_CONFIG.get("num_folds", 5))
    use_cuda = bool(TRAIN_CONFIG.get("use_cuda", True)) and torch.cuda.is_available()
    parallel_folds = bool(TRAIN_CONFIG.get("parallel_folds", False))

    if not use_cuda:
        logging.info("CUDA unavailable or disabled, fallback to single-process CPU/CUDA default device.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = []
        for fold_idx in range(1, num_folds + 1):
            results.append(train_one_fold(fold_idx, device, full_df, GRAPH_DIR))
        summarize_cv_results(results)
        return

    visible_gpu_count = torch.cuda.device_count()
    visible_gpus = list(range(visible_gpu_count))
    logging.info(f"Visible physical GPUs: {visible_gpus}")

    configured_gpu_list = list(TRAIN_CONFIG.get("gpu_list", [0]))
    worker_gpus = [g for g in configured_gpu_list if isinstance(g, int) and 0 <= g < visible_gpu_count]
    if len(worker_gpus) == 0:
        worker_gpus = [0]
    logging.info(f"Worker CUDA indices: {worker_gpus}")
    logging.info(f"Num folds: {num_folds}")

    if not parallel_folds or len(worker_gpus) == 1:
        device = torch.device(f"cuda:{worker_gpus[0]}")
        results = []
        for fold_idx in range(1, num_folds + 1):
            results.append(train_one_fold(fold_idx, device, full_df, GRAPH_DIR))
        summarize_cv_results(results)
        return

    ctx = mp.get_context("spawn")
    fold_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()
    for fold_idx in range(1, num_folds + 1):
        fold_queue.put(fold_idx)

    workers = []
    _CHILD_PROCESSES.clear()
    for gpu_idx in worker_gpus:
        p = ctx.Process(
            target=gpu_worker,
            args=(gpu_idx, fold_queue, result_queue, full_df, GRAPH_DIR),
        )
        p.start()
        workers.append(p)
        _CHILD_PROCESSES.append(p)

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    expected = num_folds
    received = 0
    while received < expected:
        try:
            status, payload = result_queue.get(timeout=30)
            received += 1
            if status == "ok":
                results.append(payload)
                logging.info(
                    f"Main process received fold result: {received}/{expected} | "
                    f"fold={payload.get('fold')}"
                )
            else:
                errors.append(payload)
                logging.error(f"Main process received worker error: {payload}")
        except queue.Empty:
            alive_states = [(p.pid, p.is_alive(), p.exitcode) for p in workers]
            logging.info(
                f"Waiting worker results... received={received}/{expected} | "
                f"worker_states={alive_states}"
            )
            if not any(p.is_alive() for p in workers):
                logging.error("All workers exited before all fold results were received.")
                break

    try:
        for p in workers:
            p.join()
    finally:
        # 如果主进程因为异常/信号准备退出，确保直接子进程被清理。
        if errors or received < expected:
            terminate_child_processes(workers)

    if errors:
        logging.error("Some folds failed:")
        for e in errors:
            logging.error(str(e))
        raise RuntimeError(f"{len(errors)} fold(s) failed. Check logs above.")

    summarize_cv_results(results)


if __name__ == "__main__":
    main()
