import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from config import TRAIN_CONFIG, COLUMN_CONFIG
from model import ddGModel
from train import (
    GraphCacheDatasetByID,
    build_dataloader,
    build_criterion,
    compute_metrics,
    normalize_split_key,
    label_scaler_from_checkpoint,
)

# train.py 新版有这些函数；这里做兼容导入，避免旧版 train.py 报错
try:
    from train import move_batch_to_device, apply_ablation_to_batch, extract_model_prediction
except Exception:  # pragma: no cover
    move_batch_to_device = None
    apply_ablation_to_batch = None
    extract_model_prediction = None


# =========================================================
# 默认交叉外测配置：命令行参数可覆盖 ABBIND SKEMPI S641 S877 S487
# =========================================================
PROJECT_ROOT = Path("/home/zhao/gwc/NEW3")
MODEL_DATASET = "ABBIND"
VAL_DATASET = "SARS"
ABLATION_TAG = "full_model"

DEFAULT_CKPT_KINDS = ["best_composite"]
VALID_CKPT_KINDS = ["best_composite"]


# =========================================================
# 基础工具
# =========================================================
def get_id_col() -> str:
    return str(COLUMN_CONFIG.get("mut_pdb_id", "ID"))


def get_label_col() -> str:
    return str(COLUMN_CONFIG.get("label", "ddG"))


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, float) and np.isnan(x):
            return default
        return float(x)
    except Exception:
        return default


def filter_ddg(df: pd.DataFrame, min_ddg: float = -8.0, max_ddg: float = 8.0) -> pd.DataFrame:
    """保持和训练阶段一致：只保留 min_ddg < ddG < max_ddg。"""
    label_col = get_label_col()
    df = df.copy()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col])
    df = df[(df[label_col] > float(min_ddg)) & (df[label_col] < float(max_ddg))].copy()
    return df


def check_missing_graphs(df: pd.DataFrame, graph_dir: Path) -> List[str]:
    """检查验证集每个 ID 是否存在对应的 ID.pt 图缓存。"""
    id_col = get_id_col()
    missing = []
    for x in df[id_col].astype(str).str.strip():
        sample_id = normalize_split_key(x)
        pt_path = graph_dir / f"{sample_id}.pt"
        if not pt_path.exists():
            missing.append(sample_id)
    return missing


def choose_device(gpu_id: Optional[int] = None) -> torch.device:
    """外测单进程单卡。gpu_id 不传时使用 TRAIN_CONFIG['gpu_list'] 第一张可用卡。"""
    use_cuda = bool(TRAIN_CONFIG.get("use_cuda", True)) and torch.cuda.is_available()
    if not use_cuda:
        return torch.device("cpu")

    visible_gpu_count = torch.cuda.device_count()
    if gpu_id is not None:
        if not (0 <= int(gpu_id) < visible_gpu_count):
            raise ValueError(f"Invalid gpu_id={gpu_id}; visible cuda devices={visible_gpu_count}")
        gpu = int(gpu_id)
    else:
        gpu = 0
        for g in TRAIN_CONFIG.get("gpu_list", [0]):
            if isinstance(g, int) and 0 <= g < visible_gpu_count:
                gpu = int(g)
                break

    torch.cuda.set_device(gpu)
    return torch.device(f"cuda:{gpu}")


def load_model_from_ckpt(ckpt_path: Path, device: torch.device, strict: bool = True):
    """加载单折 checkpoint。新版 interaction_delta_v1_contact 仍使用 ddGModel + model_state_dict。"""
    model = ddGModel().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"checkpoint missing model_state_dict: {ckpt_path}")

    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if (not strict) and (missing or unexpected):
        logging.warning(f"Non-strict checkpoint load | missing={missing} | unexpected={unexpected}")
    model.eval()
    return model, ckpt


def default_extract_model_prediction(model_out: Any) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """兼容旧 Tensor 输出和新版 dict 输出。"""
    if callable(extract_model_prediction):
        return extract_model_prediction(model_out)

    if isinstance(model_out, dict):
        if "pred" not in model_out:
            raise KeyError("Model output dict must contain key 'pred'.")
        pred = model_out["pred"]
        comps = {k: v for k, v in model_out.items() if k != "pred" and isinstance(v, torch.Tensor)}
        return pred, comps
    return model_out, {}


def default_move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    if callable(move_batch_to_device):
        return move_batch_to_device(batch, device)

    def move_obj(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if hasattr(x, "to") and callable(getattr(x, "to")):
            try:
                return x.to(device)
            except Exception:
                return x
        if isinstance(x, dict):
            return {k: move_obj(v) for k, v in x.items()}
        if isinstance(x, list):
            return [move_obj(v) for v in x]
        if isinstance(x, tuple):
            return tuple(move_obj(v) for v in x)
        return x

    return move_obj(batch)


def default_apply_ablation_to_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    if callable(apply_ablation_to_batch):
        return apply_ablation_to_batch(batch)
    return batch


def component_to_raw_scale(comp_tensor: torch.Tensor, label_scaler) -> np.ndarray:
    """
    模型分量处于训练标签尺度。
    如果 label standardization 开启：pred_raw = pred_z * std + mean；每个分量贡献量 = comp_z * std。
    label mean 作为 y_pred_offset 单独保存，不强行分摊到某个分量。
    """
    comp = comp_tensor.view(-1).detach()
    if bool(getattr(label_scaler, "enabled", False)):
        comp = comp * float(label_scaler.std)
    return comp.cpu().numpy()


@torch.no_grad()
def evaluate_external(
    model: torch.nn.Module,
    dataloader,
    criterion: torch.nn.Module,
    device: torch.device,
    label_scaler,
):
    """
    外部验证专用 evaluate：
    1) 兼容 interaction_delta_v0 / v1_contact 的 dict 输出；
    2) 保存 local/contact/calibration/contact_delta 等分量；
    3) 返回原始 ddG 尺度上的 y_pred。
    """
    model.eval()

    total_loss = 0.0
    all_y_true: List[np.ndarray] = []
    all_y_pred: List[np.ndarray] = []
    all_ids: List[str] = []
    all_components: Dict[str, List[np.ndarray]] = {}

    for batch in dataloader:
        batch = default_move_batch_to_device(batch, device)
        batch = default_apply_ablation_to_batch(batch)

        y_true_raw = batch["ddg"].view(-1)
        y_true_eval = label_scaler.transform_tensor(y_true_raw)

        model_out = model(batch)
        y_pred_eval, components = default_extract_model_prediction(model_out)
        y_pred_eval = y_pred_eval.view(-1)

        loss = criterion(y_pred_eval, y_true_eval)
        y_pred_raw = label_scaler.inverse_tensor(y_pred_eval)

        total_loss += float(loss.item()) * y_true_raw.size(0)
        all_y_true.append(y_true_raw.detach().cpu().numpy())
        all_y_pred.append(y_pred_raw.detach().cpu().numpy())
        all_ids.extend([str(x) for x in batch["sample_id"]])

        for name, tensor in components.items():
            all_components.setdefault(name, []).append(component_to_raw_scale(tensor, label_scaler))

    y_true_arr = np.concatenate(all_y_true) if all_y_true else np.array([], dtype=np.float32)
    y_pred_arr = np.concatenate(all_y_pred) if all_y_pred else np.array([], dtype=np.float32)
    metrics = compute_metrics(y_true_arr, y_pred_arr)
    metrics["loss"] = total_loss / max(len(dataloader.dataset), 1)
    metrics["label_normalized"] = bool(label_scaler.enabled)
    metrics["label_mean"] = float(label_scaler.mean)
    metrics["label_std"] = float(label_scaler.std)

    component_arrays = {
        name: np.concatenate(values) if values else np.array([], dtype=np.float32)
        for name, values in all_components.items()
    }
    return metrics, y_true_arr, y_pred_arr, all_ids, component_arrays


def component_stats(y_true: np.ndarray, components: Dict[str, np.ndarray]) -> Dict[str, float]:
    """输出每个分量的均值、标准差、与标签的 Pearson，便于诊断。"""
    out: Dict[str, float] = {}
    y = np.asarray(y_true).reshape(-1)
    for name, arr in components.items():
        a = np.asarray(arr).reshape(-1)
        if a.size != y.size:
            continue
        out[f"{name}_mean"] = float(np.mean(a)) if a.size else 0.0
        out[f"{name}_std"] = float(np.std(a)) if a.size else 0.0
        if a.size > 1 and np.std(a) > 1e-12 and np.std(y) > 1e-12:
            out[f"{name}_pearson_to_y"] = float(np.corrcoef(y, a)[0, 1])
        else:
            out[f"{name}_pearson_to_y"] = 0.0
    return out




def make_metric_stats(
    summary_df: pd.DataFrame,
    metric_names: Sequence[str],
    suffix: str = "",
    ddof: int = 0,
) -> Dict[str, Any]:
    """
    根据每折指标计算 mean / std / "mean ± std" 字符串。

    参数说明：
      metric_names=["rmse", "mae", "pearson"]
      suffix=""          -> 使用 rmse / mae / pearson
      suffix="_pred_neg" -> 使用 rmse_pred_neg / mae_pred_neg / pearson_pred_neg
      suffix="_true_neg" -> 使用 rmse_true_neg / mae_true_neg / pearson_true_neg

    ddof=0 表示总体标准差；论文表格常用这一种。
    如果你想用样本标准差，可改为 ddof=1。
    """
    out: Dict[str, Any] = {}
    for metric_name in metric_names:
        col = f"{metric_name}{suffix}"
        if col not in summary_df.columns:
            raise KeyError(f"Missing metric column in summary_df: {col}")

        values = pd.to_numeric(summary_df[col], errors="coerce").dropna().astype(float).values
        if values.size == 0:
            mean_val = 0.0
            std_val = 0.0
        else:
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=ddof))

        out[f"{col}_mean"] = mean_val
        out[f"{col}_std"] = std_val
        out[f"{col}_mean_std"] = f"{mean_val:.4f} ± {std_val:.4f}"

    return out


def parse_ckpt_kinds(args) -> List[str]:
    """
    默认只验证 best_composite。
    --ckpt-kind all / default 也会被解析为只跑 best_composite。
    --ckpt-kinds 如传入多个值，将只允许 best_composite。
    """
    if getattr(args, "ckpt_kinds", ""):
        raw = str(args.ckpt_kinds).replace(";", ",").split(",")
        kinds = [x.strip().lower() for x in raw if x.strip()]
    else:
        ckpt_kind = str(getattr(args, "ckpt_kind", "all")).strip().lower()
        if ckpt_kind in {"all", "three", "default"}:
            kinds = list(DEFAULT_CKPT_KINDS)
        else:
            kinds = [ckpt_kind]

    invalid = [k for k in kinds if k not in VALID_CKPT_KINDS]
    if invalid:
        raise ValueError(f"Invalid ckpt kind(s): {invalid}. Valid: {VALID_CKPT_KINDS} or all")

    # 去重且保持顺序
    out: List[str] = []
    for k in kinds:
        if k not in out:
            out.append(k)
    if not out:
        out = list(DEFAULT_CKPT_KINDS)
    return out


def resolve_paths(args):
    project_root = Path(args.project_root).resolve()
    model_dataset = args.model_dataset
    val_dataset = args.val_dataset
    ablation_tag = args.ablation_tag

    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else project_root / "output" / model_dataset / ablation_tag / "checkpoints"
    val_csv = Path(args.val_csv) if args.val_csv else project_root / "data" / val_dataset / f"{val_dataset}_val.csv"
    graph_dir = Path(args.graph_dir) if args.graph_dir else project_root / "feature" / val_dataset / "graph"
    out_base_dir = Path(args.out_dir) if args.out_dir else (
        project_root
        / "output"
        / model_dataset
        / ablation_tag
        / "external_test"
        / f"{model_dataset}_to_{val_dataset}_best_composite"
    )
    return project_root, model_dataset, val_dataset, ablation_tag, ckpt_dir, val_csv, graph_dir, out_base_dir


def resolve_fold_ckpt_path(ckpt_dir: Path, fold_idx: int, ckpt_kind: str) -> Path:
    """只解析 best_composite checkpoint 路径。"""
    ckpt_kind = str(ckpt_kind).strip().lower()
    if ckpt_kind != "best_composite":
        raise ValueError(f"Unsupported ckpt_kind={ckpt_kind}. This version only supports best_composite.")

    candidates = [
        ckpt_dir / f"fold_{fold_idx:02d}_best_composite.pt",
        ckpt_dir / f"fold_{fold_idx}_best_composite.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"best_composite checkpoint not found for fold={fold_idx}. "
        f"Tried: {[str(p) for p in candidates]}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-dataset external validation for interaction_delta v0/v1 ddG models.")
    parser.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    parser.add_argument("--model-dataset", type=str, default=MODEL_DATASET, help="training dataset name, e.g. SKEMPI or ABBIND")
    parser.add_argument("--val-dataset", type=str, default=VAL_DATASET, help="external validation dataset name, e.g. S641/S487/S877")
    parser.add_argument("--ablation-tag", type=str, default=ABLATION_TAG)
    parser.add_argument("--ckpt-dir", type=str, default="", help="override checkpoint dir")
    parser.add_argument("--val-csv", type=str, default="", help="override validation csv path")
    parser.add_argument("--graph-dir", type=str, default="", help="override validation graph cache dir")
    parser.add_argument("--out-dir", type=str, default="", help="override base output dir; ckpt-kind subdirectories are created inside it")
    parser.add_argument("--gpu", type=int, default=None, help="single cuda device index in current visible devices")
    parser.add_argument("--num-folds", type=int, default=int(TRAIN_CONFIG.get("num_folds", 5)))
    parser.add_argument("--no-ddg-filter", action="store_true", help="do not apply -8 < ddG < 8 filter")
    parser.add_argument("--min-ddg", type=float, default=-8.0)
    parser.add_argument("--max-ddg", type=float, default=8.0)
    parser.add_argument("--non-strict-load", action="store_true", help="use strict=False when loading checkpoint")
    parser.add_argument(
        "--ckpt-kind",
        type=str,
        default="all",
        choices=["all", "best_composite"],
        help=(
            "which checkpoint selector to use. Default all validates only best_composite."
        ),
    )
    parser.add_argument(
        "--ckpt-kinds",
        type=str,
        default="",
        help="comma-separated checkpoint kinds. Only best_composite is supported in this version. Overrides --ckpt-kind.",
    )
    return parser


def prepare_validation_context(args, out_base_dir: Path):
    id_col = get_id_col()
    label_col = get_label_col()

    df = pd.read_csv(args.val_csv_path)
    raw_n = len(df)

    if id_col not in df.columns:
        raise ValueError(f"Validation CSV missing ID column '{id_col}'. Existing columns: {df.columns.tolist()}")
    if label_col not in df.columns:
        raise ValueError(f"Validation CSV missing label column '{label_col}'. Existing columns: {df.columns.tolist()}")

    df = df.copy()
    df[id_col] = df[id_col].astype(str).str.strip()
    df["_norm_id"] = df[id_col].map(normalize_split_key)

    if df["_norm_id"].eq("").any():
        bad_n = int(df["_norm_id"].eq("").sum())
        raise ValueError(f"Validation CSV contains {bad_n} empty ID values.")

    if df["_norm_id"].duplicated().any():
        dup = df.loc[df["_norm_id"].duplicated(), "_norm_id"].tolist()
        dup_out = out_base_dir / f"{args.model_dataset}_to_{args.val_dataset}_duplicated_ids.csv"
        out_base_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"duplicated_id": dup}).to_csv(dup_out, index=False, encoding="utf-8-sig")
        raise ValueError(f"Validation CSV has duplicated normalized IDs. First 20: {dup[:20]} | saved: {dup_out}")

    if not args.no_ddg_filter:
        df = filter_ddg(df, min_ddg=args.min_ddg, max_ddg=args.max_ddg)
        logging.info(f"ddG filter applied: keep {args.min_ddg} < ddG < {args.max_ddg} | {raw_n} -> {len(df)}")
    else:
        logging.info(f"ddG filter disabled | samples={len(df)}")

    if len(df) == 0:
        raise RuntimeError("Validation set is empty after filtering.")

    missing = check_missing_graphs(df, args.graph_dir_path)
    if missing:
        miss_out = out_base_dir / f"{args.model_dataset}_to_{args.val_dataset}_missing_graphs.csv"
        out_base_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"sample_id": missing}).to_csv(miss_out, index=False, encoding="utf-8-sig")
        raise FileNotFoundError(
            f"Missing {len(missing)} graph cache files in {args.graph_dir_path}. First 20: {missing[:20]} | saved: {miss_out}"
        )
    logging.info(f"All graph cache files found. Validation samples: {len(df)}")

    records = df.to_dict("records")
    dataset = GraphCacheDatasetByID(
        records=records,
        GRAPH_DIR=args.graph_dir_path,
        preload_in_memory=bool(TRAIN_CONFIG.get("preload_split_in_memory", False)),
    )
    dataloader = build_dataloader(dataset, shuffle=False, seed=int(TRAIN_CONFIG.get("seed", 42)))

    extra_cols = [c for c in ["PDB", "ID", "Mutation", "Partners", label_col] if c in df.columns]
    meta_df = df[["_norm_id"] + extra_cols].copy()
    meta_df = meta_df.rename(columns={"_norm_id": "sample_id"})
    return df, dataloader, meta_df


def validate_one_ckpt_kind(
    args,
    ckpt_kind: str,
    dataloader,
    meta_df: pd.DataFrame,
    criterion: torch.nn.Module,
    device: torch.device,
    out_base_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    model_dataset = args.model_dataset
    val_dataset = args.val_dataset
    ckpt_dir = args.ckpt_dir_path
    out_dir = out_base_dir / ckpt_kind
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: List[Dict[str, Any]] = []
    ensemble_preds: List[np.ndarray] = []
    ensemble_components: Dict[str, List[np.ndarray]] = {}
    y_true_ref = None
    sample_ids_ref = None

    logging.info("=" * 100)
    logging.info(f"Start validating checkpoint kind: {ckpt_kind}")
    logging.info(f"Output dir: {out_dir}")
    logging.info("=" * 100)

    for fold_idx in range(1, int(args.num_folds) + 1):
        ckpt_path = resolve_fold_ckpt_path(ckpt_dir=ckpt_dir, fold_idx=fold_idx, ckpt_kind=ckpt_kind)

        logging.info("-" * 100)
        logging.info(f"Loading Fold {fold_idx} checkpoint: {ckpt_path}")
        model, ckpt = load_model_from_ckpt(ckpt_path, device, strict=not args.non_strict_load)
        label_scaler = label_scaler_from_checkpoint(ckpt)
        logging.info(
            f"Fold {fold_idx} label scaler: enabled={label_scaler.enabled} | "
            f"mean={label_scaler.mean:.6f}, std={label_scaler.std:.6f}"
        )

        metrics, y_true, y_pred, sample_ids, components = evaluate_external(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            device=device,
            label_scaler=label_scaler,
        )

        # 外部验证集可能和训练集 ddG 定义方向相反。
        # 因此每折同时计算三套指标：
        #   original: compute_metrics(y_true, y_pred)
        #   pred_neg: compute_metrics(y_true, -y_pred)
        #   true_neg: compute_metrics(-y_true, y_pred)
        # pred_neg 和 true_neg 的 RMSE/MAE/Pearson 数值等价，
        # 但含义不同：前者表示校正模型输出方向，后者表示校正外部标签方向。
        metrics_pred_neg = compute_metrics(y_true, -y_pred)
        metrics_true_neg = compute_metrics(-y_true, y_pred)

        if y_true_ref is None:
            y_true_ref = y_true.copy()
            sample_ids_ref = list(sample_ids)
        elif list(sample_ids) != list(sample_ids_ref):
            raise RuntimeError("Sample order changed across folds.")

        ensemble_preds.append(y_pred.copy())
        for name, arr in components.items():
            ensemble_components.setdefault(name, []).append(arr.copy())

        pred_df = pd.DataFrame({
            "sample_id": sample_ids,
            "y_true": y_true.reshape(-1),
            "y_pred": y_pred.reshape(-1),
            "y_pred_neg": (-y_pred).reshape(-1),
            "y_true_neg": (-y_true).reshape(-1),
            "y_pred_offset_label_mean": float(label_scaler.mean) if label_scaler.enabled else 0.0,
            "ckpt_kind": ckpt_kind,
        })
        for name, arr in components.items():
            pred_df[name] = arr.reshape(-1)

        pred_df = pred_df.merge(meta_df, on="sample_id", how="left", suffixes=("", "_csv"))

        pred_path = out_dir / f"{model_dataset}_to_{val_dataset}_{ckpt_kind}_fold_{fold_idx:02d}_predictions.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

        row = {
            "model_dataset": model_dataset,
            "val_dataset": val_dataset,
            "ckpt_kind": ckpt_kind,
            "fold": fold_idx,
            "ckpt_path": str(ckpt_path),
            "ckpt_epoch": ckpt.get("epoch", None),
            "ckpt_best_metrics": ckpt.get("metrics", None),
            "label_normalized": bool(label_scaler.enabled),
            "label_mean": float(label_scaler.mean),
            "label_std": float(label_scaler.std),
            "num_samples": len(y_true),
            # 原始方向：compute_metrics(y_true, y_pred)
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "pearson": float(metrics["pearson"]),
            "loss": float(metrics["loss"]),

            # 预测值取反方向：compute_metrics(y_true, -y_pred)
            # 如果外部验证集标签方向和训练集相反，通常报告这一套校正后指标。
            "rmse_pred_neg": float(metrics_pred_neg["rmse"]),
            "mae_pred_neg": float(metrics_pred_neg["mae"]),
            "pearson_pred_neg": float(metrics_pred_neg["pearson"]),

            # 真实标签取反方向：compute_metrics(-y_true, y_pred)
            # 与 pred_neg 数值等价，用于诊断“标签方向相反”。
            "rmse_true_neg": float(metrics_true_neg["rmse"]),
            "mae_true_neg": float(metrics_true_neg["mae"]),
            "pearson_true_neg": float(metrics_true_neg["pearson"]),

            "pred_path": str(pred_path),
        }
        row.update(component_stats(y_true, components))
        fold_rows.append(row)

        comp_msg = ""
        if components:
            comp_msg = " | components=" + ",".join(sorted(components.keys()))
        logging.info(
            f"[{model_dataset} -> {val_dataset} | Fold {fold_idx} | {ckpt_kind}] "
            f"ckpt_epoch={ckpt.get('epoch', None)} | "
            f"original: RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} Pearson={metrics['pearson']:.4f} | "
            f"pred_neg: RMSE={metrics_pred_neg['rmse']:.4f} MAE={metrics_pred_neg['mae']:.4f} "
            f"Pearson={metrics_pred_neg['pearson']:.4f} | Loss={metrics['loss']:.4f}{comp_msg}"
        )
        logging.info(f"Saved fold predictions to: {pred_path}")

    summary_df = pd.DataFrame(fold_rows)
    summary_path = out_dir / f"{model_dataset}_to_{val_dataset}_{ckpt_kind}_fold_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    ensemble_pred = np.mean(np.stack(ensemble_preds, axis=0), axis=0)
    ensemble_metrics = compute_metrics(y_true_ref, ensemble_pred)
    ensemble_metrics_pred_neg = compute_metrics(y_true_ref, -ensemble_pred)
    ensemble_metrics_true_neg = compute_metrics(-y_true_ref, ensemble_pred)
    ensemble_metrics_both_neg = compute_metrics(-y_true_ref, -ensemble_pred)

    ensemble_df = pd.DataFrame({
        "sample_id": sample_ids_ref,
        "y_true": y_true_ref.reshape(-1),
        "y_true_neg": (-y_true_ref).reshape(-1),
        "y_pred_ensemble_mean": ensemble_pred.reshape(-1),
        "y_pred_ensemble_mean_neg": (-ensemble_pred).reshape(-1),
        "ckpt_kind": ckpt_kind,
    })
    for i, pred in enumerate(ensemble_preds, start=1):
        ensemble_df[f"y_pred_fold_{i:02d}"] = pred.reshape(-1)

    ensemble_component_means: Dict[str, np.ndarray] = {}
    for name, arrs in ensemble_components.items():
        if len(arrs) == len(ensemble_preds):
            stack = np.stack(arrs, axis=0)
            mean_arr = np.mean(stack, axis=0)
            ensemble_df[f"{name}_ensemble_mean"] = mean_arr.reshape(-1)
            ensemble_component_means[name] = mean_arr
            for i, arr in enumerate(arrs, start=1):
                ensemble_df[f"{name}_fold_{i:02d}"] = arr.reshape(-1)

    ensemble_df = ensemble_df.merge(meta_df, on="sample_id", how="left", suffixes=("", "_csv"))

    ensemble_pred_path = out_dir / f"{model_dataset}_to_{val_dataset}_{ckpt_kind}_ensemble_predictions.csv"
    ensemble_df.to_csv(ensemble_pred_path, index=False, encoding="utf-8-sig")

    ensemble_row: Dict[str, Any] = {
        "model_dataset": model_dataset,
        "val_dataset": val_dataset,
        "ckpt_kind": ckpt_kind,
        "num_samples": len(y_true_ref),

        # 原始方向
        "rmse": float(ensemble_metrics["rmse"]),
        "mae": float(ensemble_metrics["mae"]),
        "pearson": float(ensemble_metrics["pearson"]),

        # 只把预测取反：常用于检查模型输出方向是否反了
        "rmse_pred_neg": float(ensemble_metrics_pred_neg["rmse"]),
        "mae_pred_neg": float(ensemble_metrics_pred_neg["mae"]),
        "pearson_pred_neg": float(ensemble_metrics_pred_neg["pearson"]),

        # 只把真实标签取反：常用于检查外部数据集标签方向是否和训练集相反
        "rmse_true_neg": float(ensemble_metrics_true_neg["rmse"]),
        "mae_true_neg": float(ensemble_metrics_true_neg["mae"]),
        "pearson_true_neg": float(ensemble_metrics_true_neg["pearson"]),

        # 双方都取反：Pearson 和原始相同，主要用于 sanity check
        "rmse_both_neg": float(ensemble_metrics_both_neg["rmse"]),
        "mae_both_neg": float(ensemble_metrics_both_neg["mae"]),
        "pearson_both_neg": float(ensemble_metrics_both_neg["pearson"]),

        "prediction_path": str(ensemble_pred_path),
        "fold_summary_path": str(summary_path),
        "output_dir": str(out_dir),
    }
    ensemble_row.update(component_stats(y_true_ref, ensemble_component_means))

    ensemble_summary_path = out_dir / f"{model_dataset}_to_{val_dataset}_{ckpt_kind}_ensemble_summary.csv"
    pd.DataFrame([ensemble_row]).to_csv(ensemble_summary_path, index=False, encoding="utf-8-sig")
    ensemble_row["ensemble_summary_path"] = str(ensemble_summary_path)

    print("=" * 100)
    print("Cross-dataset external validation finished for one checkpoint kind")
    print(f"MODEL_DATASET: {model_dataset}")
    print(f"VAL_DATASET:   {val_dataset}")
    print(f"CKPT_KIND:     {ckpt_kind}")
    print(f"Samples:       {len(y_true_ref)}")
    print("-" * 100)
    print(summary_df[[
        "fold", "ckpt_epoch",
        "rmse", "mae", "pearson",
        "rmse_pred_neg", "mae_pred_neg", "pearson_pred_neg",
        "loss",
    ]].to_string(index=False))
    print("-" * 100)
    print(
        f"Ensemble mean | RMSE={ensemble_metrics['rmse']:.4f} "
        f"MAE={ensemble_metrics['mae']:.4f} Pearson={ensemble_metrics['pearson']:.4f}"
    )
    print(
        f"Ensemble mean | original: RMSE={ensemble_metrics['rmse']:.4f} "
        f"MAE={ensemble_metrics['mae']:.4f} Pearson={ensemble_metrics['pearson']:.4f}"
    )
    print(
        f"Ensemble mean | pred neg: RMSE={ensemble_metrics_pred_neg['rmse']:.4f} "
        f"MAE={ensemble_metrics_pred_neg['mae']:.4f} Pearson={ensemble_metrics_pred_neg['pearson']:.4f}"
    )
    print(
        f"Ensemble mean | true neg: RMSE={ensemble_metrics_true_neg['rmse']:.4f} "
        f"MAE={ensemble_metrics_true_neg['mae']:.4f} Pearson={ensemble_metrics_true_neg['pearson']:.4f}"
    )
    if ensemble_component_means:
        print("-" * 100)
        print("Ensemble component columns saved:")
        for name in sorted(ensemble_component_means.keys()):
            arr = ensemble_component_means[name]
            print(f"  {name}: mean={float(np.mean(arr)):.4f}, std={float(np.std(arr)):.4f}")
    print("-" * 100)
    print(f"Saved fold summary:         {summary_path}")
    print(f"Saved ensemble predictions: {ensemble_pred_path}")
    print(f"Saved ensemble summary:     {ensemble_summary_path}")
    print("=" * 100)

    return summary_df, ensemble_row


def main():
    args = build_parser().parse_args()
    ckpt_kinds = parse_ckpt_kinds(args)
    (
        project_root,
        model_dataset,
        val_dataset,
        ablation_tag,
        ckpt_dir,
        val_csv,
        graph_dir,
        out_base_dir,
    ) = resolve_paths(args)

    # 把解析后的路径回写到 args，后续函数统一读取。
    args.project_root_path = project_root
    args.model_dataset = model_dataset
    args.val_dataset = val_dataset
    args.ablation_tag = ablation_tag
    args.ckpt_dir_path = ckpt_dir
    args.val_csv_path = val_csv
    args.graph_dir_path = graph_dir

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info("=" * 100)
    logging.info("Start cross-dataset external validation for interaction_delta v0/v1 model")
    logging.info(f"PROJECT_ROOT:  {project_root}")
    logging.info(f"MODEL_DATASET: {model_dataset}")
    logging.info(f"VAL_DATASET:   {val_dataset}")
    logging.info(f"ABLATION_TAG:  {ablation_tag}")
    logging.info(f"CKPT_KINDS:    {ckpt_kinds}")
    logging.info(f"Checkpoint dir: {ckpt_dir}")
    logging.info(f"Validation CSV: {val_csv}")
    logging.info(f"Graph dir:      {graph_dir}")
    logging.info(f"Base output dir: {out_base_dir}")
    logging.info("=" * 100)

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")
    if not val_csv.exists():
        raise FileNotFoundError(f"Validation CSV not found: {val_csv}")
    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph cache dir not found: {graph_dir}")
    out_base_dir.mkdir(parents=True, exist_ok=True)

    _df, dataloader, meta_df = prepare_validation_context(args, out_base_dir=out_base_dir)

    device = choose_device(args.gpu)
    logging.info(f"Using device: {device}")
    criterion = build_criterion()

    all_fold_summaries: List[pd.DataFrame] = []
    all_ensemble_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for ckpt_kind in ckpt_kinds:
        try:
            fold_summary, ensemble_row = validate_one_ckpt_kind(
                args=args,
                ckpt_kind=ckpt_kind,
                dataloader=dataloader,
                meta_df=meta_df,
                criterion=criterion,
                device=device,
                out_base_dir=out_base_dir,
            )
            all_fold_summaries.append(fold_summary)
            all_ensemble_rows.append(ensemble_row)
        except Exception as e:
            logging.exception(f"Validation failed for ckpt_kind={ckpt_kind}: {e}")
            errors.append({
                "model_dataset": model_dataset,
                "val_dataset": val_dataset,
                "ckpt_kind": ckpt_kind,
                "error": str(e),
            })
            # 继续跑其它 checkpoint kind，最后统一报错提示。
            continue

    combined_fold_path = out_base_dir / f"{model_dataset}_to_{val_dataset}_best_composite_fold_summary.csv"
    combined_ensemble_path = out_base_dir / f"{model_dataset}_to_{val_dataset}_best_composite_ensemble_summary.csv"
    error_path = out_base_dir / f"{model_dataset}_to_{val_dataset}_best_composite_errors.csv"

    if all_fold_summaries:
        pd.concat(all_fold_summaries, axis=0, ignore_index=True).to_csv(
            combined_fold_path,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        pd.DataFrame().to_csv(combined_fold_path, index=False, encoding="utf-8-sig")

    if all_ensemble_rows:
        ensemble_summary_df = pd.DataFrame(all_ensemble_rows)
        ensemble_summary_df.to_csv(combined_ensemble_path, index=False, encoding="utf-8-sig")
    else:
        ensemble_summary_df = pd.DataFrame()
        ensemble_summary_df.to_csv(combined_ensemble_path, index=False, encoding="utf-8-sig")

    if errors:
        pd.DataFrame(errors).to_csv(error_path, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("Cross-dataset external validation finished for best_composite")
    print(f"MODEL_DATASET: {model_dataset}")
    print(f"VAL_DATASET:   {val_dataset}")
    print(f"CKPT_KINDS:    {ckpt_kinds}")
    print("-" * 100)
    if not ensemble_summary_df.empty:
        cols = [
            "ckpt_kind", "num_samples",
            "rmse", "mae", "pearson",
            "rmse_pred_neg", "mae_pred_neg", "pearson_pred_neg",
            "prediction_path",
        ]
        print(ensemble_summary_df[cols].to_string(index=False))
    else:
        print("No successful checkpoint-kind validation results.")
    print("-" * 100)
    print(f"Saved combined fold summary:     {combined_fold_path}")
    print(f"Saved combined ensemble summary: {combined_ensemble_path}")
    if errors:
        print(f"Some checkpoint kinds failed. Saved errors: {error_path}")
    print("=" * 100)

    if errors and not all_ensemble_rows:
        raise RuntimeError(f"All checkpoint-kind validations failed. See: {error_path}")


if __name__ == "__main__":
    main()
