#!/usr/bin/env python3
"""
Train an MLP to predict sensor deltas instead of absolute sensor values.

Target:
    delta_sensor_t = sensor_t - sensor_{t-1}

Reconstruction:
    pred_sensor_t = sensor_{t-1} + pred_delta_sensor_t

Supported input modes:
1. joints only
2. joints + velocity
3. joints + velocity + previous sensor values

This script keeps the existing project structure and split style close to
train_mlp_v_prev_sensor.py while changing the learning target to sensor delta.
"""

import os
import sys
import random
from pathlib import Path
from datetime import datetime
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from self_detection_raw.data.loader import load_file, extract_features
from self_detection_raw.data.loader_v import extract_features_v
from self_detection_raw.models.mlp_b_v import ModelBV
from self_detection_raw.train.train import train_epoch
from self_detection_raw.utils.io import ensure_dir, save_json
from self_detection_raw.utils.metrics import compute_channel_metrics

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DATA_DIR = "/home/song/rb10_Proximity/src/self_detection_raw/dataset"

TRAIN_FILES = [
    "1dataset_50_25_1.txt",
    "2dataset_50_25_1.txt",
    "3dataset_50_25_1.txt",

]

VAL_FILES = [
    "4dataset_50_25_1.txt",

]

script_root = Path(__file__).parent.absolute()
OUT_DIR = str(script_root / "model")
NAME_PREFIX = "mlp_vel_delta"

USE_VEL = True
USE_PREV_SENSOR = True
PREV_SENSOR_CHANNELS = ""
VEL_WINDOW = 10
STD_FLOOR = 1e-2
EPOCHS = 300
BATCH = 256
LR = 1e-3
WD = 1e-4
HIDDEN = 128
HEAD_HIDDEN = 64
DROPOUT = 0.1
SEED = 42
NUM_WORKERS = 4

CHANNEL_NAMES = [f"raw{i}" for i in range(1, 9)]
RAW_ERROR_THRESHOLD = 2000.0
CHANNEL_LOSS_WEIGHTS = [1.0] * len(CHANNEL_NAMES)
BEST_METRIC_NAME = "within_2000"
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.0
EARLY_STOP_WARMUP_EPOCHS = 10
WITHIN_THRESHOLDS = (1000.0, 2000.0, 5000.0, 10000.0)
# ---------------------------------------------------------------------------


def full_paths(names, base_dir):
    paths = []
    for fn in names:
        p = Path(fn)
        if not p.is_absolute():
            paths.append(str(Path(base_dir) / fn))
        else:
            paths.append(str(p))
    return paths


def build_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


class WeightedChannelwiseHuberLoss(nn.Module):
    """
    Weighted Huber loss in normalized delta space.

    The raw engineering threshold is mapped to normalized delta space as:
        delta_norm[c] = raw_threshold / delta_std[c]
    """

    def __init__(self, delta_norm, channel_weights):
        super().__init__()
        delta_tensor = torch.as_tensor(delta_norm, dtype=torch.float32)
        weight_tensor = torch.as_tensor(channel_weights, dtype=torch.float32)
        if delta_tensor.shape != weight_tensor.shape:
            raise ValueError(
                f"delta_norm shape {tuple(delta_tensor.shape)} must match "
                f"channel_weights shape {tuple(weight_tensor.shape)}"
            )
        normalized_weights = weight_tensor / torch.clamp(weight_tensor.mean(), min=1e-8)
        self.register_buffer("delta", torch.clamp(delta_tensor, min=1e-8))
        self.register_buffer("weights", normalized_weights)

    def forward(self, pred, target):
        delta = self.delta.to(device=pred.device, dtype=pred.dtype)
        weights = self.weights.to(device=pred.device, dtype=pred.dtype)
        error = pred - target
        abs_error = error.abs()
        quadratic = torch.minimum(abs_error, delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic.pow(2) + delta * linear
        weighted_loss = loss * weights.unsqueeze(0)
        return weighted_loss.mean()


class DeltaPredictionDataset(Dataset):
    """
    Dataset for delta prediction.

    Inputs are normalized jointly, including previous sensor inputs when enabled.
    Targets are normalized delta values:
        delta_t = sensor_t - sensor_{t-1}
    """

    def __init__(
        self,
        X,
        prev_sensor_raw,
        sensor_curr_raw,
        delta_raw,
        X_mean=None,
        X_std=None,
        Y_mean=None,
        Y_std=None,
        std_floor=1e-2,
    ):
        self.X = X.astype(np.float32)
        self.prev_sensor_raw = prev_sensor_raw.astype(np.float32)
        self.sensor_curr_raw = sensor_curr_raw.astype(np.float32)
        self.delta_raw = delta_raw.astype(np.float32)
        self.std_floor = std_floor

        if X_mean is None:
            X_mean = np.mean(self.X, axis=0)
            X_std = np.std(self.X, axis=0)
        if Y_mean is None:
            Y_mean = np.mean(self.delta_raw, axis=0)
            Y_std = np.std(self.delta_raw, axis=0)

        self.X_mean = X_mean.astype(np.float32)
        self.X_std = np.maximum(X_std, std_floor).astype(np.float32)
        self.Y_mean = Y_mean.astype(np.float32)
        self.Y_std = np.maximum(Y_std, std_floor).astype(np.float32)

        self.X_norm = (self.X - self.X_mean) / self.X_std
        self.Y_norm = (self.delta_raw - self.Y_mean) / self.Y_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_norm[idx]),
            torch.from_numpy(self.Y_norm[idx]),
        )

    def get_norm_params(self):
        return {
            "X_mean": self.X_mean,
            "X_std": self.X_std,
            "Y_mean": self.Y_mean,
            "Y_std": self.Y_std,
            "target_type": "delta_sensor",
        }


def parse_prev_sensor_channels(value):
    if value is None or value == "":
        return list(range(len(CHANNEL_NAMES)))

    indices = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if token.startswith("raw"):
            idx = int(token[3:]) - 1
        else:
            idx = int(token)
            if idx >= 1:
                idx -= 1
        if idx < 0 or idx >= len(CHANNEL_NAMES):
            raise ValueError(f"Invalid previous sensor channel specifier: {token}")
        indices.append(idx)

    if not indices:
        raise ValueError("prev_sensor_channels resolved to an empty list")
    return sorted(set(indices))


def extract_joint_and_sensor(filepath, use_vel, vel_window):
    data = load_file(filepath)
    if use_vel:
        return extract_features_v(data, use_vel=True, vel_window=vel_window)
    return extract_features(data, use_vel=False)


def align_delta_samples(X_joint, sensor_abs, use_prev_sensor, prev_sensor_indices):
    if len(X_joint) < 2:
        raise ValueError("Need at least 2 samples per file for delta prediction")

    X_curr = X_joint[1:]
    prev_sensor_full = sensor_abs[:-1]
    sensor_curr = sensor_abs[1:]
    delta_sensor = sensor_curr - prev_sensor_full

    if use_prev_sensor:
        prev_sensor_selected = prev_sensor_full[:, prev_sensor_indices]
        X_input = np.concatenate([X_curr, prev_sensor_selected], axis=1)
    else:
        X_input = X_curr

    return X_input, prev_sensor_full, sensor_curr, delta_sensor


def load_multiple_files_delta(filepaths, use_vel, vel_window, use_prev_sensor, prev_sensor_indices):
    X_list = []
    prev_list = []
    curr_list = []
    delta_list = []

    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f"[WARN] File not found: {filepath}")
            continue

        X_joint, sensor_abs = extract_joint_and_sensor(filepath, use_vel=use_vel, vel_window=vel_window)
        X_input, prev_sensor_full, sensor_curr, delta_sensor = align_delta_samples(
            X_joint,
            sensor_abs,
            use_prev_sensor=use_prev_sensor,
            prev_sensor_indices=prev_sensor_indices,
        )

        X_list.append(X_input)
        prev_list.append(prev_sensor_full)
        curr_list.append(sensor_curr)
        delta_list.append(delta_sensor)

    if not X_list:
        return np.array([]), np.array([]), np.array([]), np.array([])

    return (
        np.concatenate(X_list, axis=0),
        np.concatenate(prev_list, axis=0),
        np.concatenate(curr_list, axis=0),
        np.concatenate(delta_list, axis=0),
    )


def describe_input_config(use_vel, use_prev_sensor, prev_sensor_indices):
    if not use_vel and not use_prev_sensor:
        return "joints only"
    if use_vel and not use_prev_sensor:
        return "joints + velocity"
    if not use_vel and use_prev_sensor:
        selected = [CHANNEL_NAMES[idx] for idx in prev_sensor_indices]
        return "joints + previous sensors (" + ", ".join(selected) + ")"
    selected = [CHANNEL_NAMES[idx] for idx in prev_sensor_indices]
    if len(selected) == len(CHANNEL_NAMES):
        return "joints + velocity + previous sensors"
    return "joints + velocity + previous sensors (" + ", ".join(selected) + ")"


def compute_practical_metrics(pred_sensor_abs, target_sensor_abs):
    residuals = target_sensor_abs - pred_sensor_abs
    abs_errors = np.abs(residuals)
    sq_errors = residuals ** 2

    per_channel_rmse = np.sqrt(np.mean(sq_errors, axis=0))
    per_channel_mae = np.mean(abs_errors, axis=0)
    ranked_indices = np.argsort(per_channel_rmse)
    channel_metrics = compute_channel_metrics(target_sensor_abs, residuals)

    metrics = {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(sq_errors))),
        "avg_std": float(np.mean(channel_metrics["residual_std"])),
        "per_channel_rmse": per_channel_rmse,
        "per_channel_mae": per_channel_mae,
        "best_channel_name": CHANNEL_NAMES[ranked_indices[0]],
        "best_channel_rmse": float(per_channel_rmse[ranked_indices[0]]),
        "worst_channel_name": CHANNEL_NAMES[ranked_indices[-1]],
        "worst_channel_rmse": float(per_channel_rmse[ranked_indices[-1]]),
        "channel_metrics": channel_metrics,
    }

    for threshold in WITHIN_THRESHOLDS:
        metrics[f"within_{int(threshold)}"] = float(np.mean(abs_errors <= threshold))

    return metrics


@torch.no_grad()
def evaluate_practical(model, dataset, device, batch_size):
    model.eval()

    preds_norm = []
    for start in range(0, len(dataset), batch_size):
        batch = torch.from_numpy(dataset.X_norm[start:start + batch_size]).to(device)
        pred_norm = model(batch).cpu().numpy()
        preds_norm.append(pred_norm)

    pred_delta_norm = np.concatenate(preds_norm, axis=0)
    pred_delta = pred_delta_norm * dataset.Y_std + dataset.Y_mean
    pred_sensor_abs = dataset.prev_sensor_raw + pred_delta
    target_sensor_abs = dataset.sensor_curr_raw
    return compute_practical_metrics(pred_sensor_abs, target_sensor_abs)


def is_better_metric(candidate, current_best, metric_name):
    if current_best is None:
        return True
    if metric_name == "within_2000":
        return candidate > current_best
    return candidate < current_best


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(value) for value in obj]
    return obj


def save_final_metrics(run_dir, metrics, args, delta_norm):
    payload = {
        "best_metric_name": args.best_metric_name,
        "raw_error_threshold": args.raw_threshold,
        "normalized_threshold_per_channel": delta_norm,
        "final_validation_metrics": metrics,
    }
    save_json(to_serializable(payload), str(Path(run_dir) / "final_metrics.json"))


def main():
    parser = argparse.ArgumentParser(description="Train MLP with delta sensor targets")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--train-files", default=",".join(TRAIN_FILES))
    parser.add_argument("--val-files", default=",".join(VAL_FILES))
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--name-prefix", default=NAME_PREFIX)

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--wd", type=float, default=WD)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--head-hidden", type=int, default=HEAD_HIDDEN)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)

    parser.add_argument("--use-vel", type=int, default=int(USE_VEL), help="Use velocity features")
    parser.add_argument(
        "--use-prev-sensor",
        type=int,
        default=int(USE_PREV_SENSOR),
        help="Append previous sensor values to current joint features",
    )
    parser.add_argument(
        "--prev-sensor-channels",
        default=PREV_SENSOR_CHANNELS,
        help="Comma-separated previous sensor channels, e.g. raw1,raw3 or 1,3. Empty = all",
    )
    parser.add_argument("--vel-window", type=int, default=VEL_WINDOW, help="Velocity smoothing window size")
    parser.add_argument(
        "--raw-threshold",
        type=float,
        default=RAW_ERROR_THRESHOLD,
        help="Raw-scale Huber threshold in original sensor units",
    )
    parser.add_argument(
        "--best-metric-name",
        default=BEST_METRIC_NAME,
        choices=["within_2000", "rmse", "avg_std"],
        help="Validation metric used to select best checkpoint",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=EARLY_STOP_PATIENCE,
        help="Stop training if validation RMSE does not improve for this many epochs",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=EARLY_STOP_MIN_DELTA,
        help="Minimum RMSE improvement required to reset early stopping",
    )
    parser.add_argument(
        "--early-stop-warmup",
        type=int,
        default=EARLY_STOP_WARMUP_EPOCHS,
        help="Do not apply early stopping before this epoch",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    train_files = args.train_files.split(",") if args.train_files else []
    val_files = args.val_files.split(",") if args.val_files else []
    use_vel = bool(args.use_vel)
    use_prev_sensor = bool(args.use_prev_sensor)
    vel_window = args.vel_window
    prev_sensor_indices = parse_prev_sensor_channels(args.prev_sensor_channels)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_workers = args.num_workers
    if device.type != "cuda" and num_workers == NUM_WORKERS:
        num_workers = 0

    train_paths = full_paths(train_files, data_dir)
    val_paths = full_paths(val_files, data_dir)

    print(f"\n데이터 로딩 중... (Velocity Smoothing Window: {vel_window})")
    print(f"Input mode: {describe_input_config(use_vel, use_prev_sensor, prev_sensor_indices)}")
    if use_prev_sensor:
        selected_names = [CHANNEL_NAMES[idx] for idx in prev_sensor_indices]
        print("Previous sensor channels: " + ", ".join(selected_names))
    print("Alignment note: first sample of each file is dropped because delta_t uses sensor_{t-1}.")

    X_train, prev_train, sensor_train, delta_train = load_multiple_files_delta(
        train_paths,
        use_vel=use_vel,
        vel_window=vel_window,
        use_prev_sensor=use_prev_sensor,
        prev_sensor_indices=prev_sensor_indices,
    )
    if val_paths:
        X_val, prev_val, sensor_val, delta_val = load_multiple_files_delta(
            val_paths,
            use_vel=use_vel,
            vel_window=vel_window,
            use_prev_sensor=use_prev_sensor,
            prev_sensor_indices=prev_sensor_indices,
        )
    else:
        X_val = np.array([]).reshape(0, X_train.shape[1])
        prev_val = np.array([]).reshape(0, len(CHANNEL_NAMES))
        sensor_val = np.array([]).reshape(0, len(CHANNEL_NAMES))
        delta_val = np.array([]).reshape(0, len(CHANNEL_NAMES))

    if X_train.size == 0 or delta_train.size == 0:
        raise RuntimeError("No training samples available after delta alignment. Check file paths and flags.")

    print(f"Train Input Shape: {X_train.shape}")
    print(f"Train Delta Target Shape: {delta_train.shape}")

    train_ds = DeltaPredictionDataset(
        X_train,
        prev_train,
        sensor_train,
        delta_train,
        std_floor=STD_FLOOR,
    )
    val_ds = DeltaPredictionDataset(
        X_val,
        prev_val,
        sensor_val,
        delta_val,
        X_mean=train_ds.X_mean,
        X_std=train_ds.X_std,
        Y_mean=train_ds.Y_mean,
        Y_std=train_ds.Y_std,
        std_floor=STD_FLOOR,
    )

    pin_memory = device.type == "cuda"
    train_loader = build_dataloader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = ModelBV(
        in_dim=X_train.shape[1],
        trunk_hidden=args.hidden,
        head_hidden=args.head_hidden,
        out_dim=delta_train.shape[1],
        dropout=args.dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    delta_norm = (args.raw_threshold / np.maximum(train_ds.Y_std, train_ds.std_floor)).astype(np.float32)
    channel_loss_weights = np.asarray(CHANNEL_LOSS_WEIGHTS, dtype=np.float32)
    if channel_loss_weights.shape[0] != delta_train.shape[1]:
        raise ValueError(
            f"CHANNEL_LOSS_WEIGHTS length ({channel_loss_weights.shape[0]}) must match "
            f"number of output channels ({delta_train.shape[1]})"
        )
    criterion = WeightedChannelwiseHuberLoss(delta_norm, channel_loss_weights).to(device)

    try:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=25, verbose=True)
    except Exception:
        scheduler = None

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    run_dir = ensure_dir(os.path.join(args.out_dir, f"{args.name_prefix}_{timestamp}"))

    best_metric_value = None
    best_epoch = None
    best_rmse_for_early_stop = None
    no_improve_epochs = 0
    history = {
        "train_loss": [],
        "val_avg_std": [],
        "val_rmse": [],
        "val_within_2000": [],
    }

    run_config = vars(args).copy()
    run_config["use_vel"] = use_vel
    run_config["use_prev_sensor"] = use_prev_sensor
    run_config["prev_sensor_indices"] = prev_sensor_indices
    run_config["prev_sensor_channel_names"] = [CHANNEL_NAMES[idx] for idx in prev_sensor_indices]
    run_config["input_mode"] = describe_input_config(use_vel, use_prev_sensor, prev_sensor_indices)
    run_config["target_type"] = "delta_sensor"
    run_config["normalized_threshold_per_channel"] = delta_norm.tolist()
    run_config["channel_loss_weights"] = channel_loss_weights.tolist()
    save_json(to_serializable(run_config), str(Path(run_dir) / "config.json"))

    print("\n학습 시작...")
    print(f"Raw threshold: {args.raw_threshold:.1f}")
    print("Normalized threshold per channel: " + ", ".join(f"{value:.4f}" for value in delta_norm))
    print("Channel loss weights: " + ", ".join(f"{value:.3f}" for value in channel_loss_weights))

    for epoch in range(1, args.epochs + 1):
        try:
            loss = train_epoch(model, train_loader, criterion, optimizer, device)
        except PermissionError as exc:
            if num_workers > 0:
                print(
                    f"DataLoader worker startup failed ({exc}). "
                    "Retrying with --num-workers 0."
                )
                num_workers = 0
                train_loader = build_dataloader(
                    train_ds,
                    batch_size=args.batch,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                )
                loss = train_epoch(model, train_loader, criterion, optimizer, device)
            else:
                raise

        metrics = evaluate_practical(model, val_ds, device, args.batch) if len(val_paths) > 0 else {}

        if scheduler and "rmse" in metrics:
            scheduler.step(metrics["rmse"])

        history["train_loss"].append(loss)
        if metrics:
            history["val_avg_std"].append(metrics["avg_std"])
            history["val_rmse"].append(metrics["rmse"])
            history["val_within_2000"].append(metrics["within_2000"])

            selection_value = metrics[args.best_metric_name]
            if is_better_metric(selection_value, best_metric_value, args.best_metric_name):
                best_metric_value = selection_value
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": to_serializable(metrics),
                        "normalization": train_ds.get_norm_params(),
                        "args": run_config,
                    },
                    Path(run_dir) / "model.pt",
                )
                print(
                    f"Saved best model to {Path(run_dir) / 'model.pt'} "
                    f"({args.best_metric_name}={selection_value:.4f})"
                )

            rmse_value = metrics["rmse"]
            if (
                best_rmse_for_early_stop is None
                or rmse_value < (best_rmse_for_early_stop - args.early_stop_min_delta)
            ):
                best_rmse_for_early_stop = rmse_value
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

        if epoch % 10 == 0 or epoch == 1:
            msg = f"Epoch {epoch}/{args.epochs} | Train loss: {loss:.6f}"
            if metrics:
                msg += (
                    f" | Val RMSE: {metrics['rmse']:.2f}"
                    f" | Val MAE: {metrics['mae']:.2f}"
                    f" | W2000: {metrics['within_2000']:.3f}"
                    f" | AvgSTD: {metrics['avg_std']:.2f}"
                    f" | BestCh: {metrics['best_channel_name']}={metrics['best_channel_rmse']:.2f}"
                    f" | WorstCh: {metrics['worst_channel_name']}={metrics['worst_channel_rmse']:.2f}"
                )
            print(msg)

        if (
            metrics
            and epoch >= args.early_stop_warmup
            and no_improve_epochs >= args.early_stop_patience
        ):
            print(
                f"Early stopping at epoch {epoch}: "
                f"validation RMSE did not improve for {args.early_stop_patience} epochs "
                f"(best_rmse={best_rmse_for_early_stop:.2f})"
            )
            break

    final_metrics = {}
    if len(val_paths) > 0 and best_epoch is not None:
        checkpoint = torch.load(Path(run_dir) / "model.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_metrics = evaluate_practical(model, val_ds, device, args.batch)

    if final_metrics:
        save_final_metrics(run_dir, final_metrics, args, delta_norm.tolist())

    print(
        f"Finished training. Best {args.best_metric_name}: "
        f"{best_metric_value if best_metric_value is not None else 'n/a'}"
    )
    if best_rmse_for_early_stop is not None:
        print(f"Best RMSE for early stopping: {best_rmse_for_early_stop:.2f}")
    if best_epoch is not None:
        print(f"Best epoch: {best_epoch}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
