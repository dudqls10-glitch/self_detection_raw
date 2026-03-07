#!/usr/bin/env python3
"""
Training script for Method B:
MLP(main) + TCN(residual) with Loss3 = fit + lambda_res * |residual|_1.
"""

import argparse
import csv
import os
import random
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, leave=True):
        if desc:
            print(desc)
        return iterable

from self_detection_raw.data.loader import (
    load_file,
    extract_features,
    split_files_train_val,
)
from self_detection_raw.data.stats import compute_stats_from_array, save_norm_params
from self_detection_raw.models.mlp_tcn_residual import MLP_TCN_ResidualModel
from self_detection_raw.utils.io import ensure_dir, find_files_by_pattern, save_json
from self_detection_raw.utils.metrics import compute_channel_metrics, format_metrics_report


class SelfDetectionSeqDataset(Dataset):
    """Sequence dataset without crossing file boundaries."""

    def __init__(
        self,
        X_files: List[np.ndarray],
        Y_files: List[np.ndarray],
        seq_len: int,
        stride: int = 1,
        X_mean: np.ndarray = None,
        X_std: np.ndarray = None,
        Y_mean: np.ndarray = None,
        Y_std: np.ndarray = None,
        std_floor: float = 1e-2,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.std_floor = std_floor

        self.X_files = [x.astype(np.float32) for x in X_files]
        self.Y_files = [y.astype(np.float32) for y in Y_files]

        if not self.X_files:
            raise ValueError("No input files for dataset")

        X_all = np.concatenate(self.X_files, axis=0)
        Y_all = np.concatenate(self.Y_files, axis=0)

        if X_mean is None:
            X_mean, X_std = compute_stats_from_array(X_all, std_floor)
        if Y_mean is None:
            Y_mean, Y_std = compute_stats_from_array(Y_all, std_floor)

        self.X_mean = X_mean.astype(np.float32)
        self.X_std = X_std.astype(np.float32)
        self.Y_mean = Y_mean.astype(np.float32)
        self.Y_std = Y_std.astype(np.float32)

        self.Xn_files = [(x - self.X_mean) / self.X_std for x in self.X_files]
        self.Yn_files = [(y - self.Y_mean) / self.Y_std for y in self.Y_files]

        # index map: (file_idx, end_idx)
        self.indices: List[Tuple[int, int]] = []
        for f_idx, x in enumerate(self.Xn_files):
            n = len(x)
            if n < seq_len:
                continue
            for end_idx in range(seq_len - 1, n, stride):
                self.indices.append((f_idx, end_idx))

        if not self.indices:
            raise ValueError(
                f"No valid sequence windows. Need file length >= seq_len ({seq_len})."
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        f_idx, end_idx = self.indices[idx]
        start_idx = end_idx - self.seq_len + 1

        x_seq = self.Xn_files[f_idx][start_idx:end_idx + 1]  # (T, 12)
        y = self.Yn_files[f_idx][end_idx]  # (8,)

        return torch.from_numpy(x_seq), torch.from_numpy(y)

    def get_norm_params(self):
        return {
            "X_mean": self.X_mean,
            "X_std": self.X_std,
            "Y_mean": self.Y_mean,
            "Y_std": self.Y_std,
        }


def find_default_data_dir():
    possible_dirs = [
        "/home/son_rb/rb_ws/src/robotory_rb10_ros2/scripts",
        "/home/son_rb/rb_ws/src/self_detection",
        os.path.join(os.path.dirname(__file__), "../../..", "robotory_rb10_ros2", "scripts"),
        os.path.join(os.path.dirname(__file__), "../../..", "self_detection"),
    ]
    for data_dir in possible_dirs:
        if os.path.exists(data_dir):
            files = find_files_by_pattern(data_dir, "robot_data_*.txt")
            if files:
                return data_dir
    return None


def load_xy_files(filepaths: List[str], use_vel: bool = False):
    x_files, y_files = [], []
    for fp in filepaths:
        data = load_file(fp)
        x, y = extract_features(data, use_vel=use_vel)
        x_files.append(x)
        y_files.append(y)
    return x_files, y_files


def configure_stage(model: MLP_TCN_ResidualModel, stage: str):
    """Freeze/unfreeze params per stage."""
    if stage == "main_only":
        for p in model.main.parameters():
            p.requires_grad = True
        for p in model.residual.parameters():
            p.requires_grad = False
    elif stage == "res_only":
        for p in model.main.parameters():
            p.requires_grad = False
        for p in model.residual.parameters():
            p.requires_grad = True
    elif stage == "finetune":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown stage: {stage}")


def run_epoch(model, loader, optimizer, criterion, device, lambda_res, use_residual):
    model.train()
    total_loss = 0.0
    total_fit = 0.0
    total_res = 0.0
    n = 0

    for x_seq, y in tqdm(loader, desc="Training", leave=False):
        x_seq = x_seq.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        y_hat, y_res = model(x_seq, use_residual=use_residual)

        loss_fit = criterion(y_hat, y)
        loss_res = torch.mean(torch.abs(y_res))
        loss = loss_fit + (lambda_res * loss_res if use_residual else 0.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_fit += float(loss_fit.item())
        total_res += float(loss_res.item())
        n += 1

    return {
        "loss": total_loss / max(n, 1),
        "fit": total_fit / max(n, 1),
        "res": total_res / max(n, 1),
    }


@torch.no_grad()
def evaluate(model, loader, dataset, device, use_residual=True):
    model.eval()

    preds_norm = []
    targets_norm = []
    residual_l1 = []

    for x_seq, y in loader:
        x_seq = x_seq.to(device)
        y = y.to(device)

        y_hat, y_res = model(x_seq, use_residual=use_residual)
        preds_norm.append(y_hat.cpu().numpy())
        targets_norm.append(y.cpu().numpy())
        residual_l1.append(torch.mean(torch.abs(y_res)).item())

    preds_norm = np.concatenate(preds_norm, axis=0)
    targets_norm = np.concatenate(targets_norm, axis=0)

    preds = preds_norm * dataset.Y_std + dataset.Y_mean
    targets = targets_norm * dataset.Y_std + dataset.Y_mean
    residuals = targets - preds

    metrics = compute_channel_metrics(targets, residuals)

    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "avg_std": float(np.mean(metrics["residual_std"])),
        "res_l1": float(np.mean(residual_l1)),
        "channel_metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train Method B (MLP main + TCN residual, Loss3)"
    )

    default_data_dir = find_default_data_dir()
    default_out_dir = f"train/outputs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    parser.add_argument("--data_dir", type=str, default=default_data_dir)
    parser.add_argument("--glob", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=default_out_dir)

    parser.add_argument("--val_split", type=str, default="file", choices=["file", "random"])
    parser.add_argument("--val_ratio", type=float, default=0.2)

    parser.add_argument("--use_vel", type=int, default=0,
                        help="Deprecated. Input is fixed to 12-dim sin/cos.")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--x_norm", type=int, default=1)
    parser.add_argument("--y_norm", type=int, default=1)
    parser.add_argument("--std_floor", type=float, default=1e-2)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--stage", type=str, default="finetune",
                        choices=["main_only", "res_only", "finetune"])
    parser.add_argument("--lambda_res", type=float, default=1e-3)

    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--head_hidden", type=int, default=64)
    parser.add_argument("--tcn_hidden", type=int, default=64)
    parser.add_argument("--tcn_kernel", type=int, default=3)
    parser.add_argument("--tcn_dilations", type=str, default="1,2,4,8")
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=50)

    args = parser.parse_args()

    if args.data_dir is None:
        raise ValueError("No data directory found. Please specify --data_dir")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = ensure_dir(args.out_dir)

    all_files = find_files_by_pattern(args.data_dir, "robot_data_*.txt")
    if not all_files:
        all_files = find_files_by_pattern(args.data_dir, "*.txt")
    if not all_files:
        raise ValueError(f"No data files found in {args.data_dir}")

    filepaths = find_files_by_pattern(args.data_dir, args.glob) if args.glob else []
    if not filepaths:
        print("[INFO] --glob not matched. using all robot_data files.")
        filepaths = all_files

    train_files, val_files = split_files_train_val(filepaths, args.val_ratio, args.val_split, args.seed)
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    if len(train_files) == 0:
        raise ValueError("No training files after split")

    print("Loading train files...")
    x_train_files, y_train_files = load_xy_files(train_files, use_vel=False)

    # Fallback if val files do not exist (e.g., only one file)
    if len(val_files) > 0:
        print("Loading val files...")
        x_val_files, y_val_files = load_xy_files(val_files, use_vel=False)
    else:
        print("[WARN] No val files. Splitting each train file into train/val chunks.")
        x_val_files, y_val_files = [], []
        new_x_train_files, new_y_train_files = [], []
        for xf, yf in zip(x_train_files, y_train_files):
            n = len(xf)
            split_idx = max(args.seq_len, int(n * (1.0 - args.val_ratio)))
            split_idx = min(split_idx, n - args.seq_len)
            if split_idx <= args.seq_len:
                continue
            new_x_train_files.append(xf[:split_idx])
            new_y_train_files.append(yf[:split_idx])
            x_val_files.append(xf[split_idx - args.seq_len + 1:])
            y_val_files.append(yf[split_idx - args.seq_len + 1:])

        if not new_x_train_files or not x_val_files:
            raise ValueError(
                "Could not create valid train/val split for seq training. Add more data or reduce --seq_len."
            )
        x_train_files, y_train_files = new_x_train_files, new_y_train_files

    train_dataset = SelfDetectionSeqDataset(
        x_train_files,
        y_train_files,
        seq_len=args.seq_len,
        stride=args.stride,
        std_floor=args.std_floor,
    )
    val_dataset = SelfDetectionSeqDataset(
        x_val_files,
        y_val_files,
        seq_len=args.seq_len,
        stride=args.stride,
        X_mean=train_dataset.X_mean,
        X_std=train_dataset.X_std,
        Y_mean=train_dataset.Y_mean,
        Y_std=train_dataset.Y_std,
        std_floor=args.std_floor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dilations = tuple(int(x.strip()) for x in args.tcn_dilations.split(",") if x.strip())
    if not dilations:
        raise ValueError("--tcn_dilations is empty")

    model = MLP_TCN_ResidualModel(
        in_dim=12,
        out_dim=8,
        trunk_hidden=args.hidden,
        head_hidden=args.head_hidden,
        tcn_hidden=args.tcn_hidden,
        tcn_kernel=args.tcn_kernel,
        tcn_dilations=dilations,
        dropout=args.dropout,
    ).to(device)

    configure_stage(model, args.stage)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters. stage configuration is invalid.")

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"Model params: {num_params:,} (trainable: {num_trainable:,})")
    print(
        f"stage={args.stage}, seq_len={args.seq_len}, lambda_res={args.lambda_res}, "
        f"dilations={dilations}"
    )

    criterion = nn.SmoothL1Loss()
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.wd)
    try:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, verbose=True)
    except TypeError:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)

    history = {
        "train_loss": [],
        "train_fit": [],
        "train_res": [],
        "val_avg_std": [],
        "val_mae": [],
        "val_rmse": [],
        "val_res_l1": [],
    }

    best_val_std = float("inf")
    best_epoch = 0
    stale_epochs = 0

    print("\n" + "=" * 60)
    print("Starting training (Method B: MLP main + TCN residual)")
    print("=" * 60)

    use_residual_train = args.stage != "main_only"

    for epoch in range(1, args.epochs + 1):
        train_stat = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args.lambda_res,
            use_residual=use_residual_train,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            val_dataset,
            device,
            use_residual=use_residual_train,
        )

        scheduler.step(val_metrics["avg_std"])

        history["train_loss"].append(train_stat["loss"])
        history["train_fit"].append(train_stat["fit"])
        history["train_res"].append(train_stat["res"])
        history["val_avg_std"].append(val_metrics["avg_std"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["val_res_l1"].append(val_metrics["res_l1"])

        if val_metrics["avg_std"] < best_val_std:
            best_val_std = val_metrics["avg_std"]
            best_epoch = epoch
            stale_epochs = 0
            norm_params = train_dataset.get_norm_params()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "normalization": norm_params,
                    "model_type": "mlp_tcn_residual",
                },
                out_dir / "model.pt",
            )
        else:
            stale_epochs += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss: {train_stat['loss']:.6f} (fit={train_stat['fit']:.6f}, res={train_stat['res']:.6f}) | "
                f"Val STD: {val_metrics['avg_std']:.2f} | Val MAE: {val_metrics['mae']:.2f} | "
                f"Val |res|: {val_metrics['res_l1']:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

        if stale_epochs >= args.early_stop:
            print(f"Early stopping at epoch {epoch} (patience={args.early_stop})")
            break

    print(f"\nBest model at epoch {best_epoch} (Val STD: {best_val_std:.2f})")

    checkpoint = torch.load(out_dir / "model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_metrics = evaluate(
        model,
        val_loader,
        val_dataset,
        device,
        use_residual=use_residual_train,
    )

    norm_params = train_dataset.get_norm_params()
    save_norm_params(
        norm_params["X_mean"],
        norm_params["X_std"],
        norm_params["Y_mean"],
        norm_params["Y_std"],
        args.std_floor,
        str(out_dir / "norm_params.json"),
    )

    save_json(vars(args), str(out_dir / "config.json"))

    report = {
        "model_type": "mlp_tcn_residual",
        "best_epoch": best_epoch,
        "best_val_std": float(best_val_std),
        "final_metrics": {
            "mae": final_metrics["mae"],
            "rmse": final_metrics["rmse"],
            "avg_std": final_metrics["avg_std"],
            "res_l1": final_metrics["res_l1"],
        },
        "channel_metrics": {
            "raw_std": final_metrics["channel_metrics"]["raw_std"].tolist(),
            "residual_std": final_metrics["channel_metrics"]["residual_std"].tolist(),
            "improvement": final_metrics["channel_metrics"]["improvement"].tolist(),
        },
    }
    save_json(report, str(out_dir / "report.json"))

    with open(out_dir / "history.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_fit",
            "train_res",
            "val_avg_std",
            "val_mae",
            "val_rmse",
            "val_res_l1",
        ])
        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["train_fit"][i],
                history["train_res"][i],
                history["val_avg_std"][i],
                history["val_mae"][i],
                history["val_rmse"][i],
                history["val_res_l1"][i],
            ])

    channel_names = [f"raw{i}" for i in range(1, 9)]
    print("\n" + "=" * 60)
    print("Final Report")
    print("=" * 60)
    print(format_metrics_report(final_metrics["channel_metrics"], channel_names))
    print(f"\nModel saved to: {out_dir / 'model.pt'}")
    print(f"Normalization params saved to: {out_dir / 'norm_params.json'}")


if __name__ == "__main__":
    main()
