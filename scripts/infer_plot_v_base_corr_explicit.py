#!/usr/bin/env python3
"""
Inference and plotting for the explicit base + correction model.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from self_detection_raw.data.loader import load_file
from self_detection_raw.data.loader_v import smooth_data
from self_detection_raw.models.mlp_b_v import ModelBV

# ---------------------------------------------------------------------------
# Editable defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "/home/song/rb10_Proximity/src/self_detection_raw/dataset"
DEFAULT_INPUT_FILE = "[4]dataset_100_25_new.txt"
# ---------------------------------------------------------------------------


def find_latest_model():
    model_dir = Path(__file__).parent / "model"
    files = list(model_dir.glob("mlp_vel_base_corr_explicit_*/model.pt"))
    if not files:
        files = list(model_dir.glob("**/model.pt"))
    if not files:
        return None
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(files[0])


def encode_joint_positions_deg(joint_pos_deg):
    joint_pos_rad = np.deg2rad(joint_pos_deg)
    return np.concatenate([np.sin(joint_pos_rad), np.cos(joint_pos_rad)], axis=1)


def build_sequence_examples(data, use_vel=True, vel_window=10):
    if len(data) <= 2:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 0), dtype=np.float32),
            np.empty((0, 8), dtype=np.float32),
        )

    q_deg = data[:, 1:7]
    q_rad = np.deg2rad(q_deg)
    q_enc = encode_joint_positions_deg(q_deg)

    dq = np.zeros_like(q_rad, dtype=np.float32)
    dq[1:] = (q_rad[1:] - q_rad[:-1]).astype(np.float32)

    base_parts = [q_enc[2:]]
    if use_vel:
        jv = smooth_data(data[:, 7:13], window_size=vel_window, polyorder=2)
        base_parts.append(jv[2:].astype(np.float32))
    base_x = np.concatenate(base_parts, axis=1).astype(np.float32)

    corr_x = np.concatenate(
        [
            q_enc[1:-1],
            q_enc[:-2],
            dq[2:],
            dq[1:-1],
        ],
        axis=1,
    ).astype(np.float32)

    y = data[2:, 21:29].astype(np.float32)
    return base_x, corr_x, y


class CorrectionPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, corr_input):
        return torch.tanh(self.net(corr_input))


class FullPredictor(nn.Module):
    def __init__(
        self,
        base_in_dim,
        corr_in_dim,
        out_dim,
        trunk_hidden,
        head_hidden,
        dropout,
        corr_hidden,
        corr_dropout,
        lambda_norm,
    ):
        super().__init__()
        self.base_predictor = ModelBV(
            in_dim=base_in_dim,
            trunk_hidden=trunk_hidden,
            head_hidden=head_hidden,
            out_dim=out_dim,
            dropout=dropout,
        )
        self.correction_predictor = CorrectionPredictor(
            in_dim=corr_in_dim,
            hidden_dim=corr_hidden,
            out_dim=out_dim,
            dropout=corr_dropout,
        )
        self.register_buffer("lambda_norm", torch.as_tensor(lambda_norm, dtype=torch.float32))

    def forward(self, base_input, corr_input):
        s_base = self.base_predictor(base_input)
        c = self.correction_predictor(corr_input)
        lambda_norm = self.lambda_norm.to(device=s_base.device, dtype=s_base.dtype)
        s_hat = s_base + lambda_norm * c
        return s_hat, s_base, c


def load_norm_params(args, checkpoint):
    if args.norm_path and os.path.exists(args.norm_path):
        with open(args.norm_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if "normalization" in checkpoint:
        return checkpoint["normalization"]
    return None


def main():
    parser = argparse.ArgumentParser(description="Inference for explicit base + correction model")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--norm-path", type=str, default=None)
    parser.add_argument("--output-plot", type=str, default=None)
    parser.add_argument("--vel-window", type=int, default=10)
    parser.add_argument("--use-hw-baseline", dest="use_hardware_baseline", action="store_true")
    parser.add_argument("--use-mean-baseline", dest="use_hardware_baseline", action="store_false")
    parser.set_defaults(use_hardware_baseline=True)
    args = parser.parse_args()

    print("Starting explicit base-correction inference script...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model is None:
        args.model = find_latest_model()
        if args.model is None:
            print("Error: No model found in scripts/model/ and --model not specified.")
            return
        print(f"Auto-selected latest model: {args.model}")

    if args.data is None:
        candidate = DEFAULT_INPUT_FILE if os.path.isabs(DEFAULT_INPUT_FILE) else os.path.join(DEFAULT_DATA_DIR, DEFAULT_INPUT_FILE)
        args.data = candidate

    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    norm_params = load_norm_params(args, checkpoint)
    if norm_params is None:
        print("Error: Normalization params could not be loaded.")
        return

    use_vel = bool(train_args.get("use_vel", True))
    vel_window = int(train_args.get("vel_window", args.vel_window))
    base_mean = np.asarray(norm_params["base_mean"], dtype=np.float32)
    base_std = np.asarray(norm_params["base_std"], dtype=np.float32)
    corr_mean = np.asarray(norm_params["corr_mean"], dtype=np.float32)
    corr_std = np.asarray(norm_params["corr_std"], dtype=np.float32)
    y_mean = np.asarray(norm_params["Y_mean"], dtype=np.float32)
    y_std = np.asarray(norm_params["Y_std"], dtype=np.float32)
    lambda_norm = np.asarray(train_args["lambda_norm"], dtype=np.float32)

    print(f"Loading model from {args.model}...")
    print(f"Loading data from {args.data}...")

    data = load_file(args.data)
    base_x, corr_x, y_raw = build_sequence_examples(data, use_vel=use_vel, vel_window=vel_window)
    if base_x.size == 0:
        print("Error: Not enough samples for history alignment.")
        return

    base_norm = (base_x - base_mean) / base_std
    corr_norm = (corr_x - corr_mean) / corr_std

    model = FullPredictor(
        base_in_dim=base_x.shape[1],
        corr_in_dim=corr_x.shape[1],
        out_dim=y_raw.shape[1],
        trunk_hidden=int(train_args.get("hidden", 128)),
        head_hidden=int(train_args.get("head_hidden", 64)),
        dropout=float(train_args.get("dropout", 0.1)),
        corr_hidden=int(train_args.get("corr_hidden", 32)),
        corr_dropout=float(train_args.get("corr_dropout", 0.05)),
        lambda_norm=lambda_norm,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        base_tensor = torch.FloatTensor(base_norm).to(device)
        corr_tensor = torch.FloatTensor(corr_norm).to(device)
        pred_norm, base_pred_norm, c = model(base_tensor, corr_tensor)
        pred_norm = pred_norm.cpu().numpy()
        base_pred_norm = base_pred_norm.cpu().numpy()
        c = c.cpu().numpy()

    pred = pred_norm * y_std + y_mean
    base_pred = base_pred_norm * y_std + y_mean
    corr_raw = c * (lambda_norm * y_std)
    residual = y_raw - pred

    baseline = np.full(y_raw.shape[1], 4.0e7, dtype=np.float32) if args.use_hardware_baseline else y_mean
    compensated = residual + baseline

    fig, axes = plt.subplots(4, 2, figsize=(15, 10), sharex=True)
    axes = axes.flatten()
    t = np.arange(len(y_raw))

    for i in range(8):
        ax = axes[i]
        ax_comp = ax.twinx()
        raw_line = ax.plot(t, y_raw[:, i], label="raw", color="C0", alpha=0.3)
        base_line = ax.plot(t, base_pred[:, i], label="base", color="C3", alpha=0.6)
        pred_line = ax.plot(t, pred[:, i], label="pred", color="C1", alpha=0.7)
        comp_line = ax_comp.plot(t, compensated[:, i], label="compensated", color="C2", alpha=0.4)
        ax.set_title(
            f"raw{i+1} | ResStd: {np.std(residual[:, i]):.0f} | Mean|corr|: {np.mean(np.abs(corr_raw[:, i])):.0f}",
            fontsize=10,
        )
        ax.grid(True, alpha=0.3)
        ax_comp.tick_params(axis="y", colors="C2")
        ax_comp.spines["right"].set_color("C2")
        if i == 0:
            lines = raw_line + base_line + pred_line + comp_line
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels, loc="upper right")

    fig.suptitle(f"Explicit base + correction inference on {os.path.basename(args.data)}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = args.output_plot or os.path.join(os.path.dirname(args.model), "inference_plot_v_base_corr_explicit.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    print(f"Saved plot to {out_path}")
    print("Per-channel correction stats (mean abs, max abs):")
    for i in range(8):
        print(f" raw{i+1}: {np.mean(np.abs(corr_raw[:, i])):.3f}, {np.max(np.abs(corr_raw[:, i])):.3f}")


if __name__ == "__main__":
    main()
