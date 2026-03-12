#!/usr/bin/env python3
"""
Inference and plotting for delta-target self-detection models.

Target at training:
    delta_sensor_t = sensor_t - sensor_{t-1}

Reconstruction at inference:
    pred_sensor_t = sensor_{t-1} + pred_delta_sensor_t
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from self_detection_raw.data.loader import load_file, extract_features
from self_detection_raw.data.loader_v import extract_features_v
from self_detection_raw.models.mlp_b_v import ModelBV

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "/home/song/rb10_Proximity/src/self_detection_raw/dataset"
DEFAULT_INPUT_FILE = "5dataset_50_25_1.txt"
CHANNEL_NAMES = [f"raw{i}" for i in range(1, 9)]
HARDWARE_BASELINE = 4.0e+07
# ---------------------------------------------------------------------------


def find_latest_model():
    model_dir = Path(__file__).parent / "model"
    if not model_dir.exists():
        return None
    files = sorted(model_dir.glob("**/model.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None


def find_latest_data(base_dir=None):
    dataset_dir = Path(base_dir) if base_dir and os.path.exists(base_dir) else Path(__file__).parent.parent / "dataset"
    if not dataset_dir.exists():
        return None
    files = sorted(dataset_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0]) if files else None


def parse_prev_sensor_indices(train_args):
    indices = train_args.get("prev_sensor_indices")
    if indices is not None:
        return [int(idx) for idx in indices]
    return list(range(len(CHANNEL_NAMES)))


def resolve_data_path(data_arg):
    if data_arg is None:
        if DEFAULT_INPUT_FILE:
            candidate = (
                DEFAULT_INPUT_FILE
                if os.path.isabs(DEFAULT_INPUT_FILE)
                else os.path.join(DEFAULT_DATA_DIR, DEFAULT_INPUT_FILE)
            )
            if os.path.exists(candidate):
                return candidate
        return find_latest_data(DEFAULT_DATA_DIR)

    if os.path.exists(data_arg):
        return data_arg

    candidate = os.path.join(DEFAULT_DATA_DIR, data_arg)
    if os.path.exists(candidate):
        return candidate
    return data_arg


def extract_joint_and_sensor(data, use_vel, vel_window):
    if use_vel:
        return extract_features_v(data, use_vel=True, vel_window=vel_window)
    return extract_features(data, use_vel=False)


def build_delta_inference_inputs(X_joint, sensor_abs, use_prev_sensor, prev_sensor_indices):
    if len(X_joint) < 2:
        raise ValueError("Need at least 2 samples for delta inference")

    X_curr = X_joint[1:]
    prev_sensor_full = sensor_abs[:-1]
    sensor_curr = sensor_abs[1:]
    aligned_indices = np.arange(1, len(sensor_abs))

    if use_prev_sensor:
        prev_selected = prev_sensor_full[:, prev_sensor_indices]
        X_input = np.concatenate([X_curr, prev_selected], axis=1)
    else:
        X_input = X_curr

    return X_input, prev_sensor_full, sensor_curr, aligned_indices


def main():
    parser = argparse.ArgumentParser(description="Inference and plot for delta self-detection model")
    parser.add_argument("--model", type=str, default=None, help="Path to model.pt (default: latest)")
    parser.add_argument("--data", type=str, default=None, help="Path to input .txt data file")
    parser.add_argument("--output-plot", type=str, default=None, help="Optional output plot path")
    parser.add_argument("--vel-window", type=int, default=10, help="Fallback velocity window")
    parser.add_argument("--use-hw-baseline", dest="use_hardware_baseline", action="store_true")
    parser.add_argument("--use-mean-baseline", dest="use_hardware_baseline", action="store_false")
    parser.set_defaults(use_hardware_baseline=True)
    args = parser.parse_args()

    print("Starting delta inference script...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model is None:
        args.model = find_latest_model()
        if args.model is None:
            print("Error: No model found in scripts/model/ and --model not specified.")
            return
        print(f"Auto-selected latest model: {args.model}")

    if not os.path.exists(args.model):
        print(f"Error: Model file not found {args.model}")
        return

    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    train_args = checkpoint.get("args", {})
    norm = checkpoint.get("normalization")
    if norm is None:
        print("Error: Normalization not found in checkpoint.")
        return

    args.data = resolve_data_path(args.data)
    if args.data is None or not os.path.exists(args.data):
        print("Error: No data file found and --data not specified.")
        return
    print(f"Selected data: {args.data}")

    use_vel = bool(train_args.get("use_vel", True))
    use_prev_sensor = bool(train_args.get("use_prev_sensor", False))
    prev_sensor_indices = parse_prev_sensor_indices(train_args)
    vel_window = int(train_args.get("vel_window", args.vel_window))
    print(f"Using velocity input: {use_vel}")
    print(f"Using previous sensor input: {use_prev_sensor}")
    if use_prev_sensor:
        selected_names = [CHANNEL_NAMES[idx] for idx in prev_sensor_indices]
        print("Previous sensor channels: " + ", ".join(selected_names))
    print(f"Using Velocity Window: {vel_window}")

    X_mean = np.asarray(norm["X_mean"], dtype=np.float32)
    X_std = np.asarray(norm["X_std"], dtype=np.float32)
    Y_mean = np.asarray(norm["Y_mean"], dtype=np.float32)  # delta mean
    Y_std = np.asarray(norm["Y_std"], dtype=np.float32)    # delta std

    print(f"Loading data from {args.data}...")
    raw_data = load_file(args.data)
    X_joint, sensor_abs = extract_joint_and_sensor(raw_data, use_vel=use_vel, vel_window=vel_window)
    X_raw, prev_sensor_abs, sensor_curr_abs, aligned_indices = build_delta_inference_inputs(
        X_joint,
        sensor_abs,
        use_prev_sensor=use_prev_sensor,
        prev_sensor_indices=prev_sensor_indices,
    )

    X_norm = (X_raw - X_mean) / X_std
    X_tensor = torch.from_numpy(X_norm.astype(np.float32)).to(device)

    model = ModelBV(
        in_dim=X_raw.shape[1],
        trunk_hidden=int(train_args.get("hidden", 128)),
        head_hidden=int(train_args.get("head_hidden", 64)),
        out_dim=sensor_curr_abs.shape[1],
        dropout=float(train_args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Running inference...")
    with torch.no_grad():
        pred_delta_norm = model(X_tensor).cpu().numpy()

    pred_delta = pred_delta_norm * Y_std + Y_mean
    pred_sensor_abs = prev_sensor_abs + pred_delta
    residual = sensor_curr_abs - pred_sensor_abs

    if args.use_hardware_baseline:
        baseline = np.full(sensor_curr_abs.shape[1], HARDWARE_BASELINE, dtype=np.float32)
    else:
        baseline = np.mean(sensor_curr_abs, axis=0)
    compensated = residual + baseline

    print("Plotting results...")
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    t = aligned_indices
    for i in range(8):
        ax = axes[i]
        ax_comp = ax.twinx()

        raw_line = ax.plot(t, sensor_curr_abs[:, i], label="raw", color="C0", alpha=0.3, zorder=1)
        pred_line = ax.plot(t, pred_sensor_abs[:, i], label="pred", color="C1", alpha=0.5, zorder=2)
        comp_line = ax_comp.plot(t, compensated[:, i], label="compensated", color="C2", alpha=0.4, zorder=3)
        ax_comp.tick_params(axis="y", colors="C2")
        ax_comp.spines["right"].set_color("C2")

        resid_std = np.std(residual[:, i])
        comp_std = np.std(compensated[:, i])
        ax.set_title(f"raw{i+1} | ResStd: {resid_std:.0f} | CompStd: {comp_std:.0f}", fontsize=10)
        if i == 0:
            lines = raw_line + pred_line + comp_line
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels, loc="upper right")
        ax.grid(True, alpha=0.3)

    file_note = os.path.basename(args.data) + " (delta reconstruction, first sample dropped)"
    fig.suptitle(f"Inference results on {file_note}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_path = args.output_plot or os.path.join(os.path.dirname(args.model), "inference_plot_v_delta.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving plot to {out_path}...")
    plt.savefig(out_path)
    plt.close()

    if os.path.exists(out_path):
        print("\n" + "=" * 60)
        print("SUCCESS: Plot generated successfully!")
        print(f"Location: {os.path.abspath(out_path)}")
        print("=" * 60 + "\n")
    else:
        print(f"Error: Failed to save plot to {out_path}")

    print("Absolute prediction stats (RMSE per channel):")
    for i in range(8):
        rmse = np.sqrt(np.mean((sensor_curr_abs[:, i] - pred_sensor_abs[:, i]) ** 2))
        mae = np.mean(np.abs(sensor_curr_abs[:, i] - pred_sensor_abs[:, i]))
        print(f" raw{i+1}: RMSE={rmse:.3f}, MAE={mae:.3f}")


if __name__ == "__main__":
    main()
