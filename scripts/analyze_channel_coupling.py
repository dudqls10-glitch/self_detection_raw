#!/usr/bin/env python3
"""
Analyze channel coupling in multi-channel self-detection sensor predictions.

Outputs:
- Correlation matrices for target / prediction / residual
- Lag-1 correlation matrices for target / residual
- CSV exports
- Heatmap plots
- Summary statistics for adjacent/non-adjacent coupling

Usage:
    python3 scripts/analyze_channel_coupling.py
    python3 scripts/analyze_channel_coupling.py --model scripts/model/<run>/model.pt
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from self_detection_raw.data.loader_v import load_multiple_files_v
from self_detection_raw.models.mlp_b_v import ModelBV
from self_detection_raw.utils.io import ensure_dir

CHANNEL_NAMES = [f"raw{i}" for i in range(1, 9)]
DEFAULT_BATCH = 4096
DEFAULT_TOP_K = 5


def full_paths(names, base_dir):
    paths = []
    for fn in names:
        p = Path(fn)
        if not p.is_absolute():
            paths.append(str(Path(base_dir) / fn))
        else:
            paths.append(str(p))
    return paths


def parse_file_list(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    return [item.strip() for item in str(value).split(",") if item.strip()]


def find_latest_model():
    model_dir = Path(__file__).parent / "model"
    candidates = sorted(model_dir.glob("**/model.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else None


def load_checkpoint(model_path, device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("args", {})
    norm = checkpoint.get("normalization")
    if norm is None:
        raise ValueError(f"Checkpoint has no normalization block: {model_path}")

    in_dim = checkpoint["model_state_dict"]["trunk.0.weight"].shape[1]
    model = ModelBV(
        in_dim=in_dim,
        trunk_hidden=int(model_args.get("hidden", 128)),
        head_hidden=int(model_args.get("head_hidden", 64)),
        out_dim=len(CHANNEL_NAMES),
        dropout=float(model_args.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    norm_params = {
        "X_mean": np.asarray(norm["X_mean"], dtype=np.float32),
        "X_std": np.asarray(norm["X_std"], dtype=np.float32),
        "Y_mean": np.asarray(norm["Y_mean"], dtype=np.float32),
        "Y_std": np.asarray(norm["Y_std"], dtype=np.float32),
    }
    return checkpoint, model_args, model, norm_params


@torch.no_grad()
def predict_raw(model, X_raw, norm_params, device, batch_size):
    X_norm = (X_raw - norm_params["X_mean"]) / norm_params["X_std"]
    preds = []
    for start in range(0, len(X_norm), batch_size):
        batch = torch.from_numpy(X_norm[start:start + batch_size].astype(np.float32)).to(device)
        pred_norm = model(batch).cpu().numpy()
        preds.append(pred_norm)
    pred_norm = np.concatenate(preds, axis=0)
    return pred_norm * norm_params["Y_std"] + norm_params["Y_mean"]


def correlation_dataframe(data, channel_names):
    df = pd.DataFrame(data, columns=channel_names)
    corr = df.corr(method="pearson").fillna(0.0)
    return corr


def lag1_correlation_dataframe(data, channel_names):
    if len(data) < 2:
        raise ValueError("Need at least 2 samples to compute lag-1 correlation")

    current = data[1:]
    previous = data[:-1]
    n_channels = data.shape[1]
    corr = np.zeros((n_channels, n_channels), dtype=np.float64)

    for row_idx in range(n_channels):
        x = current[:, row_idx]
        x_std = np.std(x)
        for col_idx in range(n_channels):
            y = previous[:, col_idx]
            y_std = np.std(y)
            if x_std < 1e-12 or y_std < 1e-12:
                value = 0.0
            else:
                value = float(np.corrcoef(x, y)[0, 1])
            corr[row_idx, col_idx] = 0.0 if np.isnan(value) else value

    index = [f"{name}(t)" for name in channel_names]
    columns = [f"{name}(t-1)" for name in channel_names]
    return pd.DataFrame(corr, index=index, columns=columns)


def save_matrix_csv(matrix_df, out_dir, stem):
    path = Path(out_dir) / f"{stem}.csv"
    matrix_df.to_csv(path, float_format="%.6f")
    return path


def save_heatmap(matrix_df, out_dir, stem, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix_df.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(matrix_df.columns)))
    ax.set_yticks(np.arange(len(matrix_df.index)))
    ax.set_xticklabels(matrix_df.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix_df.index)
    ax.set_title(title)

    for row_idx in range(matrix_df.shape[0]):
        for col_idx in range(matrix_df.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{matrix_df.iloc[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    fig.tight_layout()
    path = Path(out_dir) / f"{stem}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def are_adjacent(i, j, n_channels, wrap_around):
    if abs(i - j) == 1:
        return True
    if wrap_around and {i, j} == {0, n_channels - 1}:
        return True
    return False


def summarize_matrix(matrix_df, base_channel_names, name, symmetric, wrap_around, top_k):
    pairs = []
    n_channels = len(base_channel_names)

    for row_idx in range(n_channels):
        col_range = range(row_idx + 1, n_channels) if symmetric else range(n_channels)
        for col_idx in col_range:
            if row_idx == col_idx:
                continue
            value = float(matrix_df.iloc[row_idx, col_idx])
            if symmetric:
                pair_name = f"{base_channel_names[row_idx]} <-> {base_channel_names[col_idx]}"
            else:
                pair_name = f"{base_channel_names[col_idx]}(t-1) -> {base_channel_names[row_idx]}(t)"
            pairs.append(
                {
                    "pair": pair_name,
                    "row_channel": base_channel_names[row_idx],
                    "col_channel": base_channel_names[col_idx],
                    "correlation": value,
                    "abs_correlation": abs(value),
                    "adjacent": are_adjacent(row_idx, col_idx, n_channels, wrap_around),
                }
            )

    adjacent_pairs = [item for item in pairs if item["adjacent"]]
    non_adjacent_pairs = [item for item in pairs if not item["adjacent"]]
    strongest = sorted(pairs, key=lambda item: item["abs_correlation"], reverse=True)[:top_k]
    weakest = sorted(pairs, key=lambda item: item["abs_correlation"])[:top_k]

    def avg(items, key):
        if not items:
            return 0.0
        return float(np.mean([item[key] for item in items]))

    return {
        "matrix_name": name,
        "avg_adjacent_corr": avg(adjacent_pairs, "correlation"),
        "avg_adjacent_abs_corr": avg(adjacent_pairs, "abs_correlation"),
        "avg_non_adjacent_corr": avg(non_adjacent_pairs, "correlation"),
        "avg_non_adjacent_abs_corr": avg(non_adjacent_pairs, "abs_correlation"),
        "strongest_pairs": strongest,
        "weakest_pairs": weakest,
    }


def format_summary(summary):
    lines = [
        f"[{summary['matrix_name']}]",
        f"Average adjacent correlation: {summary['avg_adjacent_corr']:.4f}",
        f"Average adjacent |correlation|: {summary['avg_adjacent_abs_corr']:.4f}",
        f"Average non-adjacent correlation: {summary['avg_non_adjacent_corr']:.4f}",
        f"Average non-adjacent |correlation|: {summary['avg_non_adjacent_abs_corr']:.4f}",
        "Strongest coupled pairs:",
    ]
    for item in summary["strongest_pairs"]:
        lines.append(f"  {item['pair']}: corr={item['correlation']:.4f}")
    lines.append("Least coupled pairs:")
    for item in summary["weakest_pairs"]:
        lines.append(f"  {item['pair']}: corr={item['correlation']:.4f}")
    return "\n".join(lines)


def save_summary_files(out_dir, matrix_summaries, metadata):
    payload = {
        "metadata": metadata,
        "matrix_summaries": matrix_summaries,
    }
    json_path = Path(out_dir) / "coupling_summary.json"
    txt_path = Path(out_dir) / "coupling_summary.txt"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Channel Coupling Summary\n")
        f.write("========================\n\n")
        for summary in matrix_summaries:
            f.write(format_summary(summary))
            f.write("\n\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze channel coupling from validation predictions")
    parser.add_argument("--model", default=find_latest_model(), help="Path to model.pt checkpoint")
    parser.add_argument("--data-dir", default=None, help="Directory containing validation files")
    parser.add_argument("--val-files", default=None, help="Comma-separated validation file list")
    parser.add_argument("--vel-window", type=int, default=None, help="Velocity smoothing window")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Batch size for inference")
    parser.add_argument("--out-dir", default=None, help="Output directory for analysis artifacts")
    parser.add_argument("--wrap-adjacent", type=int, default=1, help="Treat raw8/raw1 as adjacent")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Top-K strongest/weakest pairs")

    args = parser.parse_args()

    if args.model is None:
        raise ValueError("No model checkpoint found. Please provide --model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint, model_args, model, norm_params = load_checkpoint(args.model, device)

    data_dir = args.data_dir or model_args.get("data_dir")
    if data_dir is None:
        raise ValueError("No data directory found. Please provide --data-dir")

    val_files = parse_file_list(args.val_files)
    if val_files is None:
        val_files = parse_file_list(model_args.get("val_files"))
    if not val_files:
        raise ValueError("No validation files found. Please provide --val-files or use a checkpoint with saved val_files")

    vel_window = args.vel_window if args.vel_window is not None else int(model_args.get("vel_window", 10))
    val_paths = full_paths(val_files, data_dir)

    X_val, Y_val = load_multiple_files_v(val_paths, use_vel=True, vel_window=vel_window)
    if len(X_val) == 0:
        raise ValueError("Validation data is empty. Check data_dir and val_files")

    preds = predict_raw(model, X_val, norm_params, device, args.batch)
    residuals = Y_val - preds

    out_dir = args.out_dir or str(Path(args.model).resolve().parent / "coupling_analysis")
    ensure_dir(out_dir)

    matrices = {
        "target_corr": correlation_dataframe(Y_val, CHANNEL_NAMES),
        "prediction_corr": correlation_dataframe(preds, CHANNEL_NAMES),
        "residual_corr": correlation_dataframe(residuals, CHANNEL_NAMES),
        "target_lag1_corr": lag1_correlation_dataframe(Y_val, CHANNEL_NAMES),
        "residual_lag1_corr": lag1_correlation_dataframe(residuals, CHANNEL_NAMES),
    }

    titles = {
        "target_corr": "Ground-Truth Channel Correlation",
        "prediction_corr": "Predicted Channel Correlation",
        "residual_corr": "Residual Channel Correlation",
        "target_lag1_corr": "Ground-Truth Lag-1 Correlation",
        "residual_lag1_corr": "Residual Lag-1 Correlation",
    }

    matrix_summaries = []
    for name, matrix_df in matrices.items():
        save_matrix_csv(matrix_df, out_dir, name)
        save_heatmap(matrix_df, out_dir, name, titles[name])
        matrix_summaries.append(
            summarize_matrix(
                matrix_df,
                CHANNEL_NAMES,
                name=name,
                symmetric=not name.endswith("lag1_corr"),
                wrap_around=bool(args.wrap_adjacent),
                top_k=args.top_k,
            )
        )

    metadata = {
        "model": str(Path(args.model).resolve()),
        "data_dir": str(Path(data_dir).resolve()),
        "val_files": val_files,
        "vel_window": vel_window,
        "num_samples": int(len(Y_val)),
        "wrap_adjacent": bool(args.wrap_adjacent),
        "top_k": int(args.top_k),
    }
    save_summary_files(out_dir, matrix_summaries, metadata)

    print(f"Saved coupling analysis to: {out_dir}")
    for summary in matrix_summaries:
        print(format_summary(summary))
        print("")


if __name__ == "__main__":
    main()
