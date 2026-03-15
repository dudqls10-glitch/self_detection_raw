#!/usr/bin/env python3
"""
Real-time inference node for the explicit base + correction model.

Model structure:
    S_hat_t = S_base_t + lambda * C_t

Inputs at time t:
    base_input_t = [enc(q_t), smooth(jv_t)]
    corr_input_t = [enc(q_{t-1}), enc(q_{t-2}), dq_t, dq_{t-1}]

where dq_t = q_t(rad) - q_{t-1}(rad), matching the training script exactly.
"""

import glob
import os
import sys
from collections import deque
from datetime import datetime

import numpy as np
import rclpy
import torch
import torch.nn as nn
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState, Range

from self_detection_raw.data.loader_v import smooth_data
from self_detection_raw.models.mlp_b_v import ModelBV

N_JOINTS = 6
N_SENSORS = 8
HARDWARE_BASELINE = 4.0e+07
CHANNEL_NAMES = [f"raw{i}" for i in range(1, N_SENSORS + 1)]
DEFAULT_MODEL_FILE = None


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


def encode_joint_positions_deg(joint_pos_deg):
    joint_pos_rad = np.deg2rad(joint_pos_deg)
    return np.concatenate([np.sin(joint_pos_rad), np.cos(joint_pos_rad)], axis=0).astype(np.float32)


def find_available_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    possible_dirs = [
        os.path.join(base_dir, "scripts", "model"),
        os.path.join(base_dir, "outputs"),
        os.path.join(base_dir, "self_detection_raw", "train", "outputs"),
    ]

    model_files = []
    for root in possible_dirs:
        if os.path.exists(root):
            model_files.extend(glob.glob(os.path.join(root, "**", "model.pt"), recursive=True))

    model_files = list(set(model_files))
    preferred = [path for path in model_files if "base_corr_explicit" in path]
    candidates = preferred or model_files
    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return candidates


class RealtimeInferBaseCorrExplicitNode(Node):
    def __init__(self):
        super().__init__("realtime_infer_base_corr_explicit")

        self.declare_parameter("model_file", "")
        self.declare_parameter("use_hardware_baseline", True)
        self.declare_parameter("log_rate", 100.0)

        model_file = self.get_parameter("model_file").value
        self.use_hardware_baseline = bool(self.get_parameter("use_hardware_baseline").value)
        log_rate = float(self.get_parameter("log_rate").value)

        self.cb_group = ReentrantCallbackGroup()

        self.raw_data = np.zeros(N_SENSORS, dtype=np.float32)
        self.joint_positions = None
        self.joint_velocities = None
        self.raw_received = False
        self.joint_received = False

        self.model = None
        self.base_mean = None
        self.base_std = None
        self.corr_mean = None
        self.corr_std = None
        self.y_mean = None
        self.y_std = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_vel = True
        self.vel_window = 10
        self.lambda_norm = np.ones(N_SENSORS, dtype=np.float32)
        self.position_history_deg = deque(maxlen=3)
        self.velocity_history_deg = deque(maxlen=10)

        if not model_file:
            if DEFAULT_MODEL_FILE:
                model_file = DEFAULT_MODEL_FILE
                self.get_logger().info(f"Using DEFAULT_MODEL_FILE from Python: {model_file}")
            else:
                available = find_available_models()
                model_file = available[0] if available else None
                if model_file:
                    self.get_logger().info(
                        f"Auto-selected latest model: {os.path.basename(os.path.dirname(model_file))}"
                    )

        if not model_file:
            self.get_logger().error("model_file parameter is required and no compatible models were found")
            self._model_load_failed = True
            return

        self._load_model(model_file)
        if self.model is None:
            self.get_logger().error("Model load failed. Node aborted.")
            self._model_load_failed = True
            return

        for i in range(N_SENSORS):
            self.create_subscription(
                Range,
                f"/raw_distance{i+1}",
                lambda msg, idx=i: self.raw_callback(msg, idx),
                10,
                callback_group=self.cb_group,
            )

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_callback,
            10,
            callback_group=self.cb_group,
        )

        self.comp_pubs = []
        self.pred_pubs = []
        self.base_pubs = []
        self.corr_pubs = []
        for i in range(N_SENSORS):
            self.comp_pubs.append(self.create_publisher(Range, f"/compensated_raw_distance{i+1}", 10))
            self.pred_pubs.append(self.create_publisher(Range, f"/predicted_raw_distance{i+1}", 10))
            self.base_pubs.append(self.create_publisher(Range, f"/predicted_base_raw_distance{i+1}", 10))
            self.corr_pubs.append(self.create_publisher(Range, f"/predicted_correction_raw_distance{i+1}", 10))

        log_dir = os.path.expanduser("~/rb10_Proximity/logs")
        os.makedirs(log_dir, exist_ok=True)
        model_name = os.path.basename(os.path.dirname(model_file)) or "unknown"
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"compensated_raw_base_corr_explicit_{model_name}_{now}.txt")
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        header = (
            "# timestamp "
            + " ".join([f"j{i+1}" for i in range(N_JOINTS)]) + " "
            + " ".join([f"jv{i+1}" for i in range(N_JOINTS)]) + " "
            + " ".join([f"raw{i+1}" for i in range(N_SENSORS)]) + " "
            + " ".join([f"comp{i+1}" for i in range(N_SENSORS)]) + " "
            + " ".join([f"pred{i+1}" for i in range(N_SENSORS)]) + " "
            + " ".join([f"base{i+1}" for i in range(N_SENSORS)]) + " "
            + " ".join([f"corr{i+1}" for i in range(N_SENSORS)]) + "\n"
        )
        self.log_file.write(header)
        self.log_file.flush()

        self.timer = self.create_timer(
            1.0 / log_rate,
            self.timer_callback,
            callback_group=self.cb_group,
        )

        self.get_logger().info("=" * 60)
        self.get_logger().info("Real-time Self Detection Compensation (Explicit Base + Correction)")
        self.get_logger().info(f"Model Path: {model_file}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Use velocities: {self.use_vel}")
        if self.use_vel:
            self.get_logger().info(f"Velocity smoothing window: {self.vel_window}")
        self.get_logger().info("History features: [q_(t-1), q_(t-2), dq_t, dq_(t-1)]")
        self.get_logger().info(
            f"Baseline: {HARDWARE_BASELINE:.2e} (hardware)"
            if self.use_hardware_baseline
            else "Baseline: training target mean"
        )
        self.get_logger().info(f"Log: {self.log_path}")
        self.get_logger().info("=" * 60)

    def _load_model(self, model_file):
        if not os.path.exists(model_file):
            self.get_logger().error(f"Model not found: {model_file}")
            return

        try:
            checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
        except Exception as exc:
            self.get_logger().error(f"Failed to load checkpoint: {exc}")
            return

        model_args = checkpoint.get("args", {})
        normalization = checkpoint.get("normalization")
        if normalization is None:
            self.get_logger().error("Normalization block not found in checkpoint")
            return
        if "base_mean" not in normalization or "corr_mean" not in normalization:
            self.get_logger().error("Checkpoint normalization is not for explicit base + correction model")
            return

        self.use_vel = bool(model_args.get("use_vel", True))
        self.vel_window = int(model_args.get("vel_window", 10))
        self.lambda_norm = np.asarray(model_args.get("lambda_norm"), dtype=np.float32)

        self.model = FullPredictor(
            base_in_dim=int(model_args.get("base_in_dim")),
            corr_in_dim=int(model_args.get("corr_in_dim")),
            out_dim=N_SENSORS,
            trunk_hidden=int(model_args.get("hidden", 128)),
            head_hidden=int(model_args.get("head_hidden", 64)),
            dropout=float(model_args.get("dropout", 0.1)),
            corr_hidden=int(model_args.get("corr_hidden", 32)),
            corr_dropout=float(model_args.get("corr_dropout", 0.05)),
            lambda_norm=self.lambda_norm,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.base_mean = np.asarray(normalization["base_mean"], dtype=np.float32)
        self.base_std = np.asarray(normalization["base_std"], dtype=np.float32)
        self.corr_mean = np.asarray(normalization["corr_mean"], dtype=np.float32)
        self.corr_std = np.asarray(normalization["corr_std"], dtype=np.float32)
        self.y_mean = np.asarray(normalization["Y_mean"], dtype=np.float32)
        self.y_std = np.asarray(normalization["Y_std"], dtype=np.float32)
        self.position_history_deg = deque(maxlen=3)
        self.velocity_history_deg = deque(maxlen=max(self.vel_window, 3))

        self.get_logger().info(f"Model loaded from: {model_file}")
        self.get_logger().info(
            f"Base dim: {model_args.get('base_in_dim')}, Corr dim: {model_args.get('corr_in_dim')}"
        )

    def raw_callback(self, msg, idx):
        self.raw_data[idx] = msg.range
        self.raw_received = True

    def joint_callback(self, msg):
        expected_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        name_to_idx = {name: idx for idx, name in enumerate(msg.name) if name in expected_names}

        joint_positions_rad = []
        joint_velocities_rad = []
        for name in expected_names:
            if name in name_to_idx:
                idx = name_to_idx[name]
                joint_positions_rad.append(msg.position[idx])
                if len(msg.velocity) > idx:
                    joint_velocities_rad.append(msg.velocity[idx])
                else:
                    joint_velocities_rad.append(0.0)
            else:
                joint_positions_rad.append(0.0)
                joint_velocities_rad.append(0.0)

        self.joint_positions = np.rad2deg(np.asarray(joint_positions_rad, dtype=np.float32))
        self.joint_velocities = np.rad2deg(np.asarray(joint_velocities_rad, dtype=np.float32))
        self.position_history_deg.append(self.joint_positions.copy())
        self.velocity_history_deg.append(self.joint_velocities.copy())
        self.joint_received = True

    def _get_smoothed_velocity(self):
        if not self.use_vel:
            return np.array([], dtype=np.float32)
        if not self.velocity_history_deg:
            return np.zeros(N_JOINTS, dtype=np.float32)
        history = np.stack(self.velocity_history_deg, axis=0).astype(np.float32)
        smoothed = smooth_data(history, window_size=self.vel_window, polyorder=2)
        return smoothed[-1].astype(np.float32)

    def _build_inputs(self):
        if len(self.position_history_deg) < 3:
            return None, None

        q_t_deg = self.position_history_deg[-1]
        q_tm1_deg = self.position_history_deg[-2]
        q_tm2_deg = self.position_history_deg[-3]

        q_t_rad = np.deg2rad(q_t_deg)
        q_tm1_rad = np.deg2rad(q_tm1_deg)
        q_tm2_rad = np.deg2rad(q_tm2_deg)

        dq_t = (q_t_rad - q_tm1_rad).astype(np.float32)
        dq_tm1 = (q_tm1_rad - q_tm2_rad).astype(np.float32)

        base_parts = [encode_joint_positions_deg(q_t_deg)]
        if self.use_vel:
            base_parts.append(self._get_smoothed_velocity())
        base_input = np.concatenate(base_parts, axis=0).astype(np.float32)

        corr_input = np.concatenate(
            [
                encode_joint_positions_deg(q_tm1_deg),
                encode_joint_positions_deg(q_tm2_deg),
                dq_t,
                dq_tm1,
            ],
            axis=0,
        ).astype(np.float32)
        return base_input, corr_input

    def _publish_vector(self, publishers, values, stamp):
        for i in range(N_SENSORS):
            msg = Range()
            msg.header.stamp = stamp
            msg.range = float(values[i])
            msg.radiation_type = Range.ULTRASOUND
            msg.field_of_view = 0.1
            msg.min_range = 0.0
            msg.max_range = 100000.0
            publishers[i].publish(msg)

    def timer_callback(self):
        if not self.raw_received or not self.joint_received:
            return

        base_input, corr_input = self._build_inputs()
        if base_input is None or corr_input is None:
            return
        current_raw = self.raw_data.copy()

        base_norm = (base_input - self.base_mean) / self.base_std
        corr_norm = (corr_input - self.corr_mean) / self.corr_std

        with torch.no_grad():
            base_tensor = torch.from_numpy(base_norm).unsqueeze(0).to(self.device)
            corr_tensor = torch.from_numpy(corr_norm).unsqueeze(0).to(self.device)
            pred_norm, base_pred_norm, corr_tanh = self.model(base_tensor, corr_tensor)
            pred_norm = pred_norm.cpu().numpy().squeeze(0)
            base_pred_norm = base_pred_norm.cpu().numpy().squeeze(0)
            corr_tanh = corr_tanh.cpu().numpy().squeeze(0)

        pred_sensor = pred_norm * self.y_std + self.y_mean
        base_pred_sensor = base_pred_norm * self.y_std + self.y_mean
        corr_raw = corr_tanh * (self.lambda_norm * self.y_std)
        baseline = (
            np.full(N_SENSORS, HARDWARE_BASELINE, dtype=np.float32)
            if self.use_hardware_baseline
            else self.y_mean
        )
        compensated = current_raw - pred_sensor + baseline

        now = self.get_clock().now()
        timestamp = now.nanoseconds / 1e9
        line = (
            f"{timestamp:.9f} "
            + " ".join(f"{x:.6f}" for x in self.joint_positions) + " "
            + " ".join(f"{x:.6f}" for x in self.joint_velocities) + " "
            + " ".join(f"{x:.6f}" for x in current_raw) + " "
            + " ".join(f"{x:.6f}" for x in compensated) + " "
            + " ".join(f"{x:.6f}" for x in pred_sensor) + " "
            + " ".join(f"{x:.6f}" for x in base_pred_sensor) + " "
            + " ".join(f"{x:.6f}" for x in corr_raw)
            + "\n"
        )
        self.log_file.write(line)
        self.log_file.flush()

        stamp = now.to_msg()
        self._publish_vector(self.comp_pubs, compensated, stamp)
        self._publish_vector(self.pred_pubs, pred_sensor, stamp)
        self._publish_vector(self.base_pubs, base_pred_sensor, stamp)
        self._publish_vector(self.corr_pubs, corr_raw, stamp)

    def destroy_node(self):
        if hasattr(self, "log_file") and self.log_file:
            self.log_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RealtimeInferBaseCorrExplicitNode()

    if hasattr(node, "_model_load_failed"):
        node.destroy_node()
        rclpy.shutdown()
        return

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main(sys.argv)
