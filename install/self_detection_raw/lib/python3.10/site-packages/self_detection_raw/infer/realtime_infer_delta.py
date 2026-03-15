#!/usr/bin/env python3
"""
Real-time inference node for delta-target self-detection models.

The model predicts:
    delta_sensor_t = sensor_t - sensor_{t-1}

The node reconstructs the absolute prediction as:
    pred_sensor_t = sensor_{t-1} + pred_delta_sensor_t

If previous sensor inputs are enabled in the checkpoint, this node uses only
past sensor values from an internal history buffer. The current sensor values
are never fed back as model inputs for the same step.
"""

import glob
import os
import sys
from collections import deque
from datetime import datetime

import numpy as np
import rclpy
import torch
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState, Range

from self_detection_raw.models.mlp_b_v import ModelBV

N_JOINTS = 6
N_SENSORS = 8
HARDWARE_BASELINE = 4.0e+07
CHANNEL_NAMES = [f"raw{i}" for i in range(1, N_SENSORS + 1)]

# ---------------------------------------------------------------------------
# Default model selection (edit here in Python if you want to pin a model)
# ---------------------------------------------------------------------------
DEFAULT_MODEL_FILE = None
# Example:
# DEFAULT_MODEL_FILE = "/home/song/rb10_Proximity/src/self_detection_raw/scripts/model/mlp_vel_delta_0312_211611/model.pt"
# ---------------------------------------------------------------------------


class RealtimeInferDeltaNode(Node):
    """Real-time compensation node for delta-target models."""

    def __init__(self):
        super().__init__("realtime_infer_delta")

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
        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_vel = True
        self.use_prev_sensor = False
        self.prev_sensor_indices = list(range(N_SENSORS))
        self.prev_sensor_steps = 1
        self.target_type = "delta_sensor"
        self.sensor_history = deque(maxlen=8)

        if not model_file:
            if DEFAULT_MODEL_FILE:
                model_file = DEFAULT_MODEL_FILE
                self.get_logger().info(f"Using DEFAULT_MODEL_FILE from Python: {model_file}")
            else:
                model_file = self._find_latest_model()
                if model_file:
                    self.get_logger().info(
                        f"Auto-selected latest model: {os.path.basename(os.path.dirname(model_file))}"
                    )
            if not model_file:
                self.get_logger().error("model_file parameter is required and no models found")
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
        for i in range(N_SENSORS):
            self.comp_pubs.append(
                self.create_publisher(Range, f"/compensated_raw_distance{i+1}", 10)
            )
            self.pred_pubs.append(
                self.create_publisher(Range, f"/predicted_raw_distance{i+1}", 10)
            )

        log_dir = os.path.expanduser("~/rb10_Proximity/logs")
        os.makedirs(log_dir, exist_ok=True)
        model_name = os.path.basename(os.path.dirname(model_file)) or "unknown"
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"compensated_raw_delta_{model_name}_{now}.txt")
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        header = (
            "# timestamp "
            + " ".join([f"j{i+1}" for i in range(N_JOINTS)]) + " "
            + " ".join([f"jv{i+1}" for i in range(N_JOINTS)]) + " "
            + " ".join([f"raw{i+1}" for i in range(N_SENSORS)]) + " "
            + " ".join([f"comp{i+1}" for i in range(N_SENSORS)]) + " "
            + " ".join([f"pred{i+1}" for i in range(N_SENSORS)]) + "\n"
        )
        self.log_file.write(header)
        self.log_file.flush()

        self.timer = self.create_timer(
            1.0 / log_rate,
            self.timer_callback,
            callback_group=self.cb_group,
        )

        self.get_logger().info("=" * 60)
        self.get_logger().info("Real-time Self Detection Compensation (Delta Model)")
        self.get_logger().info(f"Model Path: {model_file}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(f"Use velocities: {self.use_vel}")
        self.get_logger().info(f"Use previous sensors: {self.use_prev_sensor}")
        self.get_logger().info(f"Previous sensor steps: {self.prev_sensor_steps}")
        self.get_logger().info(
            f"Baseline: {HARDWARE_BASELINE:.2e} (hardware)"
            if self.use_hardware_baseline
            else "Baseline: hardware-disabled fallback still uses previous-sensor reconstruction only"
        )
        self.get_logger().info(f"Log: {self.log_path}")
        self.get_logger().info("=" * 60)

    def _find_latest_model(self):
        return find_available_models()[0] if find_available_models() else None

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

        self.use_vel = bool(model_args.get("use_vel", True))
        self.use_prev_sensor = bool(model_args.get("use_prev_sensor", False))
        self.prev_sensor_indices = [int(idx) for idx in model_args.get("prev_sensor_indices", list(range(N_SENSORS)))]
        self.prev_sensor_steps = int(model_args.get("prev_sensor_steps", 1))
        self.target_type = model_args.get("target_type", "delta_sensor")

        in_dim = checkpoint["model_state_dict"]["trunk.0.weight"].shape[1]
        self.model = ModelBV(
            in_dim=in_dim,
            trunk_hidden=int(model_args.get("hidden", 128)),
            head_hidden=int(model_args.get("head_hidden", 64)),
            out_dim=N_SENSORS,
            dropout=float(model_args.get("dropout", 0.1)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.x_mean = np.asarray(normalization["X_mean"], dtype=np.float32)
        self.x_std = np.asarray(normalization["X_std"], dtype=np.float32)
        self.y_mean = np.asarray(normalization["Y_mean"], dtype=np.float32)
        self.y_std = np.asarray(normalization["Y_std"], dtype=np.float32)

        self.sensor_history = deque(maxlen=max(self.prev_sensor_steps + 2, 8))
        self.get_logger().info(f"Model loaded from: {model_file}")
        self.get_logger().info(f"Input dim: {in_dim}, target type: {self.target_type}")

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
        self.joint_received = True

    def _build_joint_features(self):
        j_pos_rad = np.deg2rad(self.joint_positions)
        sin_j = np.sin(j_pos_rad)
        cos_j = np.cos(j_pos_rad)
        features = [sin_j, cos_j]
        if self.use_vel:
            features.append(self.joint_velocities.astype(np.float32))
        return np.concatenate(features, axis=0).astype(np.float32)

    def _build_prev_sensor_features(self):
        if not self.use_prev_sensor:
            return np.array([], dtype=np.float32)

        if len(self.sensor_history) < self.prev_sensor_steps:
            return None

        pieces = []
        for lag in range(1, self.prev_sensor_steps + 1):
            prev_sensor = self.sensor_history[-lag]
            pieces.append(prev_sensor[self.prev_sensor_indices])
        return np.concatenate(pieces, axis=0).astype(np.float32)

    def timer_callback(self):
        if not self.raw_received or not self.joint_received:
            return

        current_raw = self.raw_data.copy()
        prev_sensor_for_reconstruction = self.sensor_history[-1].copy() if self.sensor_history else None

        if self.use_prev_sensor and len(self.sensor_history) < self.prev_sensor_steps:
            self.sensor_history.append(current_raw)
            return

        joint_features = self._build_joint_features()
        prev_sensor_features = self._build_prev_sensor_features()
        if prev_sensor_features is None:
            self.sensor_history.append(current_raw)
            return

        X = np.concatenate([joint_features, prev_sensor_features], axis=0).astype(np.float32)
        X_norm = (X - self.x_mean) / self.x_std

        with torch.no_grad():
            X_tensor = torch.from_numpy(X_norm).unsqueeze(0).to(self.device)
            pred_norm = self.model(X_tensor).cpu().numpy().squeeze(0)

        model_out = pred_norm * self.y_std + self.y_mean

        if self.target_type == "delta_sensor":
            if prev_sensor_for_reconstruction is None:
                self.sensor_history.append(current_raw)
                return
            pred_sensor = prev_sensor_for_reconstruction + model_out
        else:
            pred_sensor = model_out

        if self.use_hardware_baseline:
            baseline = np.full(N_SENSORS, HARDWARE_BASELINE, dtype=np.float32)
        else:
            baseline = np.full(N_SENSORS, HARDWARE_BASELINE, dtype=np.float32)

        compensated = current_raw - pred_sensor + baseline

        now = self.get_clock().now()
        timestamp = now.nanoseconds / 1e9
        line = (
            f"{timestamp:.9f} "
            + " ".join(f"{x:.6f}" for x in self.joint_positions) + " "
            + " ".join(f"{x:.6f}" for x in self.joint_velocities) + " "
            + " ".join(f"{x:.6f}" for x in current_raw) + " "
            + " ".join(f"{x:.6f}" for x in compensated) + " "
            + " ".join(f"{x:.6f}" for x in pred_sensor)
            + "\n"
        )
        self.log_file.write(line)
        self.log_file.flush()

        for i in range(N_SENSORS):
            msg = Range()
            msg.header.stamp = now.to_msg()
            msg.range = float(compensated[i])
            msg.radiation_type = Range.ULTRASOUND
            msg.field_of_view = 0.1
            msg.min_range = 0.0
            msg.max_range = 100000.0
            self.comp_pubs[i].publish(msg)

        for i in range(N_SENSORS):
            msg = Range()
            msg.header.stamp = now.to_msg()
            msg.range = float(pred_sensor[i])
            msg.radiation_type = Range.ULTRASOUND
            msg.field_of_view = 0.1
            msg.min_range = 0.0
            msg.max_range = 100000.0
            self.pred_pubs[i].publish(msg)

        self.sensor_history.append(current_raw)

    def destroy_node(self):
        if hasattr(self, "log_file") and self.log_file:
            self.log_file.close()
        super().destroy_node()


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
            files = glob.glob(os.path.join(root, "**", "model.pt"), recursive=True)
            model_files.extend(files)

    model_files = list(set(model_files))
    model_files.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    return model_files


def main(args=None):
    rclpy.init(args=args)
    node = RealtimeInferDeltaNode()

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
