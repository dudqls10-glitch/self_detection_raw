#!/usr/bin/env python3
"""
Launch file for real-time delta-model self-detection compensation.

Usage:
    ros2 launch self_detection_raw realtime_infer_delta.launch.py
    ros2 launch self_detection_raw realtime_infer_delta.launch.py model_file:=/abs/path/to/model.pt
"""

from pathlib import Path
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable


def generate_launch_description():
    package_dir = Path(__file__).resolve().parents[1]
    python_exec = "python3"

    pythonpath = str(package_dir)
    if "PYTHONPATH" in os.environ:
        pythonpath = f"{pythonpath}:{os.environ['PYTHONPATH']}"

    def launch_setup(context):
        model_file = context.launch_configurations.get("model_file", "")
        use_hardware_baseline = context.launch_configurations.get("use_hardware_baseline", "true")
        log_rate = context.launch_configurations.get("log_rate", "100.0")

        cmd = [
            python_exec,
            "-m",
            "self_detection_raw.infer.realtime_infer_delta",
            "--ros-args",
            "-r", "__node:=realtime_infer_delta",
            "-p", f"use_hardware_baseline:={use_hardware_baseline}",
            "-p", f"log_rate:={log_rate}",
        ]

        if model_file and model_file.strip():
            cmd.extend(["-p", f"model_file:={model_file}"])

        return [
            SetEnvironmentVariable("PYTHONPATH", pythonpath),
            ExecuteProcess(cmd=cmd, output="screen"),
        ]

    return LaunchDescription([
        DeclareLaunchArgument(
            "model_file",
            default_value="",
            description="Path to model checkpoint (.pt). If empty, Python DEFAULT_MODEL_FILE is used first, then the latest model is auto-selected.",
        ),
        DeclareLaunchArgument(
            "use_hardware_baseline",
            default_value="true",
            description="Use hardware baseline (4e+7) for compensation",
        ),
        DeclareLaunchArgument(
            "log_rate",
            default_value="100.0",
            description="Realtime logging/inference rate in Hz",
        ),
        OpaqueFunction(function=launch_setup),
    ])
