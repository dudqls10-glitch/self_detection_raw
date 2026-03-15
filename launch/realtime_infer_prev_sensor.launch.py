#!/usr/bin/env python3
"""
Launch file for real-time previous-sensor inference.

Usage:
    ros2 launch self_detection_raw realtime_infer_prev_sensor.launch.py
    ros2 launch self_detection_raw realtime_infer_prev_sensor.launch.py \
        model_file:=/abs/path/to/model.pt
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    model_file = LaunchConfiguration("model_file")
    use_hardware_baseline = LaunchConfiguration("use_hardware_baseline")
    log_rate = LaunchConfiguration("log_rate")

    return LaunchDescription([
        DeclareLaunchArgument(
            "model_file",
            default_value="",
            description="Path to model checkpoint (.pt). Empty means auto-select latest compatible model.",
        ),
        DeclareLaunchArgument(
            "use_hardware_baseline",
            default_value="true",
            description="Use hardware baseline (4e+7) instead of training target mean.",
        ),
        DeclareLaunchArgument(
            "log_rate",
            default_value="100.0",
            description="Inference/logging timer rate in Hz.",
        ),
        Node(
            package="self_detection_raw",
            executable="realtime_infer_prev_sensor",
            name="realtime_infer_prev_sensor",
            output="screen",
            parameters=[{
                "model_file": model_file,
                "use_hardware_baseline": use_hardware_baseline,
                "log_rate": log_rate,
            }],
        ),
    ])
