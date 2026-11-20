#!/usr/bin/env bash
# Install everything the ROS 2 Humble + MoveIt 2 path needs (Ubuntu 22.04).
set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
  ros-humble-moveit \
  ros-humble-moveit-resources-panda-moveit-config \
  ros-humble-control-msgs \
  ros-humble-robot-state-publisher \
  ros-humble-tf2-ros \
  python3-colcon-common-extensions \
  python3-pip

python3 -m pip install --upgrade "mujoco>=3.1" "anthropic>=2.12" numpy pillow pytest
echo "Done. Next: source /opt/ros/humble/setup.bash && colcon build --symlink-install"
