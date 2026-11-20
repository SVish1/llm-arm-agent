"""Bring up the MuJoCo Panda sim, MoveIt 2 move_group, and (optionally) RViz.

    ros2 launch llm_arm_agent sim_moveit.launch.py [rviz:=true] [viewer:=true]

The sim node stands in for the robot driver: it publishes /joint_states and
serves the FollowJointTrajectory / GripperCommand actions that the
moveit_resources Panda config's controller manager expects.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    rviz = LaunchConfiguration("rviz")
    viewer = LaunchConfiguration("viewer")

    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(file_path="config/panda.urdf.xacro")
        .robot_description_semantic(file_path="config/panda.srdf")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )
    overrides = {
        # The sim tracks trajectories closely, but allow for timing jitter at real time.
        "trajectory_execution.allowed_start_tolerance": 0.05,
        "trajectory_execution.allowed_execution_duration_scaling": 2.0,
        "trajectory_execution.allowed_goal_duration_margin": 1.0,
        "publish_robot_description_semantic": True,
    }

    sim = Node(
        package="llm_arm_agent",
        executable="sim_node",
        name="mujoco_sim",
        output="screen",
        parameters=[{"viewer": viewer}],
    )
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[moveit_config.to_dict(), overrides],
    )
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="log",
        parameters=[moveit_config.robot_description],
    )
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        output="log",
        arguments=["--frame-id", "world", "--child-frame-id", "panda_link0"],
    )
    rviz_config = os.path.join(
        get_package_share_directory("moveit_resources_panda_moveit_config"), "launch", "moveit.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="log",
        arguments=["-d", rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.planning_pipelines,
            moveit_config.robot_description_kinematics,
        ],
        condition=IfCondition(rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument("viewer", default_value="false", description="open the MuJoCo viewer"),
        sim, static_tf, robot_state_publisher, move_group, rviz_node,
    ])
