from glob import glob

from setuptools import find_packages, setup

package_name = "llm_arm_agent"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={"llm_arm_agent.sim": ["assets/*.xml"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "mujoco>=3.1", "numpy", "anthropic>=2.12"],
    zip_safe=False,
    maintainer="Shamita Nalamutt",
    description="LLM agent for ROS 2 / MoveIt 2 arm control with a MuJoCo benchmark",
    license="MIT",
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "llm-arm = llm_arm_agent.cli:main",
            "sim_node = llm_arm_agent.ros.sim_node:main",
        ],
    },
)
