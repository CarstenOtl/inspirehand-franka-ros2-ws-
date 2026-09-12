from glob import glob
import os

from setuptools import find_packages, setup

package_name = "inspire_franka_trajectory_replay"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="Carsten Oertel",
    maintainer_email="boizbigd@gmail.com",
    description="Coordinated trajectory replay for a Franka FR3 and Inspire RH56 hand.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "replay_trajectory = inspire_franka_trajectory_replay.replay:main",
            "make_cycle_trajectory = inspire_franka_trajectory_replay.make_cycles:main",
            "capture_demo = inspire_franka_trajectory_replay.capture:main",
            "extract_demo = inspire_franka_trajectory_replay.extract:main",
            "extract_waypoints = inspire_franka_trajectory_replay.waypoints:main",
            "splice_intervention = inspire_franka_trajectory_replay.splice_intervention:main",
        ],
    },
)
