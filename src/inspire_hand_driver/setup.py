from glob import glob

from setuptools import find_packages, setup

package_name = "inspire_hand_driver"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Carsten Oertel",
    maintainer_email="boizbigd@gmail.com",
    description="ROS 2 driver for the Inspire RH56 dexterous hand over RS485.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "inspire_hand_node = inspire_hand_driver.driver_node:main",
            "inspire_hand_probe = inspire_hand_driver.probe:main",
            "inspire_hand_benchmark = inspire_hand_driver.benchmark:main",
        ],
    },
)
