from glob import glob
from setuptools import find_packages, setup


package_name = "camera_calibration"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Carsten Oertel",
    maintainer_email="boizbigd@gmail.com",
    description="Eye-to-hand RealSense calibration using a tag on the Inspire Hand.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "calibrate = camera_calibration.calibration_node:main",
            "auto_calibrate = camera_calibration.auto_calibration:main",
        ],
    },
)
