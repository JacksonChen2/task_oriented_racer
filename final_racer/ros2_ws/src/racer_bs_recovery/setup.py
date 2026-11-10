from glob import glob

from setuptools import find_packages, setup


package_name = "racer_bs_recovery"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", ["config/default.yaml"]),
    ],
    install_requires=["setuptools", "numpy"],
    extras_require={"test": ["pytest"]},
    zip_safe=False,
    maintainer="RACER workspace maintainer",
    maintainer_email="maintainer@example.com",
    description="External base-station recovery controller for RACER",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "bs_recovery_supervisor = racer_bs_recovery.supervisor:main",
            "cmd_vel_mux = racer_bs_recovery.cmd_vel_mux:main",
        ],
    },
)
