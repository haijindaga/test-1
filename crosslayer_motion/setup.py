import os
from glob import glob

from setuptools import setup

package_name = "crosslayer_motion"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ueeee327",
    maintainer_email="ueeee327@gmail.com",
    description="Phase 2 motion executor for the cross-layer demo.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "motion_executor = crosslayer_motion.motion_executor:main",
            "interactive = crosslayer_motion.interactive:main",
        ],
    },
)
