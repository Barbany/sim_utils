from setuptools import setup

setup(
    name="sim_utils",
    packages=["sim_utils"],
    version="1.0",
    install_requires=[
        "numpy",
    ],
    extras_require={"dev": ["black", "flake8"]},
)
