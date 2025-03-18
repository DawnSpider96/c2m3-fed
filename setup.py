from setuptools import setup, find_packages

setup(
    name="c2m3",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "flwr",  # flower
    ],
    python_requires=">=3.10",
)