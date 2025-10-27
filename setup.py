from setuptools import find_packages, setup

setup(
    name="cs512_lab2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fire",
        "numpy",
        "pandas",
    ],
    author="Aruj Bansal",
    author_email="aruj.bansal@duke.edu",
    description="Lab 2: MapReduce",
    python_requires=">=3.10",
)
