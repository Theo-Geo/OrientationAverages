#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open("app/README.md", "r") as f:
    long_description = f.read()
    
setup(
    name="orientation_averages",
    version="0.0.10",
    description="A package used to take averages of crystal orientations",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Theo-Geo/symmetry_average.git",
    author="ArjanCodes",
    author_email="theo.haeussler@gmail.com",
    license="GPL-3.0-or-later",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["numba >= 0.60.0", 
                      "numpy >= 1.26.0"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.9",
)