#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name='highway_irl',
    version='0.1.0',
    python_requires='>=3.7',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'gym==0.19.0',
        'gym-minigrid==1.0.2',
        'pygame==2.1.0',
        'highway-env==1.4'
    ],
)