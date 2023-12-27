#!/usr/bin/env python3
from setuptools import find_packages, setup

setup(
    name='highway_irl_v2',
    version='0.1.0',
    python_requires='>=3.7,<3.9',
    packages=find_packages(),
    install_requires=[
        'numpy==1.19.1',
        'matplotlib',
        'tensorflow==2.7.4',
        'gym==0.19.0',
        'gym-minigrid==1.0.2',
        'pygame==2.1.0',
        'highway-env==1.4',
        'pandas==1.3.5',
        'ray',
        'tqdm',
        'torch'
    ],
)
