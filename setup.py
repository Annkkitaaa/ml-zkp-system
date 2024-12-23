# setup.py
from setuptools import setup, find_packages

setup(
    name="ml-zkp-system",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'pycryptodome>=3.15.0',
        'pytest>=7.0.0',
        'tqdm>=4.65.0',
        'matplotlib>=3.5.0',
        'scikit-learn>=1.0.0'
    ]
)