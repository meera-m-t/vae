""" Setup
"""
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

setup(
    name="vae",
    version=__version__,
    description="Mask segmentation experiments with U-NET.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Sameerah Talafha",
    author_email="sameerah@vectech.io",
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="",
    packages=["VAE"],
    include_package_data=True,
    install_requires=["torch >= 1.4", "torchvision"],
    python_requires=">=3.8",
    entry_points={"console_scripts": ["vae = VAE.__main__:run"]},
    )