import os
from setuptools import setup
import sys
import versioneer

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    author="Rodrigo Tenorio, Luana Modafferi, David Keitel, Alicia M. Sintes",
    author_email="rodrigo.tenorio@ligo.org",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    cmdclass=versioneer.get_cmdclass(),
    description=(
        "Empirically estimating the distribution "
        "of the loudest candidate from a gravitational-wave search"
    ),
    extras_require={
        "examples": [
            "matplotlib",
        ],
        "tests": ["pytest"],
    },
    install_requires=[
        "numpy",
        "scipy",
        "scikit-image",
        "tqdm",
    ],
    license="LICENSE.md",
    long_description=long_description,
    long_description_content_type="text/markdown",
    maintainer="Rodrigo Tenorio",
    maintainer_email="rodrigo.tenorio@ligo.org",
    name="distromax",
    packages=["distromax"],
    project_urls={
        "Issue tracker": "https://github.com/Rodrigo-Tenorio/distromax/issues",
    },
    python_requires=">=3.7.0, <3.10",
    url="https://github.com/Rodrigo-Tenorio/distromax",
    version=versioneer.get_version(),
)
