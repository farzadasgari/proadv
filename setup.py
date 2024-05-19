from setuptools import setup, find_packages

# Read the contents of the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Setting up
setup(
    name="proadv",
    version="2.1.2",
    author="Farzad Asgari",
    author_email="std_farzad.asgari@alumni.khu.ac.ir",
    description="Process Acoustic Doppler Velocimeter data with advanced despiking and analysis tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/farzadasgari/proadv",
    license_file="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.0",
        "scipy>=1.0",
        "fastkde",
    ],
    keywords=[
        "ProADV",
        "python",
        "signal processing",
        "data processing",
        "acoustic Doppler velocimeter",
        "ADV",
        "Denoising",
        "Despiking",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    project_urls={
        "Download URL": "https://pypi.org/project/proadv/",
        "Source Code": "https://github.com/farzadasgari/proadv",
        "Documentation": "https://proadv.readthedocs.io/en/latest/",
    },
)
