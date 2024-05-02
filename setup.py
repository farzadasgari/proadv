from setuptools import setup, find_packages

# Setting up
setup(
    name="proadv",
    version="0.1.0",
    author="Farzad Asgari",
    author_email="std_farzad.asgari@alumni.khu.ac.ir",
    packages=find_packages(),
    description="post process the data of acoustic Doppler velocimeter (ADV)",
    long_description='''A package that focuses on developing and validating the 3D-KDE filtration algorithms using 
    Python and open-source languages. Users can upload velocity time-series data and apply the 3D-KDE algorithms''',
    install_requires=[
        "numpy", 
        "pytest",
    ], 
    keywords=["python", "signal processing", "velocity", "Doppler velocimeter", "acoustic"]
)