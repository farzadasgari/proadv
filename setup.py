from setuptools import setup, find_packages

# Setting up
setup(
    name="proadv",
    version="1.3.1",
    author="Farzad Asgari",
    author_email="std_farzad.asgari@alumni.khu.ac.ir",
    packages=find_packages(),
    short_description="ProAdv: Process Acoustic Doppler Velocimeter data with advanced despiking and analysis tools",
    long_description="""
    ProAdv is a Python package designed for post-processing acoustic Doppler velocimeter (ADV) datasets. It offers a comprehensive suite of tools for denoising and despiking ADV data, ensuring reliable and accurate measurements.

    **Key Features:**

    * Advanced Despiking Algorithms: Employ robust methods like ACC, PST, mPST, VC, KDE, 3d-KDE, and m3d-KDE to effectively remove noise and outliers.
    * Comprehensive Statistical Functions: Calculate essential statistical measures like minimum, maximum, mean, median, skewness, and kurtosis to gain deeper insights into your data.
    * Additional Analysis Tools: Analyze your data further with moving average calculations, singular spectrum analysis (SSA), and pollution rate (PR) calculation.

    **Benefits:**

    * Improved Data Quality: ProADV's advanced despiking algorithms ensure clean and reliable ADV data, crucial for accurate scientific analysis.
    * Enhanced Statistical Insights: Gain a deeper understanding of your data through comprehensive statistical analysis tools.
    * Streamlined Workflow: ProAdv provides a convenient platform for post-processing and analyzing ADV datasets.
    """,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
    ],
    keywords=[
        "ProADV",
        "python",
        "signal processing",
        "data processing",
        "acoustic Doppler velocimeter",
        "ADV",
        "Despiking",
    ]
)
