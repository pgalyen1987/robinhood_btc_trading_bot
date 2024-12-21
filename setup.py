from setuptools import setup, find_packages

setup(
    name="trading_bot",
    version="0.1.0",
    description="A Bitcoin trading bot with ML-driven strategy optimization",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "yfinance>=0.2.0",
        "robin-stocks>=3.0.0",
        "scikit-learn>=1.0.0",
        "tensorflow>=2.12.0",
        "dash>=2.0.0",
        "plotly>=5.0.0",
        "python-dotenv>=0.19.0",
        "requests>=2.26.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "joblib>=1.0.0",
        "optuna>=2.10.0",
        "TA-Lib>=0.4.24",
        "backtesting>=0.3.3"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "mypy>=0.900",
            "flake8>=3.9.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "trading-bot=trading_bot.__main__:main",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
) 