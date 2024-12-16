from setuptools import setup, find_packages

setup(
    name="trading_bot",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'backtesting',
        'talib',
        'scikit-learn'
    ]
) 