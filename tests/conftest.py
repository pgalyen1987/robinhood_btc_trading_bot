"""Test configuration for the trading bot."""

import os
import sys
import pytest
import pandas as pd
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    return pd.DataFrame({
        'Open': [100.0, 101.0, 102.0],
        'High': [103.0, 104.0, 105.0],
        'Low': [98.0, 99.0, 100.0],
        'Close': [101.0, 102.0, 103.0],
        'Volume': [1000, 1100, 1200]
    }, index=pd.date_range('2023-01-01', periods=3, freq='H'))

@pytest.fixture
def config_path():
    """Get path to test configuration file."""
    return os.path.join(project_root, 'config.json')

@pytest.fixture
def test_data_dir():
    """Get path to test data directory."""
    data_dir = os.path.join(project_root, 'tests', 'data')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir 