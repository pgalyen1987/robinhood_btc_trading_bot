"""Unit tests for ML strategy optimizer."""

import pytest
import pandas as pd
import numpy as np
from trading_bot.ml.strategy_optimizer import MLOptimizer

def test_optimizer_initialization(sample_data):
    """Test optimizer initialization."""
    optimizer = MLOptimizer(sample_data)
    assert isinstance(optimizer, MLOptimizer)
    assert len(optimizer.data) == len(sample_data)
    assert optimizer.initial_capital == 10000
    assert optimizer.commission == 0.001

def test_data_validation(sample_data):
    """Test data validation."""
    # Valid data
    optimizer = MLOptimizer(sample_data)
    assert len(optimizer.data) == len(sample_data)
    
    # Invalid data - missing columns
    invalid_data = pd.DataFrame({
        'Close': [100, 101, 102],
        'Volume': [1000, 1100, 1200]
    })
    with pytest.raises(ValueError, match="Data must contain columns"):
        MLOptimizer(invalid_data)
        
    # Invalid data - too few points
    short_data = sample_data.iloc[:2]
    with pytest.raises(ValueError, match="Insufficient data points"):
        MLOptimizer(short_data)

def test_feature_preparation(sample_data):
    """Test feature preparation."""
    optimizer = MLOptimizer(sample_data)
    strategy = optimizer._evaluate_strategy({}, sample_data)['strategy']
    X, y = optimizer._prepare_features(strategy)
    
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y)
    assert y.dtype == np.int64  # Binary classification

def test_parameter_generation(sample_data):
    """Test parameter generation from feature importance."""
    optimizer = MLOptimizer(sample_data)
    
    # Mock feature importance
    feature_importance = {
        'rsi_period': 0.8,
        'macd_fast': 0.6
    }
    
    # Parameter ranges
    param_ranges = {
        'rsi_period': (10, 30, 5),
        'macd_fast': (8, 20, 4)
    }
    
    params = optimizer._generate_params(feature_importance, param_ranges)
    
    assert 'rsi_period' in params
    assert 'macd_fast' in params
    assert 10 <= params['rsi_period'] <= 30
    assert 8 <= params['macd_fast'] <= 20

def test_optimization_score(sample_data):
    """Test optimization score calculation."""
    optimizer = MLOptimizer(sample_data)
    
    # Good results
    good_results = {
        'metrics': {
            'total_return': 50.0,
            'sharpe_ratio': 2.5,
            'max_drawdown': 15.0,
            'win_rate': 60.0,
            'num_trades': 20
        }
    }
    good_score = optimizer._calculate_score(good_results)
    assert good_score > 0
    
    # Bad results - too few trades
    bad_results = {
        'metrics': {
            'total_return': 50.0,
            'sharpe_ratio': 2.5,
            'max_drawdown': 15.0,
            'win_rate': 60.0,
            'num_trades': 5  # Below minimum
        }
    }
    bad_score = optimizer._calculate_score(bad_results)
    assert bad_score == float('-inf')
    
    # Bad results - high drawdown
    bad_results['metrics']['num_trades'] = 20
    bad_results['metrics']['max_drawdown'] = 35.0  # Above maximum
    bad_score = optimizer._calculate_score(bad_results)
    assert bad_score == float('-inf') 