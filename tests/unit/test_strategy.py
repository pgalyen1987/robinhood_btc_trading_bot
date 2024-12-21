"""Unit tests for trading strategy."""

import pytest
import pandas as pd
import numpy as np
from trading_bot import TradingStrategy, BaseStrategy

def test_strategy_initialization(sample_data):
    """Test strategy initialization with sample data."""
    strategy = TradingStrategy(None, sample_data, {})
    assert isinstance(strategy, TradingStrategy)
    assert isinstance(strategy, BaseStrategy)  # Should inherit from BaseStrategy
    assert not strategy.position.is_long
    assert len(strategy.data) == len(sample_data)

def test_strategy_indicators(sample_data):
    """Test technical indicator calculations."""
    strategy = TradingStrategy(None, sample_data, {})
    strategy._initialize_indicators()
    
    # Check RSI calculation
    assert hasattr(strategy, 'rsi')
    assert isinstance(strategy.rsi, np.ndarray)
    
    # Check MACD calculation
    assert hasattr(strategy, 'macd')
    assert isinstance(strategy.macd, np.ndarray)
    assert hasattr(strategy, 'signal')
    assert isinstance(strategy.signal, np.ndarray)

def test_position_sizing(sample_data):
    """Test position size calculation."""
    strategy = TradingStrategy(None, sample_data, {'position_size': 0.1})
    size = strategy._calculate_position_size(price=100.0)
    assert 0 < size <= 1.0  # Position size should be between 0 and 1
    assert size <= 0.1  # Should respect position size parameter

def test_risk_management(sample_data):
    """Test risk management rules."""
    strategy = TradingStrategy(None, sample_data, {
        'stop_loss': 0.02,
        'take_profit': 0.04
    })
    
    # Test stop loss calculation
    entry_price = 100.0
    stop_price = entry_price * (1 - strategy.params['stop_loss'])
    assert stop_price < entry_price
    assert stop_price == 98.0  # 2% below entry
    
    # Test take profit calculation
    take_profit_price = entry_price * (1 + strategy.params['take_profit'])
    assert take_profit_price > entry_price
    assert take_profit_price == 104.0  # 4% above entry