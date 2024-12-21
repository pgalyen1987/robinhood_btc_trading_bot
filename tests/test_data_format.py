import pandas as pd
import numpy as np
from src.trading.api import CryptoAPITrading
from backtesting import Backtest
from src.trading.strategy import TradingStrategy
from src.utils.logger import logger
from src.ml.feature_engineering import FeatureEngineer
from src.ml.strategy_optimizer import StrategyOptimizer
from src.ml.common import prepare_data
from src.trading_bot.data_downloader import DataDownloader
from datetime import datetime, timedelta

def get_test_data() -> pd.DataFrame:
    """Get sample data for testing."""
    downloader = DataDownloader(data_dir="data")
    return downloader.get_data(
        filename="test_data.csv",
        days_back=30,  # Only need 30 days for testing
        interval="1h"
    )

def test_data_format():
    """Test data formatting and feature engineering"""
    try:
        # Get sample data
        api = CryptoAPITrading()
        df = api.get_historical_data()
        
        # Test feature engineering
        fe = FeatureEngineer(df)
        df_with_features = fe.add_technical_indicators()
        
        return df_with_features
        
    except Exception as e:
        logger.error(f"Data format test failed: {e}")
        raise

def test_data_validation():
    """Test data validation across different components"""
    try:
        # Get sample data
        downloader = DataDownloader(data_dir="data")
        df = downloader.get_data("test_data.csv")
        
        # Verify data format
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        assert all(col in df.columns for col in required_columns)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert not df.empty
        
        return True
        
    except Exception as e:
        logger.error(f"Data validation test failed: {e}")
        logger.exception("Full traceback:")
        return False

def test_strategy_data_handling():
    """Test strategy data handling and validation"""
    try:
        # Get sample data using DataDownloader
        df = get_test_data()
        
        # Test optimizer data handling
        optimizer = StrategyOptimizer(df)
        params = {
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'position_size': 0.1
        }
        
        # Test single parameter evaluation
        score, stats = optimizer._evaluate_params(params)
        assert score != float('-inf'), "Parameter evaluation failed"
        
        return True
        
    except Exception as e:
        logger.error(f"Strategy data test failed: {e}")
        logger.exception("Full traceback:")
        return False

def test_backtesting_data_handling():
    """Test handling of backtesting._Data type"""
    try:
        # Get sample data using DataDownloader
        downloader = DataDownloader(data_dir="data")
        df = downloader.get_data(
            filename="test_data.csv",
            days_back=30,
            interval="1h"
        )
        
        # Create backtest instance
        bt = Backtest(
            df,
            TradingStrategy,
            cash=100_000
        )
        
        # Get backtesting data
        bt_data = bt._data
        logger.debug(f"Backtesting data type: {type(bt_data)}")
        
        # Test strategy initialization with backtesting data
        strategy = TradingStrategy(
            broker=None,
            data=bt_data,
            params={'rsi_period': 14}
        )
        
        assert hasattr(strategy, '_data'), "Strategy data not initialized"
        assert isinstance(strategy._data, pd.DataFrame), "Strategy data not converted to DataFrame"
        
        return True
        
    except Exception as e:
        logger.error(f"Backtesting data test failed: {e}")
        logger.exception("Full traceback:")
        return False

def test_data_type_handling():
    """Test handling of different data types"""
    try:
        # Get sample data
        api = CryptoAPITrading()
        df = api.get_historical_data()
        
        # Test with DataFrame
        strategy1 = TradingStrategy(
            broker=None,
            data=df,
            params={'rsi_period': 14}
        )
        
        # Test with backtesting data
        bt = Backtest(df, TradingStrategy)
        strategy2 = TradingStrategy(
            broker=None,
            data=bt._data,
            params={'rsi_period': 14}
        )
        
        assert isinstance(strategy1._data, pd.DataFrame)
        assert isinstance(strategy2._data, pd.DataFrame)
        
        return True
        
    except Exception as e:
        logger.error(f"Data type handling test failed: {e}")
        logger.exception("Full traceback:")
        return False

def test_backtesting_data_conversion():
    """Test conversion of backtesting data"""
    try:
        # Reference original data format test
        df = test_data_format()
        
        # Create backtest instance
        bt = Backtest(df, TradingStrategy)
        
        # Initialize strategy with backtesting data
        strategy = TradingStrategy(
            broker=None,
            data=bt._data,
            params={'rsi_period': 14}
        )
        
        # Verify data conversion
        assert isinstance(strategy._data, pd.DataFrame)
        assert all(col in strategy._data.columns 
                  for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
        assert isinstance(strategy._data.index, pd.DatetimeIndex)
        
        return True
        
    except Exception as e:
        logger.error(f"Backtesting data conversion test failed: {e}")
        logger.exception("Full traceback:")
        return False

if __name__ == "__main__":
    try:
        df = test_data_format()
        print("\n✓ All tests completed")
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")