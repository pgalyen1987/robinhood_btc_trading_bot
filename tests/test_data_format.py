import pandas as pd
import numpy as np
from src.trading.api import CryptoAPITrading
from backtesting import Backtest
from src.trading.strategy import TradingStrategy
from src.utils.logger import logger
from src.ml.feature_engineering import FeatureEngineer

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

if __name__ == "__main__":
    try:
        df = test_data_format()
        print("\n✓ All tests completed")
    except Exception as e:
        print(f"\n❌ Tests failed: {e}")