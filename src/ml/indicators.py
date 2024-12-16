import numpy as np
import talib
from typing import Dict, Any
from src.utils.logger import logger

def calculate_indicators(prices: np.ndarray, params: Dict[str, Any] = None) -> Dict[str, np.ndarray]:
    """Calculate technical indicators for the given price data"""
    try:
        if params is None:
            params = {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            }
            
        indicators = {}
        
        # Calculate RSI
        indicators['rsi'] = talib.RSI(prices, timeperiod=params['rsi_period'])
        
        # Calculate MACD
        macd, signal, hist = talib.MACD(
            prices,
            fastperiod=params['macd_fast'],
            slowperiod=params['macd_slow'],
            signalperiod=params['macd_signal']
        )
        
        indicators['macd'] = macd
        indicators['macd_signal'] = signal
        indicators['macd_hist'] = hist
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return {} 