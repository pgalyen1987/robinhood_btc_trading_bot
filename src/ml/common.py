import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.utils.logger import logger
from src.ml.indicators import calculate_indicators

class CommonTrading:
    """Optimized base class for trading operations"""
    
    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self.indicators: Dict[str, np.ndarray] = {}
        
    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            raise ValueError("Data not initialized")
        return self._data
        
    @data.setter
    def data(self, value: pd.DataFrame) -> None:
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Data must be a pandas DataFrame")
        if 'Close' not in value.columns:
            raise ValueError("DataFrame must contain 'Close' column")
        self._data = value
        
    def calculate_indicators(self) -> bool:
        """Calculate technical indicators for trading"""
        try:
            if self._data is None:
                raise ValueError("Data not initialized")
                
            close_prices = self._data['Close'].values
            self.indicators = calculate_indicators(close_prices, {
                'rsi_period': 14,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9
            })
            
            return bool(self.indicators)
            
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {e}")
            return False