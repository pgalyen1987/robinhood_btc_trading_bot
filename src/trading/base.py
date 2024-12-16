from backtesting import Strategy
import pandas as pd
import numpy as np
from typing import Dict, Any

class BaseStrategy(Strategy):
    """Base strategy class with common functionality"""
    
    def __init__(self, broker, data: pd.DataFrame, params: Dict[str, Any]):
        super().__init__(broker, data, params)
        
        # Update class parameters with optimization values
        for name, value in params.items():
            setattr(self, name, float(value))
            
    def _initialize_data(self, data: pd.DataFrame) -> None:
        """Initialize data and precompute static values"""
        self._data = data 