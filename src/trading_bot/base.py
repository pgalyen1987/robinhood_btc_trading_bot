from backtesting import Strategy
import pandas as pd
import numpy as np
from typing import Dict, Any, Union
from trading_bot.utils.logger import logger
import sys

def exit_on_error(e: Exception, message: str):
    """Helper function to log error and exit"""
    logger.error(f"FATAL ERROR - {message}: {str(e)}")
    logger.exception("Full traceback:")
    sys.exit(1)

class BaseStrategy(Strategy):
    def __init__(self, broker, data, params: Dict[str, Any]):
        try:
            logger.debug(f"Initializing BaseStrategy with data type: {type(data)}")
            
            # Handle backtesting data arrays
            if hasattr(data, 'Open'):
                # Convert arrays to numpy first to ensure consistent lengths
                try:
                    data_arrays = {
                        'Open': np.asarray(data.Open),
                        'High': np.asarray(data.High),
                        'Low': np.asarray(data.Low),
                        'Close': np.asarray(data.Close),
                        'Volume': np.asarray(data.Volume)
                    }
                except Exception as e:
                    exit_on_error(e, "Failed to convert data arrays")
                
                # Verify all arrays have the same length
                array_lengths = [len(arr) for arr in data_arrays.values()]
                if len(set(array_lengths)) != 1:
                    logger.error(f"FATAL ERROR - Inconsistent array lengths: {array_lengths}")
                    sys.exit(1)
                
                # Create DataFrame with verified arrays
                try:
                    data_copy = pd.DataFrame(
                        data_arrays,
                        index=data.index[:array_lengths[0]]
                    )
                except Exception as e:
                    exit_on_error(e, "Failed to create DataFrame")
            else:
                if not isinstance(data, pd.DataFrame):
                    logger.error(f"FATAL ERROR - Invalid data type: {type(data)}")
                    sys.exit(1)
                data_copy = data.copy(deep=True)
            
            # Initialize parent class
            try:
                super().__init__(broker, data, params)
            except Exception as e:
                exit_on_error(e, "Failed to initialize parent Strategy")
            
            self._data = data_copy
            self._update_params(params)
            
            logger.debug(f"Strategy initialized with data shape: {self._data.shape}")
            
        except Exception as e:
            exit_on_error(e, "Strategy initialization failed")
    
    def _update_params(self, params: Dict[str, Any]) -> None:
        """Update strategy parameters with type conversion"""
        try:
            if not params:
                return
                
            for name, value in params.items():
                try:
                    if isinstance(value, (np.number, float)):
                        setattr(self, name, float(value))
                    elif isinstance(value, (np.integer, int)):
                        setattr(self, name, int(value))
                    else:
                        setattr(self, name, value)
                except Exception as e:
                    exit_on_error(e, f"Failed to set parameter {name}")
                    
        except Exception as e:
            exit_on_error(e, "Parameter update failed")