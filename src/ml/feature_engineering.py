from .common import CommonTrading
import pandas as pd
from utils.logger import logger

class FeatureEngineer(CommonTrading):
    def __init__(self, data: pd.DataFrame = None):
        super().__init__()
        if data is not None:
            self.data = data
            
    def add_technical_indicators(self) -> pd.DataFrame:
        """Add technical indicators to the dataframe"""
        try:
            if self.data is None:
                raise ValueError("No data has been set")
                
            self.calculate_indicators()
            
            df = self.data.copy()
            for indicator_name, indicator_values in self.indicators.items():
                df[indicator_name] = indicator_values
                
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            raise