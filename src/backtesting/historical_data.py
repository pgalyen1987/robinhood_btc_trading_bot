# backtesting_module/historical_data.py
import pandas as pd
import yfinance as yf
from typing import Optional
from utils.logger import logger

class HistoricalDataDownloader:
    """Efficient historical data downloader with caching"""
    
    def __init__(self, 
                 symbol: str = "BTC-USD", 
                 period: str = "1y", 
                 interval: str = "1h"):
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self._cached_data: Optional[pd.DataFrame] = None
        
    def download_data(self) -> pd.DataFrame:
        """Download historical data with caching and error handling"""
        try:
            if self._cached_data is not None:
                return self._cached_data
                
            df = yf.download(
                self.symbol,
                period=self.period,
                interval=self.interval,
                progress=False  # Disable progress bar for efficiency
            )
            
            if df.empty:
                raise ValueError("Downloaded data is empty")
                
            # Optimize data types and structure
            df.index = pd.to_datetime(df.index)
            self._cached_data = df
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading historical data: {e}")
            return pd.DataFrame()