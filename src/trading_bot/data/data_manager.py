"""Data management utilities for trading bot."""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd
import yfinance as yf
from pathlib import Path

from ..utils.logger import logger

class DataManager:
    """Manages market data downloading, caching, and validation."""
    
    def __init__(self, 
                 data_dir: str = "data",
                 symbol: str = "BTC-USD",
                 default_interval: str = "1h"):
        """Initialize data manager.
        
        Args:
            data_dir: Directory to store data files
            symbol: Trading symbol (default: BTC-USD)
            default_interval: Default data interval (default: 1h)
        """
        self.data_dir = Path(data_dir)
        self.symbol = symbol
        self.default_interval = default_interval
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def get_data(self,
                 start_date: Optional[str] = None,
                 end_date: Optional[str] = None,
                 days_back: int = 730,
                 interval: Optional[str] = None,
                 use_cache: bool = True) -> pd.DataFrame:
        """Get market data, either from cache or by downloading.
        
        Args:
            start_date: Start date (format: YYYY-MM-DD)
            end_date: End date (format: YYYY-MM-DD)
            days_back: Number of days to look back if no dates provided
            interval: Data interval (e.g., '1h', '1d')
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with market data
        """
        interval = interval or self.default_interval
        
        # Calculate dates if not provided
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.strptime(end_date, "%Y-%m-%d") - 
                         timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        # Try loading from cache
        cache_file = self._get_cache_path(start_date, end_date, interval)
        if use_cache and cache_file.exists():
            return self._load_cached_data(cache_file)
            
        # Download fresh data
        return self._download_data(start_date, end_date, interval)
    
    def _get_cache_path(self, start_date: str, end_date: str, interval: str) -> Path:
        """Get path for cached data file."""
        filename = f"{self.symbol}_{start_date}_{end_date}_{interval}.csv"
        return self.data_dir / filename
    
    def _load_cached_data(self, cache_path: Path) -> pd.DataFrame:
        """Load data from cache file."""
        try:
            logger.info(f"Loading cached data from {cache_path}")
            data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return self._validate_data(data)
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")
            raise
    
    def _download_data(self, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """Download market data from source."""
        try:
            logger.info(f"Downloading {self.symbol} data from {start_date} to {end_date}")
            data = yf.download(self.symbol, start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            # Validate and cache data
            data = self._validate_data(data)
            self._cache_data(data, start_date, end_date, interval)
            
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise
    
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean market data."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean data
        data = data.copy()
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        data = data.dropna()
        
        # Convert to numeric
        for col in required_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Final validation
        if data.empty:
            raise ValueError("No valid data after cleaning")
            
        return data
    
    def _cache_data(self, data: pd.DataFrame, start_date: str, end_date: str, interval: str) -> None:
        """Cache downloaded data."""
        try:
            cache_path = self._get_cache_path(start_date, end_date, interval)
            data.to_csv(cache_path)
            logger.info(f"Cached data to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
            # Don't raise - caching failure shouldn't stop execution 