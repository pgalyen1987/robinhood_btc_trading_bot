"""Data downloading utilities."""

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from ..utils.logger import logger

class DataDownloader:
    """Downloads and manages historical market data."""

    def __init__(self, data_dir: str = "data", symbol: str = "BTC-USD"):
        """Initialize data downloader.
        
        Args:
            data_dir: Directory to store downloaded data.
            symbol: Trading symbol (default: BTC-USD)
        """
        self.data_dir = data_dir
        self.symbol = symbol
        os.makedirs(data_dir, exist_ok=True)

    def download_historical_data(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Download historical market data."""
        try:
            if end_date is None:
                end_date = datetime.now().strftime("%Y-%m-%d")

            cache_path = self._get_cache_path(start_date, end_date, interval)
            
            # Try loading from cache first
            if os.path.exists(cache_path):
                logger.info("Loading data from cache...")
                return self.load_data(os.path.basename(cache_path))

            logger.info(f"Downloading {self.symbol} data from {start_date} to {end_date}")
            data = yf.download(self.symbol, start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {self.symbol}")
                
            # Validate and clean the data
            data = self._validate_data(data)
            
            # Save to cache
            self.save_data(data, os.path.basename(cache_path))
            
            logger.info(f"Downloaded {len(data)} rows of data")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise

    def save_data(self, data: pd.DataFrame, filename: str) -> None:
        """Save downloaded data to CSV file.
        
        Args:
            data: DataFrame containing market data to save
            filename: Output filename for the CSV file
        """
        try:
            file_path = os.path.join(self.data_dir, filename)
            data.to_csv(file_path)
            logger.info(f"Data saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            filename: Input filename.
            
        Returns:
            DataFrame with loaded data.
        """
        try:
            filepath = os.path.join(self.data_dir, filename)
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"Loaded {len(data)} rows from {filepath}")
            return data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise 

    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean downloaded data.
        
        Args:
            data: DataFrame to validate
        
        Returns:
            Validated and cleaned DataFrame
        
        Raises:
            ValueError: If data validation fails
        """
        try:
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Remove any NaN values and ensure proper types
            data = data.dropna()
            data = data.sort_index()
            
            # Ensure datetime index
            data.index = pd.to_datetime(data.index)
            data.index.name = 'Date'
            
            # Convert columns to numeric
            for col in required_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
                
            # Final cleanup
            data = data.dropna()
            
            if data.empty:
                raise ValueError("No valid data remains after cleaning")
            
            return data
            
        except Exception as e:
            logger.error(f"Data validation error: {e}")
            raise 

    def _get_cache_path(self, start_date: str, end_date: str, interval: str) -> str:
        """Get cache file path for given parameters."""
        cache_filename = f"{self.symbol}_{start_date}_{end_date}_{interval}.csv"
        return os.path.join(self.data_dir, cache_filename)

    def _load_cached_data(self, cache_path: str) -> Optional[pd.DataFrame]:
        """Load data from cache if available."""
        try:
            if os.path.exists(cache_path):
                data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                logger.info(f"Loaded cached data from {cache_path}")
                return data
            return None
        except Exception as e:
            logger.warning(f"Failed to load cached data: {e}")
            return None

    def download_latest_data(self, days_back: int = 730, interval: str = "1h", filename: str = "historical_btc.csv") -> pd.DataFrame:
        """Download latest market data for specified period.
        
        Args:
            days_back: Number of days of historical data to download
            interval: Data interval ('1h' or '1d')
            filename: Output filename for saving data
            
        Returns:
            DataFrame containing downloaded data
        """
        try:
            # Calculate date range
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            
            logger.info(f"Downloading {self.symbol} data for last {days_back} days")
            
            # Download data
            data = self.download_historical_data(
                start_date=start_date,
                end_date=end_date,
                interval=interval
            )
            
            # Save to specified file
            if filename:
                self.save_data(data, filename)
                
            return data
            
        except Exception as e:
            logger.error(f"Error downloading latest data: {e}")
            raise

    def get_data(self, filename: str, days_back: int = 730, interval: str = "1h") -> pd.DataFrame:
        """Load data from file or download if not exists.
        
        Args:
            filename: Name of the data file
            days_back: Number of days to download if file doesn't exist
            interval: Data interval for download ('1h' or '1d')
            
        Returns:
            DataFrame containing market data
        """
        try:
            file_path = os.path.join(self.data_dir, filename)
            
            if os.path.exists(file_path):
                logger.info(f"Loading existing data from {filename}")
                data = self.load_data(filename)
                return self._validate_data(data)
            
            logger.info(f"Data file {filename} not found, downloading fresh data")
            data = self.download_latest_data(
                days_back=days_back,
                interval=interval,
                filename=filename
            )
            return self._validate_data(data)
            
        except Exception as e:
            logger.error(f"Error getting data: {e}")
            raise