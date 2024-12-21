#!/usr/bin/env python3
"""Script to download historical BTC data."""

import sys
from datetime import datetime
from trading_bot.utils.data_downloader import DataDownloader
from trading_bot.utils.logger import setup_logger, logger

def main():
    """Download historical BTC data."""
    try:
        # Set up logging
        setup_logger(level="INFO")
        
        # Initialize downloader
        downloader = DataDownloader(symbol="BTC-USD")
        
        # Download data
        start_date = "2023-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Downloading {downloader.symbol} data from {start_date} to {end_date}")
        data = downloader.download_historical_data(
            start_date=start_date,
            end_date=end_date
        )
        
        # Save data
        output_file = "btc_historical.csv"
        downloader.save_data(data, output_file)
        
        logger.info(f"Successfully downloaded {len(data)} rows of data")
        return 0
        
    except Exception as e:
        logger.error(f"Failed to download data: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 