#!/usr/bin/env python3
import os
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from src.utils.logger import logger

def download_historical_data():
    """Download and save historical BTC data"""
    try:
        # Create the data directory if it doesn't exist
        if not os.path.exists('data'):
            os.makedirs('data')
            logger.info("Created data directory")

        # Download BTC-USD data for the last 2 years
        btc = yf.download(
            'BTC-USD', 
            start=(datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'),
            end=datetime.now().strftime('%Y-%m-%d'),
            interval='1d'
        )

        # Verify data quality
        if btc.empty:
            raise ValueError("Downloaded data is empty")

        # Keep only required columns in correct order
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        btc = btc[required_columns]

        # Clean up the data structure
        btc.columns = btc.columns.get_level_values(0)
        btc.index = pd.to_datetime(btc.index)
        btc.index.name = 'Date'

        # Save with index but without extra headers
        btc.to_csv('data/historical_btc.csv', header=True)
        logger.info(f"First few rows of saved data:\n{btc.head()}")
        
        return True

    except Exception as e:
        logger.error(f"Error downloading historical data: {e}")
        return False

if __name__ == "__main__":
    download_historical_data() 