#!/usr/bin/env python3
import sys
from src.utils.download_data import download_historical_data
from src.backtesting.backtest_runner import run_optimized_backtest
from src.utils.logger import logger
import logging

def setup_logging():
    """Setup logging to both file and console"""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)

def main():
    """Run the complete trading system pipeline"""
    setup_logging()
    
    try:
        # Step 1: Download historical data
        print("\n=== Downloading Historical Data ===")
        logger.info("Starting historical data download...")
        if not download_historical_data():
            logger.error("Failed to download historical data")
            return False
        logger.info("Historical data download completed successfully")
        
        # Step 2: Run backtest
        print("\n=== Running Backtest Optimization ===")
        logger.info("Starting backtest optimization...")
        validation_stats = run_optimized_backtest()
        logger.info(f"Backtest completed with stats: {validation_stats}")
        
        print("\n=== Trading System Run Complete ===")
        return True
        
    except Exception as e:
        logger.error(f"Error running trading system: {e}")
        print("\n=== Trading System Run Failed ===", file=sys.stderr)
        return False

if __name__ == "__main__":
    main()