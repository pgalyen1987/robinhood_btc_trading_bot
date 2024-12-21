"""Main script for running the trading bot."""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

from . import (
    TradingStrategy,
    Backtester,
    StrategyOptimizer,
    setup_logger,
    logger,
    DataDownloader,
    __version__
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=f'Cryptocurrency Trading Bot v{__version__}'
    )
    
    parser.add_argument(
        '--mode',
        choices=['backtest', 'optimize', 'live'],
        default='backtest',
        help='Trading mode'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to historical data CSV file'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    return parser.parse_args()

def load_data(file_path: str) -> pd.DataFrame:
    """Load historical data from CSV file."""
    try:
        # Extract directory and filename from path
        data_dir = os.path.dirname(file_path) or "data"
        filename = os.path.basename(file_path)
        
        # Use DataDownloader to load or download data
        downloader = DataDownloader(data_dir=data_dir)
        return downloader.get_data(filename)
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def run_backtest(data: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run backtesting with given data and configuration."""
    try:
        logger.info("Starting backtesting...")
        
        # Initialize backtester
        backtester = Backtester(data, TradingStrategy)
        
        # Run backtest with config parameters
        results = backtester.run(
            params=config['strategy'],
            cash=config['trading']['initial_capital'],
            commission=config['trading']['commission']
        )
        
        logger.info("Backtesting completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Backtesting error: {e}")
        raise

def run_optimization(data: pd.DataFrame, param_ranges: Dict[str, Any]) -> Dict[str, Any]:
    """Run parameter optimization."""
    try:
        backtester = Backtester(data)
        optimizer = StrategyOptimizer(backtester)
        results = optimizer.optimize(param_ranges)
        
        logger.info("Optimization completed successfully")
        logger.info("Best parameters:")
        for param, value in results['best_parameters'].items():
            logger.info(f"{param}: {value}")
            
        return results
        
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise

def run_live_trading(params: Dict[str, Any]) -> None:
    """Run live trading."""
    try:
        # Start live trading loop
        while True:
            try:
                # Update trading data and metrics
                # This is a placeholder - implement actual live trading logic
                pass
                
            except Exception as e:
                logger.error(f"Live trading error: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Live trading initialization error: {e}")
        raise

def main():
    """Main entry point."""
    try:
        # Parse command line arguments
        args = parse_args()
        
        # Set up logging
        log_level = logging.DEBUG if args.debug else logging.INFO
        setup_logger(level=log_level)
        
        # Load historical data if provided
        data = None
        if args.data:
            data = load_data(args.data)
            
        # Load configuration
        params = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'position_size': 0.2
        }
        
        # Run selected mode
        if args.mode == 'backtest':
            if data is None:
                raise ValueError("Historical data required for backtesting")
            results = run_backtest(data, params)
            
        elif args.mode == 'optimize':
            if data is None:
                raise ValueError("Historical data required for optimization")
            param_ranges = {
                'rsi_period': (10, 30, 5),
                'rsi_oversold': (20, 40, 5),
                'rsi_overbought': (60, 80, 5),
                'macd_fast': (8, 20, 4),
                'macd_slow': (20, 40, 5),
                'macd_signal': (5, 15, 3)
            }
            results = run_optimization(data, param_ranges)
            
        elif args.mode == 'live':
            run_live_trading(params)
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 