#!/usr/bin/env python3

"""Start script for the trading bot."""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

from src.trading_bot import (
    DataManager,
    TradingStrategy,
    Backtester,
    MLOptimizer,
    BacktestVisualizer,
    DashboardApp,
    setup_logger,
    logger
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trading Bot')
    
    parser.add_argument(
        '--mode',
        choices=['backtest', 'optimize', 'live', 'dashboard'],
        default='backtest',
        help='Operating mode'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=730,
        help='Days of historical data to use'
    )
    
    parser.add_argument(
        '--interval',
        choices=['1m', '5m', '15m', '30m', '1h', '1d'],
        default='1h',
        help='Data interval'
    )
    
    parser.add_argument(
        '--symbol',
        default='BTC-USD',
        help='Trading symbol'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    return parser.parse_args()

def setup_environment():
    """Set up the environment."""
    # Create necessary directories
    for dir_name in ['data', 'logs', 'models', 'plots']:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = f"logs/trading_bot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logger(log_file=log_file)
    
    # Check environment variables
    required_vars = ['ROBINHOOD_API_KEY', 'ROBINHOOD_PRIVATE_KEY', 'ROBINHOOD_PUBLIC_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set them in your .env file")
        sys.exit(1)

def run_backtest(data_manager: DataManager, days: int, interval: str):
    """Run backtesting."""
    try:
        # Get historical data
        data = data_manager.get_data(days_back=days, interval=interval)
        
        # Initialize and run backtester
        backtester = Backtester(data)
        results = backtester.run()
        
        # Visualize results
        visualizer = BacktestVisualizer(data, results)
        trades_plot = visualizer.plot_trades()
        equity_plot = visualizer.plot_equity_curve()
        returns_plot = visualizer.plot_returns_distribution()
        
        # Save plots
        visualizer.save_figure(trades_plot, 'trades.html')
        visualizer.save_figure(equity_plot, 'equity.html')
        visualizer.save_figure(returns_plot, 'returns.html')
        
        logger.info("Backtesting completed successfully")
        logger.info(f"Results saved to plots directory")
        
    except Exception as e:
        logger.error(f"Backtesting error: {e}")
        raise

def run_optimization(data_manager: DataManager, days: int, interval: str):
    """Run strategy optimization."""
    try:
        # Get historical data
        data = data_manager.get_data(days_back=days, interval=interval)
        
        # Define parameter ranges
        param_ranges = {
            'rsi_period': (10, 30, 5),
            'rsi_oversold': (20, 40, 5),
            'rsi_overbought': (60, 80, 5),
            'macd_fast': (8, 20, 4),
            'macd_slow': (20, 40, 5),
            'macd_signal': (5, 15, 3)
        }
        
        # Initialize and run optimizer
        optimizer = MLOptimizer(data)
        results = optimizer.optimize(param_ranges)
        
        logger.info("Optimization completed successfully")
        logger.info("Best parameters:")
        for param, value in results.parameters.items():
            logger.info(f"{param}: {value}")
            
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        raise

def run_live_trading(data_manager: DataManager):
    """Run live trading."""
    try:
        # Initialize dashboard
        dashboard = DashboardApp()
        
        # Start live trading loop
        while True:
            try:
                # Get latest data
                data = data_manager.get_data(days_back=7, interval='1h')
                
                # Update dashboard
                dashboard.update_data(data, {})
                
            except Exception as e:
                logger.error(f"Live trading error: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Live trading initialization error: {e}")
        raise

def main():
    """Main entry point."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Set up environment
        setup_environment()
        
        if args.debug:
            logger.setLevel(logging.DEBUG)
        
        # Initialize data manager
        data_manager = DataManager(symbol=args.symbol)
        
        # Run selected mode
        if args.mode == 'backtest':
            run_backtest(data_manager, args.days, args.interval)
            
        elif args.mode == 'optimize':
            run_optimization(data_manager, args.days, args.interval)
            
        elif args.mode == 'live':
            run_live_trading(data_manager)
            
        elif args.mode == 'dashboard':
            dashboard = DashboardApp()
            dashboard.app.run_server(debug=args.debug)
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 