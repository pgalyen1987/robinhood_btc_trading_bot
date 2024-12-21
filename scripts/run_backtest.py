# run_backtest.py
from trading_bot import (
    MLOptimizer,
    logger,
    TradingStrategy
)
from backtesting import Backtest
import pandas as pd
import sys
from trading_bot.utils.data_downloader import DataDownloader

def run_optimized_backtest():
    # Load and validate historical data
    downloader = DataDownloader(data_dir="data")
    historical_data = downloader.get_data("historical_btc.csv")
    logger.info(f"Loaded {len(historical_data)} rows of data")
    
    # Calculate the midpoint date
    mid_point = len(historical_data) // 2
    split_date = historical_data.index[mid_point]
    
    # Split into training and validation periods
    train_data = historical_data.iloc[:mid_point]
    valid_data = historical_data.iloc[mid_point:]
    
    # Optimize strategy
    optimizer = MLOptimizer(train_data)
    best_params = optimizer.optimize(generations=5)
    
    logger.info(f"Best parameters found: {best_params}")
    logger.info(f"Training metrics: {optimizer.best_metrics['stats']}")
    
    # Validate on unseen data
    bt = Backtest(
        valid_data,
        TradingStrategy,
        cash=200_000,
        commission=0.002,
        exclusive_orders=True,
        trade_on_close=True
    )
    
    validation_stats = bt.run(best_params)
    
    logger.info("Validation Results:")
    logger.info(f"Return: {validation_stats['Return [%]']:.2f}%")
    logger.info(f"Sharpe Ratio: {validation_stats['Sharpe Ratio']:.2f}")
    logger.info(f"Max Drawdown: {validation_stats['Max. Drawdown [%]']:.2f}%")
    
    return {
        'train_metrics': optimizer.best_metrics['stats'],
        'validation_metrics': validation_stats,
        'parameters': best_params
    }

if __name__ == "__main__":
    try:
        results = run_optimized_backtest()
        logger.info("Backtest completed successfully")
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)