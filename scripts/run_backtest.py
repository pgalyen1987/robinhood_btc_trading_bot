# run_backtest.py
from src.ml.strategy_optimizer import StrategyOptimizer
from src.utils.logger import logger
from src.trading.strategy import TradingStrategy
from backtesting import Backtest
import pandas as pd

def run_optimized_backtest():
    # Load historical data
    historical_data = pd.read_csv('data/historical_btc.csv')
    logger.info(f"Loaded columns: {historical_data.columns}")
    
    # Convert the first column to datetime index
    historical_data['Date'] = pd.to_datetime(historical_data.iloc[:, 0])
    historical_data.set_index('Date', inplace=True)
    
    # Verify required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in historical_data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Calculate the midpoint date
    mid_point = len(historical_data) // 2
    split_date = historical_data.index[mid_point]
    
    # Split into training and validation periods
    train_data = historical_data.iloc[:mid_point]
    valid_data = historical_data.iloc[mid_point:]
    
    # Optimize strategy
    optimizer = StrategyOptimizer(train_data)
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
    
    validation_stats = bt.run(**best_params)
    logger.info(f"Validation metrics: {validation_stats}")
    return validation_stats
    
if __name__ == "__main__":
    run_optimized_backtest()