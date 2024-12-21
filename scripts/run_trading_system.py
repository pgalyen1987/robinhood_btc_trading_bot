#!/usr/bin/env python3
import sys
import os
from datetime import datetime, timedelta
import json
import pandas as pd
import logging
from trading_bot import (
    DataDownloader,
    TradingStrategy,
    MLOptimizer,
    BacktestPlotter,
    DataViewer,
    logger,
    setup_logger
)
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def setup_logging():
    """Setup logging configuration."""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)

def download_data() -> pd.DataFrame:
    """Download and validate historical data."""
    try:
        print("\n=== Downloading Historical Data ===")
        logger.info("Starting historical data download...")
        
        downloader = DataDownloader(data_dir="data")
        return downloader.get_data(
            filename="historical_btc.csv",
            days_back=730,
            interval="1h"
        )
            
    except Exception as e:
        logger.error(f"Failed to download historical data: {e}")
        raise

def optimize_strategy(data: pd.DataFrame) -> dict:
    """Optimize trading strategy using ML."""
    try:
        print("\n=== Running Strategy Optimization ===")
        logger.info("Starting strategy optimization...")
        
        # Split data into training and validation sets
        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx]
        valid_data = data.iloc[split_idx:]
        
        # Create and run optimizer
        optimizer = MLOptimizer(
            data=train_data,
            cash=200_000,
            commission=0.002,
            n_splits=5,
            model_dir="models"
        )
        
        best_params = optimizer.optimize(generations=5)
        logger.info(f"Best parameters found: {best_params}")
        
        # Validate on unseen data
        validation_stats = optimizer._evaluate_strategy(best_params, valid_data)
        logger.info("Validation metrics:")
        logger.info(f"Return: {validation_stats['stats'].get('Return [%]', 0):.2f}%")
        logger.info(f"Sharpe Ratio: {validation_stats['stats'].get('Sharpe Ratio', 0):.2f}")
        logger.info(f"Max Drawdown: {validation_stats['stats'].get('Max. Drawdown [%]', 0):.2f}%")
        
        return {
            'parameters': best_params,
            'train_data': train_data,
            'valid_data': valid_data,
            'validation_stats': validation_stats
        }
        
    except Exception as e:
        logger.error(f"Error optimizing strategy: {e}")
        raise

def visualize_results(results: dict):
    """Visualize backtest results."""
    try:
        print("\n=== Generating Visualizations ===")
        
        # Create plots directory
        os.makedirs('plots', exist_ok=True)
        
        # Create backtest plotter
        plotter = BacktestPlotter(
            data=results['valid_data'],
            results=results['validation_stats']
        )
        
        # Plot trades
        trades_fig = plotter.plot_trades(show_indicators=True)
        trades_fig.write_html('plots/trades.html')
        logger.info("Trade visualization saved to plots/trades.html")
        
        # Plot equity curve
        equity_fig = plotter.plot_equity_curve()
        equity_fig.write_html('plots/equity.html')
        logger.info("Equity curve saved to plots/equity.html")
        
        # Plot returns distribution
        returns_fig = plotter.plot_returns_distribution()
        returns_fig.write_html('plots/returns.html')
        logger.info("Returns distribution saved to plots/returns.html")
        
        # Show interactive data viewer
        viewer = DataViewer()
        viewer.update_data(
            data=results['valid_data'],
            trades=results['validation_stats'].get('trades', [])
        )
        viewer.run(host='localhost', port=8050, debug=False)
        
    except Exception as e:
        logger.error(f"Error visualizing results: {e}")
        raise

def save_results(results: dict):
    """Save optimization results."""
    try:
        print("\n=== Saving Results ===")
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f'results/optimization_results_{timestamp}.json'
        
        # Prepare results for saving
        save_data = {
            'parameters': results['parameters'],
            'validation_stats': results['validation_stats']['stats'],
            'timestamp': timestamp
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=4)
            
        logger.info(f"Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        raise

def main():
    """Run the complete trading system pipeline."""
    try:
        # Setup logging
        setup_logging()
        
        # Download and prepare data
        data = download_data()
        
        # Optimize strategy
        results = optimize_strategy(data)
        
        # Visualize results
        visualize_results(results)
        
        # Save results
        save_results(results)
        
        print("\n=== Trading System Run Complete ===")
        return True
        
    except Exception as e:
        logger.error(f"Trading system error: {e}")
        print("\n=== Trading System Run Failed ===", file=sys.stderr)
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)