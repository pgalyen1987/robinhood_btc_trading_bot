#!/usr/bin/env python3
"""Main script for running the automated trading bot."""

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any
import threading

from trading_bot import (
    MLOptimizer,
    DataDownloader,
    RobinhoodTrader,
    logger,
    setup_logger
)
from trading_bot.visualization import DashboardApp

class AutomatedTradingBot:
    """Automated trading bot that combines optimization and live trading."""
    
    def __init__(
        self,
        config_path: str = "config.json",
        optimization_interval: int = 24,  # Hours between strategy optimizations
        trading_interval: int = 1,  # Hours between trading checks
        lookback_period: int = 30,  # Days of historical data to analyze
        dashboard_port: int = 8050
    ):
        """Initialize the trading bot."""
        self.config = self._load_config(config_path)
        self.optimization_interval = optimization_interval
        self.trading_interval = trading_interval
        self.lookback_period = lookback_period
        
        # Initialize components
        self.downloader = DataDownloader(
            data_dir="data",
            symbol="BTC-USD"
        )
        
        self.optimizer = MLOptimizer(
            data=None,  # Will be set during optimization
            initial_capital=self.config['trading']['initial_capital'],
            commission=self.config['backtesting']['commission'],
            n_splits=5,
            model_dir="models"
        )
        
        self.trader = RobinhoodTrader(
            api_key=self.config['robinhood']['api_key'],
            api_secret=self.config['robinhood']['api_secret']
        )
        
        # Initialize dashboard
        self.dashboard = DashboardApp(port=dashboard_port)
        self.dashboard_thread = threading.Thread(target=self.dashboard.run)
        self.dashboard_thread.daemon = True
        
        # Track current strategy and last optimization
        self.current_strategy = None
        self.last_optimization = None
        self.last_trade_check = None
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def optimize_strategy(self) -> Dict[str, Any]:
        """Run strategy optimization and return the best strategy."""
        logger.info("Starting strategy optimization...")
        
        # Download recent historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_period)
        data = self.downloader.download_historical_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            interval=self.config['trading']['timeframe']
        )
        self.optimizer.data = data
        
        # Run optimization
        results = self.optimizer.optimize(self.config['optimization']['param_ranges'])
        
        # Save optimization results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        results_file = os.path.join(results_dir, f"optimization_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results.to_dict(), f, indent=4)
        
        # Update dashboard
        self.dashboard.update_data(data, results.to_dict())
        
        logger.info(f"Optimization complete. Best strategy score: {results.score:.2f}")
        return results.to_dict()
    
    def check_and_trade(self) -> None:
        """Check current market conditions and execute trades if needed."""
        if not self.current_strategy:
            logger.warning("No strategy available. Skipping trade check.")
            return
        
        try:
            # Get current market data
            current_data = self.downloader.get_latest_data()
            
            # Generate trading signal
            signal = self.optimizer.generate_signal(
                self.current_strategy['parameters'],
                current_data
            )
            
            # Update dashboard with latest data
            if hasattr(self, 'dashboard'):
                current_data['signal'] = signal
                self.dashboard.update_data(current_data, self.current_strategy)
            
            # Get current position
            position = self.trader.get_position("BTC")
            current_price = self.trader.get_current_price("BTC")
            
            # Execute trades based on signal
            if signal > 0 and not position:  # Buy signal
                # Calculate position size
                capital = self.trader.get_buying_power()
                position_size = capital * self.current_strategy['parameters']['position_size']
                
                # Place buy order
                self.trader.place_order(
                    symbol="BTC",
                    quantity=position_size / current_price,
                    side="buy"
                )
                logger.info(f"Placed buy order for BTC at {current_price}")
                
            elif signal < 0 and position:  # Sell signal
                # Place sell order
                self.trader.place_order(
                    symbol="BTC",
                    quantity=position['quantity'],
                    side="sell"
                )
                logger.info(f"Placed sell order for BTC at {current_price}")
            
            # Update stop loss and take profit orders
            if position:
                self.trader.update_risk_orders(
                    symbol="BTC",
                    current_price=current_price,
                    stop_loss=self.current_strategy['parameters']['stop_loss'],
                    take_profit=self.current_strategy['parameters']['take_profit']
                )
        
        except Exception as e:
            logger.error(f"Error during trade check: {e}")
    
    def run(self) -> None:
        """Run the trading bot continuously."""
        try:
            logger.info("Starting automated trading bot...")
            
            # Start dashboard in separate thread
            self.dashboard_thread.start()
            logger.info(f"Dashboard started at http://localhost:{self.dashboard.port}")
            
            while True:
                try:
                    current_time = datetime.now()
                    
                    # Check if we need to optimize strategy
                    if (not self.last_optimization or 
                        (current_time - self.last_optimization).total_seconds() >= self.optimization_interval * 3600):
                        self.current_strategy = self.optimize_strategy()
                        self.last_optimization = current_time
                    
                    # Check if we need to evaluate trades
                    if (not self.last_trade_check or 
                        (current_time - self.last_trade_check).total_seconds() >= self.trading_interval * 3600):
                        self.check_and_trade()
                        self.last_trade_check = current_time
                    
                    # Sleep for a minute before next check
                    time.sleep(60)
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    logger.exception("Full traceback:")
                    time.sleep(60)  # Wait before retrying
                
        except KeyboardInterrupt:
            logger.info("Shutting down trading bot...")
        except Exception as e:
            logger.error(f"Fatal error in trading bot: {e}")
            logger.exception("Full traceback:")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources before shutting down."""
        logger.info("Shutting down trading bot...")
        self.dashboard_thread.join()
        logger.info("Trading bot shut down successfully.")

def main():
    """Run the automated trading bot."""
    # Set up logging
    setup_logger(level="INFO")
    
    try:
        # Create and run the trading bot
        bot = AutomatedTradingBot(
            config_path="config.json",
            optimization_interval=24,  # Optimize strategy daily
            trading_interval=1,  # Check for trades hourly
            lookback_period=30,  # Use 30 days of historical data
            dashboard_port=8050  # Port for the dashboard
        )
        bot.run()
        
    except Exception as e:
        logger.error(f"Bot execution failed: {e}")
        raise

if __name__ == "__main__":
    main() 