#!/usr/bin/env python3
"""Script to launch the trading bot with web interface."""

import os
import sys
from datetime import datetime
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from trading_bot import (
    setup_logger,
    DataDownloader,
    MLOptimizer,
    RobinhoodTrader,
    logger,
    Dashboard
)

def load_config(config_path: str = "config.json") -> dict:
    """Load and validate configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required sections
    required_sections = ['trading', 'strategy', 'backtesting', 'optimization']
    missing_sections = [s for s in required_sections if s not in config]
    if missing_sections:
        raise ValueError(f"Missing required config sections: {missing_sections}")
    
    # Validate trading section
    required_trading = ['symbol', 'timeframe', 'initial_capital']
    missing_trading = [k for k in required_trading if k not in config['trading']]
    if missing_trading:
        raise ValueError(f"Missing required trading config: {missing_trading}")
    
    return config

def main():
    # Set up logging
    setup_logger(level="INFO")
    logger.info("Starting Trading Bot with Web Interface...")
    
    try:
        # Load configuration
        config = load_config()
        
        # Create data directories if they don't exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
        
        # Initialize trading bot with dashboard
        try:
            from scripts.run_trading_bot import AutomatedTradingBot
            
            bot = AutomatedTradingBot(
                config_path="config.json",
                optimization_interval=24,  # Optimize strategy daily
                trading_interval=1,  # Check for trades hourly
                lookback_period=30,  # Use 30 days of historical data
                dashboard_port=8050  # Port for the dashboard
            )
            
            # Start the bot
            bot.run()
            
        except ImportError as e:
            logger.error(f"Failed to import AutomatedTradingBot: {e}")
            raise
        except KeyError as e:
            logger.error(f"Missing configuration key: {e}")
            raise
        
        logger.info("Bot initialized successfully")
        logger.info("Dashboard will be available at http://localhost:8050")
        
    except KeyboardInterrupt:
        logger.info("Shutting down bot gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        raise

if __name__ == "__main__":
    main() 