"""Configuration settings for the trading bot."""

import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# API credentials
API_KEY = os.getenv('ROBINHOOD_API_KEY')
PUBLIC_KEY = os.getenv('ROBINHOOD_PUBLIC_KEY')
PRIVATE_KEY = os.getenv('ROBINHOOD_PRIVATE_KEY')

if not all([API_KEY, PUBLIC_KEY, PRIVATE_KEY]):
    raise ValueError(
        "Missing required API credentials. Please ensure ROBINHOOD_API_KEY, "
        "ROBINHOOD_PUBLIC_KEY, and ROBINHOOD_PRIVATE_KEY are set in your .env file."
    )

# Trading parameters
INVESTMENT_AMOUNT = float(os.getenv('INVESTMENT_AMOUNT', '10000'))
TARGET_PROFIT = float(os.getenv('TARGET_PROFIT', '0.005'))  # 0.5%
SYMBOL = os.getenv('TRADING_SYMBOL', 'BTC-USD')

# API endpoints
BASE_URL = os.getenv('API_BASE_URL', 'https://api.robinhood.com').rstrip('/')

# Rate limiting
MAX_REQUESTS_PER_MINUTE = int(os.getenv('MAX_REQUESTS_PER_MINUTE', '30'))
MAX_REQUESTS_PER_HOUR = int(os.getenv('MAX_REQUESTS_PER_HOUR', '1200'))

# Backtesting
INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '10000'))
COMMISSION_RATE = float(os.getenv('COMMISSION_RATE', '0.001'))  # 0.1%

# Optimization
OPTIMIZATION_DAYS = int(os.getenv('OPTIMIZATION_DAYS', '30'))
OPTIMIZATION_INTERVAL = int(os.getenv('OPTIMIZATION_INTERVAL', '24'))  # Hours

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'trading_bot.log')

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Frontend configuration
FRONTEND_API_URL = os.getenv('REACT_APP_API_URL', 'http://localhost:8000')
FRONTEND_WS_URL = os.getenv('REACT_APP_WS_URL', 'ws://localhost:8000/ws')