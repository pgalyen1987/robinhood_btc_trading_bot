from ..base import BaseStrategy
import talib
import numpy as np
from ..utils.logger import logger
import json
import os
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal, ROUND_DOWN
import time
from datetime import datetime, timedelta
from backtesting import Strategy
import sys

class TradingStrategy(BaseStrategy):
    """Cryptocurrency trading strategy with realistic limit order simulation"""
    
    # Strategy parameters
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    position_size = 0.2
    
    # Crypto-specific constants
    MIN_TRADE_VALUE = 10.0
    MIN_PRICE_INCREMENT = 0.01
    MIN_QUANTITY_INCREMENT = 0.00000001
    
    # Rate limiting constants
    MAX_ORDERS_PER_MINUTE = 5
    MAX_ORDERS_PER_HOUR = 100
    RATE_LIMIT_SLEEP = 12.1  # seconds between orders
    
    # Additional limit order settings
    LIMIT_ORDER_TIMEOUT = 300  # 5 minutes
    PRICE_ADJUSTMENT_THRESHOLD = 0.005  # 0.5% price movement
    MAX_ORDER_ATTEMPTS = 3
    FILL_CHECK_INTERVAL = 60  # Check fill status every minute
