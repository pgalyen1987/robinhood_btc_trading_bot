# backtesting_module/strategy.py
from src.trading.base import BaseStrategy
import talib
import numpy as np
from src.utils.logger import logger

class TradingStrategy(BaseStrategy):
    """Trading strategy implementation using RSI and MACD indicators"""
    
    # Default strategy parameters
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    position_size = 0.2
    
    def init(self) -> None:
        """Initialize strategy indicators"""
        try:
            close_prices = np.asarray(self.data.Close, dtype=np.float64)
            
            self.rsi = self.I(talib.RSI, close_prices, timeperiod=self.rsi_period)
            macd_all = self.I(talib.MACD, close_prices,
                             fastperiod=self.macd_fast,
                             slowperiod=self.macd_slow,
                             signalperiod=self.macd_signal)
            
            self.macd = macd_all[0]
            self.signal = macd_all[1]
            
        except Exception as e:
            logger.error(f"Failed to initialize strategy: {e}")
            raise
    
    def next(self) -> None:
        """Execute trading logic"""
        try:
            if len(self.data) < max(self.macd_slow, self.rsi_period):
                return
                
            if (np.isnan(self.rsi[-1]) or np.isnan(self.macd[-1]) or 
                np.isnan(self.signal[-1])):
                return
                
            position_size = self.position.size
            
            if position_size == 0:  # No position
                if (self.rsi[-1] < self.rsi_oversold and 
                    self.macd[-1] > self.signal[-1]):
                    # Calculate position size
                    size = (self.equity * float(self.position_size)) / self.data.Close[-1]
                    
                    # Validate size before executing trade
                    if size <= 0:
                        logger.warning("Invalid position size calculated, skipping trade")
                        return
                        
                    if size < 1:  # If less than 1 unit
                        size = 1.0  # Minimum trade size
                    else:
                        size = float(int(size))  # Round down to whole number
                        
                    self.buy(size=size)
                    logger.info(f"Opening long position: {size} units")
                    
            elif position_size > 0:  # Long position
                if (self.rsi[-1] > self.rsi_overbought or 
                    self.macd[-1] < self.signal[-1]):
                    self.position.close()
                    logger.info("Closing long position")
                    
        except Exception as e:
            logger.error(f"Error in trading logic: {e}")
            if self.position.size > 0:
                self.position.close()  # Close position on error