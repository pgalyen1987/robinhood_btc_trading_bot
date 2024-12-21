"""Backtesting implementation for trading strategies."""

from typing import Dict, Any, Type, Optional
import pandas as pd
from backtesting import Backtest
from ..trading.strategy import TradingStrategy
from ..utils.logger import logger

class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, data: pd.DataFrame, strategy_class: Type = TradingStrategy):
        """Initialize backtester with data and strategy."""
        self.data = self._prepare_data(data)
        self.strategy_class = strategy_class
        
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for backtesting."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
            
        # Ensure data is sorted by time
        data = data.sort_index()
        
        # Remove any NaN values
        data = data.dropna()
        
        return data
        
    def run(self, params: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Run backtest with given parameters."""
        try:
            if params is None:
                params = {}
                
            # Create and run backtest
            bt = Backtest(
                self.data,
                self.strategy_class,
                cash=10000,  # Initial cash
                commission=0.001,  # 0.1% commission
                margin=1.0,  # No margin
                trade_on_close=False,  # More realistic simulation
                hedging=False,  # No hedging
                exclusive_orders=True  # One order at a time
            )
            
            # Run optimization if multiple parameter combinations
            if any(isinstance(v, tuple) for v in params.values()):
                results = bt.optimize(
                    maximize='Equity Final [$]',
                    method='grid',  # Grid search
                    **params,
                    **kwargs
                )
            else:
                results = bt.run(**params)
                
            return self._process_results(results)
            
        except Exception as e:
            logger.error(f"Backtesting error: {e}")
            raise
            
    def _process_results(self, results: Any) -> Dict[str, Any]:
        """Process backtest results."""
        metrics = {
            'total_return': results['Return [%]'],
            'max_drawdown': results['Max. Drawdown [%]'],
            'sharpe_ratio': results['Sharpe Ratio'],
            'win_rate': results['Win Rate [%]'],
            'trades': results['# Trades'],
            'equity_final': results['Equity Final [$]'],
            'equity_peak': results['Equity Peak [$]']
        }
        
        return {
            'metrics': metrics,
            'trades': results._trades,
            'full_results': results
        } 