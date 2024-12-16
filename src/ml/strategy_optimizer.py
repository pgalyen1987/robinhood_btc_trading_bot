import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import ParameterGrid
from backtesting import Backtest
from src.utils.logger import logger
from src.trading.strategy import TradingStrategy

class StrategyOptimizer:
    """Optimize trading strategy parameters using grid search"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.best_params = None
        self.best_metrics = None
        
    def generate_parameter_grid(self) -> List[Dict]:
        """Generate grid of parameters to test"""
        param_grid = {
            'rsi_period': range(10, 30, 2),
            'rsi_oversold': range(20, 40, 5),
            'rsi_overbought': range(60, 80, 5),
            'macd_fast': range(8, 16, 2),
            'macd_slow': range(20, 32, 2),
            'macd_signal': range(7, 12, 1),
            'position_size': [0.1, 0.2, 0.3, 0.4]
        }
        return list(ParameterGrid(param_grid))
        
    def evaluate_strategy(self, params: Dict) -> Tuple[float, Dict]:
        """Evaluate a single parameter set"""
        try:
            bt = Backtest(
                self.data,
                TradingStrategy,
                cash=200_000,
                commission=0.002,
                exclusive_orders=True,
                trade_on_close=True
            )
            
            stats = bt.run(**params)
            
            score = (
                stats['Return [%]'] * 0.4 +
                stats['Sharpe Ratio'] * 0.3 +
                (1 - stats['Max. Drawdown [%]'] / 100) * 0.2 +
                stats['Win Rate [%]'] * 0.1
            )
            
            return score, stats
            
        except Exception as e:
            logger.error(f"Error evaluating parameters {params}: {e}")
            return float('-inf'), {}
            
    def optimize(self, generations: int = 5) -> Dict:
        """Run optimization for specified number of generations"""
        try:
            param_grid = self.generate_parameter_grid()
            best_score = float('-inf')
            
            for params in param_grid:
                score, stats = self.evaluate_strategy(params)
                if score > best_score:
                    best_score = score
                    self.best_params = params
                    self.best_metrics = {'score': score, 'stats': stats}
                    
            if self.best_params is None:
                raise ValueError("No valid parameters found during optimization")
                    
            return self.best_params
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise