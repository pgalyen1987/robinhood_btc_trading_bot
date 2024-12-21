"""Parameter optimization for trading strategies."""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from itertools import product
from .backtester import Backtester
from ..utils.logger import logger

class StrategyOptimizer:
    """Optimizer for trading strategy parameters."""
    
    def __init__(self, backtester: Backtester):
        """Initialize optimizer with backtester."""
        self.backtester = backtester
        
    def generate_parameter_combinations(self, param_ranges: Dict[str, Tuple[float, float, int]]) -> Dict[str, Tuple]:
        """Generate parameter combinations for optimization.
        
        Args:
            param_ranges: Dictionary of parameter ranges with format:
                {param_name: (start, end, num_points)}
        
        Returns:
            Dictionary of parameter combinations suitable for backtesting.
        """
        param_combinations = {}
        
        for param_name, (start, end, num_points) in param_ranges.items():
            if num_points <= 1:
                param_combinations[param_name] = (start,)
            else:
                values = np.linspace(start, end, num_points)
                param_combinations[param_name] = tuple(values)
                
        return param_combinations
        
    def optimize(self, param_ranges: Dict[str, Tuple[float, float, int]], **kwargs) -> Dict[str, Any]:
        """Run optimization with parameter ranges.
        
        Args:
            param_ranges: Dictionary of parameter ranges.
            **kwargs: Additional arguments for backtesting.
            
        Returns:
            Dictionary containing optimization results.
        """
        try:
            # Generate parameter combinations
            param_combinations = self.generate_parameter_combinations(param_ranges)
            
            # Run optimization
            results = self.backtester.run(param_combinations, **kwargs)
            
            # Process and return results
            return self._process_optimization_results(results)
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            raise
            
    def _process_optimization_results(self, results: Any) -> Dict[str, Any]:
        """Process optimization results.
        
        Args:
            results: Raw optimization results.
            
        Returns:
            Processed results with best parameters and performance metrics.
        """
        if not hasattr(results, 'optimize'):
            return {'error': 'No optimization results available'}
            
        # Get best parameters
        best_params = results._strategy._params
        
        # Extract performance metrics
        metrics = {
            'total_return': results['Return [%]'],
            'max_drawdown': results['Max. Drawdown [%]'],
            'sharpe_ratio': results['Sharpe Ratio'],
            'win_rate': results['Win Rate [%]'],
            'trades': results['# Trades']
        }
        
        return {
            'best_parameters': best_params,
            'metrics': metrics,
            'full_results': results
        }
        
    def cross_validate(self, param_ranges: Dict[str, Tuple[float, float, int]], 
                      num_folds: int = 5, **kwargs) -> Dict[str, Any]:
        """Perform cross-validation of parameter optimization.
        
        Args:
            param_ranges: Dictionary of parameter ranges.
            num_folds: Number of folds for cross-validation.
            **kwargs: Additional arguments for backtesting.
            
        Returns:
            Cross-validation results.
        """
        try:
            # Split data into folds
            data = self.backtester.data
            fold_size = len(data) // num_folds
            results = []
            
            for i in range(num_folds):
                # Create train/test split
                test_start = i * fold_size
                test_end = (i + 1) * fold_size
                
                train_data = pd.concat([
                    data.iloc[:test_start],
                    data.iloc[test_end:]
                ])
                test_data = data.iloc[test_start:test_end]
                
                # Create new backtester instances
                train_backtester = Backtester(train_data, self.backtester.strategy_class)
                test_backtester = Backtester(test_data, self.backtester.strategy_class)
                
                # Optimize on training data
                train_optimizer = StrategyOptimizer(train_backtester)
                train_results = train_optimizer.optimize(param_ranges, **kwargs)
                
                # Test best parameters on test data
                best_params = train_results['best_parameters']
                test_results = test_backtester.run(best_params)
                
                results.append({
                    'fold': i + 1,
                    'train_metrics': train_results['metrics'],
                    'test_metrics': test_results['metrics'],
                    'best_parameters': best_params
                })
                
            return {
                'fold_results': results,
                'average_test_return': np.mean([r['test_metrics']['total_return'] for r in results]),
                'std_test_return': np.std([r['test_metrics']['total_return'] for r in results])
            }
            
        except Exception as e:
            logger.error(f"Cross-validation error: {e}")
            raise 