"""ML-based strategy optimizer for backtesting."""


from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from dataclasses import dataclass
from datetime import datetime
import joblib
import os

from ..utils.logger import logger
from ..backtesting.backtester import Backtester
from ..trading.strategy import TradingStrategy

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    model: Any
    feature_importance: Dict[str, float]
    score: float
    timestamp: datetime

class MLOptimizer:
    """ML-based strategy optimizer with cross-validation."""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 initial_capital: float = 10000,
                 commission: float = 0.001,
                 n_splits: int = 5,
                 model_dir: str = "models"):
        """Initialize optimizer.
        
        Args:
            data: Historical OHLCV data
            initial_capital: Initial capital for backtesting
            commission: Trading commission rate
            n_splits: Number of cross-validation splits
            model_dir: Directory to save models
        """
        self.data = self._validate_data(data)
        self.initial_capital = initial_capital
        self.commission = commission
        self.n_splits = n_splits
        self.model_dir = model_dir
        self.results: List[OptimizationResult] = []
        
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and prepare input data."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
            
        if len(data) < 100:  # Need enough data for training
            raise ValueError("Insufficient data points for optimization")
            
        return data.copy()
        
    def _prepare_features(self, strategy: TradingStrategy) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML model."""
        # Get features from strategy
        features = strategy.get_features()
        
        # Remove NaN values
        features = features.dropna()
        
        # Calculate returns (target variable)
        returns = features['returns'].shift(-1)
        
        # Create binary classification target (1 for positive returns)
        y = (returns > 0).astype(int)
        
        # Drop target-related columns from features
        X = features.drop(['returns', 'returns_std'], axis=1)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled[:-1], y[:-1]  # Remove last row due to shift
        
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
        """Train ML model for strategy optimization."""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=42
        )
        
        model.fit(X, y)
        return model
        
    def _evaluate_strategy(self, params: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate strategy with given parameters."""
        backtester = Backtester(
            data=data,
            strategy_class=TradingStrategy,
            initial_capital=self.initial_capital,
            commission=self.commission
        )
        
        results = backtester.run(params)
        return results
        
    def optimize(self, param_ranges: Dict[str, Tuple[float, float, int]]) -> OptimizationResult:
        """Run ML-based strategy optimization.
        
        Args:
            param_ranges: Dictionary of parameter ranges to optimize
                Format: {param_name: (min_value, max_value, num_steps)}
                
        Returns:
            Optimization results with best parameters
        """
        try:
            # Create time series cross-validation splits
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            
            best_result = None
            best_score = float('-inf')
            
            # Cross-validation loop
            for fold, (train_idx, test_idx) in enumerate(tscv.split(self.data)):
                logger.info(f"Processing fold {fold + 1}/{self.n_splits}")
                
                # Split data
                train_data = self.data.iloc[train_idx]
                test_data = self.data.iloc[test_idx]
                
                # Initial backtest to get features
                initial_params = TradingStrategy.default_params
                strategy = TradingStrategy(None, train_data, initial_params)
                
                # Prepare features and train model
                X, y = self._prepare_features(strategy)
                model = self._train_model(X, y)
                
                # Get feature importance
                feature_importance = dict(zip(
                    strategy.get_features().columns,
                    model.feature_importances_
                ))
                
                # Generate optimized parameters
                optimized_params = self._generate_params(feature_importance, param_ranges)
                
                # Evaluate optimized strategy
                test_results = self._evaluate_strategy(optimized_params, test_data)
                score = self._calculate_score(test_results)
                
                # Save results
                result = OptimizationResult(
                    parameters=optimized_params,
                    metrics=test_results['metrics'],
                    model=model,
                    feature_importance=feature_importance,
                    score=score,
                    timestamp=datetime.now()
                )
                
                self.results.append(result)
                
                # Update best result
                if score > best_score:
                    best_score = score
                    best_result = result
                    self._save_model(model, optimized_params)
                    
            if not best_result:
                raise ValueError("No valid results found during optimization")
                
            logger.info(f"Optimization complete. Best score: {best_score:.2f}")
            return best_result
            
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            raise
            
    def _generate_params(self, 
                        feature_importance: Dict[str, float],
                        param_ranges: Dict[str, Tuple[float, float, int]]) -> Dict[str, Any]:
        """Generate optimized parameters based on feature importance."""
        params = {}
        
        for param_name, (min_val, max_val, steps) in param_ranges.items():
            # Get related feature importance
            importance = feature_importance.get(param_name, 0.5)  # Default to middle if not found
            
            # Calculate parameter value based on importance
            value = min_val + (max_val - min_val) * importance
            params[param_name] = value
            
        return params
        
    def _calculate_score(self, results: Dict[str, Any]) -> float:
        """Calculate optimization score with constraints."""
        metrics = results['metrics']
        
        # Extract key metrics
        return_pct = float(metrics.get('total_return', 0))
        sharpe = float(metrics.get('sharpe_ratio', 0))
        max_drawdown = float(metrics.get('max_drawdown', 100))
        win_rate = float(metrics.get('win_rate', 0))
        num_trades = int(metrics.get('num_trades', 0))
        
        # Apply constraints
        if num_trades < 10:  # Need enough trades
            return float('-inf')
            
        if max_drawdown > 30:  # Maximum 30% drawdown
            return float('-inf')
            
        if win_rate < 40:  # Minimum 40% win rate
            return float('-inf')
            
        # Calculate weighted score
        score = (
            0.4 * return_pct +           # Returns
            0.3 * sharpe * 100 +         # Risk-adjusted returns
            0.2 * (100 - max_drawdown) + # Capital preservation
            0.1 * win_rate              # Consistency
        )
        
        return float(score)
        
    def _save_model(self, model: Any, params: Dict[str, Any]) -> None:
        """Save trained model and parameters."""
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_path = os.path.join(self.model_dir, f"model_{timestamp}.joblib")
            joblib.dump(model, model_path)
            
            # Save parameters
            params_path = os.path.join(self.model_dir, f"params_{timestamp}.joblib")
            joblib.dump(params, params_path)
            
            logger.info(f"Model and parameters saved to {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise 