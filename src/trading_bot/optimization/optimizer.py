"""Strategy optimization utilities."""

from typing import Dict, Any, List, Optional, Tuple, Type
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

from ..utils.logger import logger
from ..trading.strategy import TradingStrategy
from ..backtesting.backtester import Backtester

@dataclass
class OptimizationResult:
    """Container for optimization results."""
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    score: float
    timestamp: datetime
    model: Optional[Any] = None
    feature_importance: Optional[Dict[str, float]] = None

class BaseOptimizer:
    """Base class for strategy optimization."""
    
    def __init__(self,
                 data: pd.DataFrame,
                 strategy_class: Type[TradingStrategy] = TradingStrategy,
                 initial_capital: float = 10000,
                 commission: float = 0.001):
        """Initialize optimizer.
        
        Args:
            data: Historical market data
            strategy_class: Trading strategy class to optimize
            initial_capital: Initial capital for backtesting
            commission: Trading commission rate
        """
        self.data = self._validate_data(data)
        self.strategy_class = strategy_class
        self.initial_capital = initial_capital
        self.commission = commission
        self.results: List[OptimizationResult] = []
        
    def _validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate input data."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
            
        if len(data) < 100:
            raise ValueError("Insufficient data points for optimization")
            
        return data.copy()
    
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
        
        return score
    
    def _evaluate_strategy(self, params: Dict[str, Any], data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Evaluate strategy with given parameters."""
        data = data if data is not None else self.data
        
        backtester = Backtester(
            data=data,
            strategy_class=self.strategy_class,
            initial_capital=self.initial_capital,
            commission=self.commission
        )
        
        return backtester.run(params)
    
    def optimize(self, param_ranges: Dict[str, Tuple[float, float, int]], **kwargs) -> OptimizationResult:
        """Run optimization with parameter ranges.
        
        Args:
            param_ranges: Dictionary of parameter ranges
            **kwargs: Additional optimization arguments
            
        Returns:
            Optimization results
        """
        raise NotImplementedError("Subclasses must implement optimize method")

class MLOptimizer(BaseOptimizer):
    """ML-based strategy optimizer."""
    
    def __init__(self,
                 data: pd.DataFrame,
                 strategy_class: Type[TradingStrategy] = TradingStrategy,
                 initial_capital: float = 10000,
                 commission: float = 0.001,
                 n_splits: int = 5,
                 model_dir: str = "models"):
        """Initialize ML optimizer.
        
        Args:
            data: Historical market data
            strategy_class: Trading strategy class to optimize
            initial_capital: Initial capital for backtesting
            commission: Trading commission rate
            n_splits: Number of cross-validation splits
            model_dir: Directory to save trained models
        """
        super().__init__(data, strategy_class, initial_capital, commission)
        self.n_splits = n_splits
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
    def _prepare_features(self, strategy: TradingStrategy) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for ML model."""
        features = strategy.get_features()
        features = features.dropna()
        
        # Calculate returns (target variable)
        returns = features['returns'].shift(-1)
        y = (returns > 0).astype(int)
        
        # Prepare feature matrix
        X = features.drop(['returns', 'returns_std'], axis=1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled[:-1], y[:-1]
    
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
    
    def _get_feature_importance(self, model: RandomForestClassifier, feature_names: List[str]) -> Dict[str, float]:
        """Get feature importance from trained model."""
        importance = model.feature_importances_
        return dict(zip(feature_names, importance))
    
    def optimize(self, param_ranges: Dict[str, Tuple[float, float, int]], **kwargs) -> OptimizationResult:
        """Run ML-based optimization."""
        try:
            # Split data for time series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            best_score = float('-inf')
            best_result = None
            
            for train_idx, test_idx in tscv.split(self.data):
                train_data = self.data.iloc[train_idx]
                test_data = self.data.iloc[test_idx]
                
                # Initialize strategy with default parameters
                strategy = self.strategy_class(train_data)
                
                # Prepare features and train model
                X, y = self._prepare_features(strategy)
                model = self._train_model(X, y)
                
                # Get feature importance
                feature_importance = self._get_feature_importance(
                    model,
                    list(strategy.get_features().drop(['returns', 'returns_std'], axis=1).columns)
                )
                
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
                        param_ranges: Dict[str, Tuple[float, float, int]]) -> Dict[str, float]:
        """Generate optimized parameters based on feature importance."""
        params = {}
        
        for param_name, (min_val, max_val, _) in param_ranges.items():
            importance = feature_importance.get(param_name, 0.5)
            value = min_val + (max_val - min_val) * importance
            params[param_name] = value
            
        return params
    
    def _save_model(self, model: Any, params: Dict[str, Any]) -> None:
        """Save trained model and parameters."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_path = self.model_dir / f"model_{timestamp}.joblib"
            joblib.dump(model, model_path)
            
            # Save parameters
            params_path = self.model_dir / f"params_{timestamp}.joblib"
            joblib.dump(params, params_path)
            
            logger.info(f"Model and parameters saved to {self.model_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")
            # Don't raise - saving failure shouldn't stop optimization 