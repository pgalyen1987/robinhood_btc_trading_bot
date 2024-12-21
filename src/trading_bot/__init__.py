"""Bitcoin Trading Bot with ML-driven strategy optimization."""

from importlib.metadata import version

__version__ = version("trading_bot")

# Core components
from .data.data_manager import DataManager
from .trading.strategy import TradingStrategy
from .backtesting.backtester import Backtester

# Optimization
from .optimization.optimizer import BaseOptimizer, MLOptimizer

# Visualization
from .visualization.visualizer import Visualizer, BacktestVisualizer, DashboardApp

# Utilities
from .utils.logger import setup_logger, logger

__all__ = [
    # Core
    'DataManager',
    'TradingStrategy',
    'Backtester',
    
    # Optimization
    'BaseOptimizer',
    'MLOptimizer',
    
    # Visualization
    'Visualizer',
    'BacktestVisualizer',
    'DashboardApp',
    
    # Utilities
    'setup_logger',
    'logger',
    
    # Version
    '__version__',
] 