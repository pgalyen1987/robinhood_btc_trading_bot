#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def create_directory_structure():
    """Create the new directory structure"""
    directories = [
        'src/trading',
        'src/backtesting',
        'src/ml',
        'src/utils',
        'scripts',
        'tests',
        'data'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def move_files():
    """Move files to their new locations"""
    moves = {
        # Trading related
        'trading_bot.py': 'src/trading/api.py',
        'backtesting_module/strategy.py': 'src/trading/strategy.py',
        
        # Backtesting related
        'backtesting_module/backtest_runner.py': 'src/backtesting/backtest_runner.py',
        'backtesting_module/historical_data.py': 'src/backtesting/historical_data.py',
        
        # ML related
        'ml_module/strategy_optimizer.py': 'src/ml/optimizer.py',
        'ml_module/common.py': 'src/ml/common.py',
        
        # Utils
        'logger.py': 'src/utils/logger.py',
        'config.py': 'src/utils/config.py',
        
        # Scripts
        'download_historical_data.py': 'scripts/download_data.py',
        'run_backtest.py': 'scripts/run_backtest.py',
        'run_trading_system.py': 'scripts/run_trading_system.py',
        
        # Tests
        'test_data_format.py': 'tests/test_data_format.py'
    }
    
    for src, dst in moves.items():
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)

def create_init_files():
    """Create __init__.py files in all directories"""
    for root, dirs, _ in os.walk('.'):
        for d in dirs:
            if d.startswith('__'):
                continue
            init_file = os.path.join(root, d, '__init__.py')
            if not os.path.exists(init_file):
                Path(init_file).touch()

def update_imports():
    """Update import statements in all Python files"""
    replacements = {
        'from ml_module': 'from src.ml',
        'from backtesting_module': 'from src.backtesting',
        'from logger': 'from src.utils.logger',
        'from config': 'from src.utils.config',
        'from trading_bot': 'from src.trading.api',
        'from download_historical_data': 'from scripts.download_data'
    }
    
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r') as f:
                    content = f.read()
                
                for old, new in replacements.items():
                    content = content.replace(old, new)
                
                with open(filepath, 'w') as f:
                    f.write(content)

def cleanup():
    """Remove empty old directories"""
    old_dirs = ['ml_module', 'backtesting_module']
    for d in old_dirs:
        if os.path.exists(d):
            shutil.rmtree(d)

def main():
    try:
        print("Starting project reorganization...")
        create_directory_structure()
        move_files()
        create_init_files()
        update_imports()
        cleanup()
        print("Project reorganization completed successfully")
    except Exception as e:
        print(f"Error during reorganization: {e}")

if __name__ == "__main__":
    main() 