#!/usr/bin/env python3
"""Script to run ML strategy optimization."""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from scipy import stats
from typing import Dict, Any, List, Tuple
import itertools

from trading_bot import (
    MLOptimizer,
    DataDownloader,
    logger,
    setup_logger,
    BacktestPlotter
)

def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_parameter_grid(param_ranges: Dict[str, Tuple[float, float, int]]) -> List[Dict[str, float]]:
    """Create grid of parameter combinations."""
    param_values = {}
    for param, (min_val, max_val, steps) in param_ranges.items():
        param_values[param] = np.linspace(min_val, max_val, steps)
    
    # Generate all combinations
    keys = param_values.keys()
    combinations = itertools.product(*param_values.values())
    return [dict(zip(keys, combo)) for combo in combinations]

def calculate_statistical_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistical significance metrics."""
    returns = np.array([r['metrics']['total_return'] for r in results])
    sharpe_ratios = np.array([r['metrics']['sharpe_ratio'] for r in results])
    drawdowns = np.array([r['metrics']['max_drawdown'] for r in results])
    
    # T-tests
    t_stat_returns, p_value_returns = stats.ttest_1samp(returns, 0)
    t_stat_sharpe, p_value_sharpe = stats.ttest_1samp(sharpe_ratios, 0)
    
    # Confidence intervals
    ci_returns = stats.t.interval(0.95, len(returns)-1, loc=np.mean(returns), scale=stats.sem(returns))
    ci_sharpe = stats.t.interval(0.95, len(sharpe_ratios)-1, loc=np.mean(sharpe_ratios), scale=stats.sem(sharpe_ratios))
    
    # Stability metrics
    var_ratio = np.std(returns) / np.mean(returns) if np.mean(returns) != 0 else np.inf
    max_dd_ratio = np.max(drawdowns) / np.mean(returns) if np.mean(returns) != 0 else np.inf
    
    return {
        'statistical_tests': {
            'returns_t_stat': float(t_stat_returns),
            'returns_p_value': float(p_value_returns),
            'sharpe_t_stat': float(t_stat_sharpe),
            'sharpe_p_value': float(p_value_sharpe)
        },
        'confidence_intervals': {
            'returns_ci': (float(ci_returns[0]), float(ci_returns[1])),
            'sharpe_ci': (float(ci_sharpe[0]), float(ci_sharpe[1]))
        },
        'stability_metrics': {
            'returns_var_ratio': float(var_ratio),
            'max_drawdown_ratio': float(max_dd_ratio),
            'returns_skew': float(stats.skew(returns)),
            'returns_kurtosis': float(stats.kurtosis(returns))
        }
    }

def plot_optimization_results(results: List[Dict[str, Any]], param_ranges: Dict[str, Tuple[float, float, int]]) -> None:
    """Create interactive visualization of optimization results."""
    # Create figure with subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Parameter Importance',
            'Performance Distribution',
            'Parameter Sensitivity',
            'Return vs Drawdown',
            'Learning Curves',
            'Feature Correlations',
            'Statistical Significance',
            'Risk Metrics'
        )
    )
    
    # 1. Parameter Importance
    importance_df = pd.DataFrame(results)
    param_importance = importance_df['score'].groupby(importance_df['parameters']).mean()
    fig.add_trace(
        go.Bar(
            x=list(param_importance.index),
            y=param_importance.values,
            name='Parameter Importance'
        ),
        row=1, col=1
    )
    
    # 2. Performance Distribution
    scores = [r['score'] for r in results]
    fig.add_trace(
        go.Histogram(
            x=scores,
            name='Score Distribution',
            nbinsx=30
        ),
        row=1, col=2
    )
    
    # Add KDE curve
    kde_x = np.linspace(min(scores), max(scores), 100)
    kde = stats.gaussian_kde(scores)
    fig.add_trace(
        go.Scatter(
            x=kde_x,
            y=kde(kde_x) * len(scores) * (max(scores) - min(scores)) / 30,
            name='KDE',
            line=dict(color='red')
        ),
        row=1, col=2
    )
    
    # 3. Parameter Sensitivity
    for param in param_ranges:
        values = [r['parameters'][param] for r in results]
        scores = [r['score'] for r in results]
        
        # Add regression line
        z = np.polyfit(values, scores, 1)
        p = np.poly1d(z)
        
        fig.add_trace(
            go.Scatter(
                x=values,
                y=scores,
                mode='markers',
                name=f'{param} (R²={np.corrcoef(values, scores)[0,1]:.2f})'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=sorted(values),
                y=p(sorted(values)),
                mode='lines',
                name=f'{param} trend',
                line=dict(dash='dash')
            ),
            row=2, col=1
        )
    
    # 4. Return vs Drawdown with efficient frontier
    returns = [r['metrics']['total_return'] for r in results]
    drawdowns = [r['metrics']['max_drawdown'] for r in results]
    sharpes = [r['metrics']['sharpe_ratio'] for r in results]
    
    fig.add_trace(
        go.Scatter(
            x=drawdowns,
            y=returns,
            mode='markers',
            marker=dict(
                size=10,
                color=sharpes,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Sharpe Ratio')
            ),
            name='Risk-Return'
        ),
        row=2, col=2
    )
    
    # 5. Learning Curves with confidence bands
    cv_scores = [r.get('cv_scores', []) for r in results]
    if cv_scores[0]:
        mean_scores = np.mean(cv_scores, axis=0)
        std_scores = np.std(cv_scores, axis=0)
        x = list(range(1, len(mean_scores) + 1))
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=mean_scores,
                mode='lines',
                name='Mean CV Score',
                line=dict(color='blue')
            ),
            row=3, col=1
        )
        
        # Add confidence bands
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=list(mean_scores + 2*std_scores) + list(mean_scores - 2*std_scores)[::-1],
                fill='toself',
                fillcolor='rgba(0,0,255,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='95% Confidence'
            ),
            row=3, col=1
        )
    
    # 6. Feature Correlations with clustering
    if results[0].get('feature_importance'):
        feature_imp = pd.DataFrame([r['feature_importance'] for r in results])
        corr_matrix = feature_imp.corr()
        
        # Hierarchical clustering
        import scipy.cluster.hierarchy as sch
        dendro = sch.dendrogram(sch.linkage(corr_matrix, method='ward'))
        idx = dendro['leaves']
        corr_matrix = corr_matrix.iloc[idx, idx]
        
        fig.add_trace(
            go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1
            ),
            row=3, col=2
        )
    
    # 7. Statistical Significance
    stats_metrics = calculate_statistical_metrics(results)
    
    # Create bar chart of t-statistics
    t_stats = stats_metrics['statistical_tests']
    fig.add_trace(
        go.Bar(
            x=['Returns', 'Sharpe Ratio'],
            y=[t_stats['returns_t_stat'], t_stats['sharpe_t_stat']],
            name='T-Statistics'
        ),
        row=4, col=1
    )
    
    # Add significance threshold lines
    fig.add_hline(y=1.96, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=-1.96, line_dash="dash", line_color="red", row=4, col=1)
    
    # 8. Risk Metrics
    stability = stats_metrics['stability_metrics']
    fig.add_trace(
        go.Bar(
            x=['Var Ratio', 'DD Ratio', 'Skew', 'Kurtosis'],
            y=[
                stability['returns_var_ratio'],
                stability['max_drawdown_ratio'],
                stability['returns_skew'],
                stability['returns_kurtosis']
            ],
            name='Risk Metrics'
        ),
        row=4, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=1600,
        width=1600,
        showlegend=True,
        title_text="Strategy Optimization Results"
    )
    
    # Save plot
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.write_html(os.path.join(plots_dir, f"optimization_results_{timestamp}.html"))

def perform_sensitivity_analysis(optimizer: MLOptimizer, 
                              base_params: Dict[str, float],
                              param_ranges: Dict[str, Tuple[float, float, int]]) -> Dict[str, Any]:
    """Perform comprehensive parameter sensitivity analysis."""
    sensitivity_results = {
        'individual': {},
        'interaction': {},
        'stress_test': {},
        'stability': {}
    }
    
    # 1. Individual parameter sensitivity
    for param, (min_val, max_val, steps) in param_ranges.items():
        logger.info(f"Analyzing sensitivity for {param}...")
        param_values = np.linspace(min_val, max_val, steps)
        scores = []
        metrics = []
        
        for value in param_values:
            test_params = base_params.copy()
            test_params[param] = value
            
            # Run backtest with these parameters
            results = optimizer._evaluate_strategy(test_params, optimizer.data)
            score = optimizer._calculate_score(results)
            scores.append(score)
            metrics.append(results['metrics'])
        
        # Calculate sensitivity metrics
        sensitivity_results['individual'][param] = {
            'values': list(param_values),
            'scores': scores,
            'metrics': metrics,
            'gradient': np.gradient(scores, param_values).tolist(),
            'elasticity': (np.gradient(scores, param_values) * param_values[0] / scores[0]).tolist()
        }
    
    # 2. Parameter interactions
    for param1, param2 in itertools.combinations(param_ranges.keys(), 2):
        values1 = np.linspace(*param_ranges[param1][:2], 5)
        values2 = np.linspace(*param_ranges[param2][:2], 5)
        interaction_matrix = np.zeros((5, 5))
        
        for i, v1 in enumerate(values1):
            for j, v2 in enumerate(values2):
                test_params = base_params.copy()
                test_params[param1] = v1
                test_params[param2] = v2
                
                results = optimizer._evaluate_strategy(test_params, optimizer.data)
                interaction_matrix[i, j] = optimizer._calculate_score(results)
        
        sensitivity_results['interaction'][f"{param1}_vs_{param2}"] = {
            'values1': values1.tolist(),
            'values2': values2.tolist(),
            'matrix': interaction_matrix.tolist()
        }
    
    # 3. Stress testing
    market_conditions = {
        'bull': optimizer.data[optimizer.data['returns'] > 0],
        'bear': optimizer.data[optimizer.data['returns'] < 0],
        'high_vol': optimizer.data[optimizer.data['returns'].rolling(20).std() > optimizer.data['returns'].std()],
        'low_vol': optimizer.data[optimizer.data['returns'].rolling(20).std() < optimizer.data['returns'].std()]
    }
    
    for condition, condition_data in market_conditions.items():
        results = optimizer._evaluate_strategy(base_params, condition_data)
        sensitivity_results['stress_test'][condition] = {
            'metrics': results['metrics'],
            'score': optimizer._calculate_score(results)
        }
    
    # 4. Stability analysis
    window_sizes = [30, 60, 90, 180]  # days
    stability_scores = []
    
    for window in window_sizes:
        rolling_scores = []
        for i in range(0, len(optimizer.data) - window, window):
            window_data = optimizer.data.iloc[i:i+window]
            results = optimizer._evaluate_strategy(base_params, window_data)
            rolling_scores.append(optimizer._calculate_score(results))
        
        stability_scores.append({
            'window': window,
            'mean': np.mean(rolling_scores),
            'std': np.std(rolling_scores),
            'sharpe': np.mean(rolling_scores) / np.std(rolling_scores) if np.std(rolling_scores) != 0 else 0
        })
    
    sensitivity_results['stability'] = {
        'window_analysis': stability_scores,
        'overall_stability': np.mean([s['sharpe'] for s in stability_scores])
    }
    
    return sensitivity_results

def load_historical_data() -> pd.DataFrame:
    """Load or download historical data."""
    try:
        downloader = DataDownloader(data_dir="data", symbol="BTC-USD")
        return downloader.get_data(
            filename="historical_btc.csv",
            days_back=730,
            interval="1h"
        )
    except Exception as e:
        logger.error(f"Failed to load historical data: {e}")
        raise

def main():
    """Run strategy optimization."""
    # Set up logging
    setup_logger(level="INFO")
    logger.info("Starting strategy optimization...")
    
    try:
        # Load configuration
        config = load_config()
        optimization_config = config['optimization']
        
        # Load historical data
        data = load_historical_data()
        
        # Initialize optimizer with multiple ML models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=50,
                learning_rate=0.1,
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate='adaptive',
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            ),
            'adaboost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=42
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42
            )
        }
        
        optimizer = MLOptimizer(
            data=data,
            initial_capital=config['trading']['initial_capital'],
            commission=config['backtesting']['commission'],
            n_splits=5,
            model_dir="models"
        )
        
        # Define parameter ranges to optimize
        param_ranges = {
            # RSI parameters
            'rsi_period': (10, 30, 5),
            'rsi_oversold': (20, 40, 5),
            'rsi_overbought': (60, 80, 5),
            
            # MACD parameters
            'macd_fast': (8, 20, 4),
            'macd_slow': (20, 40, 5),
            'macd_signal': (5, 15, 3),
            
            # Position sizing and risk
            'position_size': (0.1, 0.5, 5),
            'stop_loss': (0.01, 0.05, 5),
            'take_profit': (0.02, 0.10, 5)
        }
        
        # Create parameter grid
        param_grid = create_parameter_grid(param_ranges)
        logger.info(f"Generated {len(param_grid)} parameter combinations")
        
        # Run optimization for each model
        all_results = []
        for model_name, model in models.items():
            logger.info(f"\nOptimizing with {model_name}...")
            optimizer._model = model
            results = optimizer.optimize(param_ranges)
            all_results.append(results)
            
            # Perform comprehensive sensitivity analysis
            logger.info("\nPerforming sensitivity analysis...")
            sensitivity = perform_sensitivity_analysis(
                optimizer,
                results.parameters,
                param_ranges
            )
            
            # Calculate statistical metrics
            stats_metrics = calculate_statistical_metrics([results])
            
            # Save results
            results_dir = "results"
            os.makedirs(results_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save sensitivity results
            sensitivity_file = os.path.join(
                results_dir,
                f"sensitivity_{model_name}_{timestamp}.json"
            )
            with open(sensitivity_file, 'w') as f:
                json.dump(sensitivity, f, indent=4, default=str)
            
            # Save statistical metrics
            stats_file = os.path.join(
                results_dir,
                f"statistics_{model_name}_{timestamp}.json"
            )
            with open(stats_file, 'w') as f:
                json.dump(stats_metrics, f, indent=4)
        
        # Create visualizations
        logger.info("\nGenerating visualizations...")
        plot_optimization_results(all_results, param_ranges)
        
        # Print summary of best results
        logger.info("\n=== Optimization Results ===")
        for model_name, results in zip(models.keys(), all_results):
            logger.info(f"\nModel: {model_name}")
            logger.info(f"Best score: {results.score:.2f}")
            
            logger.info("\nBest parameters:")
            for param, value in results.parameters.items():
                logger.info(f"{param}: {value:.4f}")
                
            logger.info("\nPerformance metrics:")
            for metric, value in results.metrics.items():
                logger.info(f"{metric}: {value:.4f}")
                
            logger.info("\nTop feature importance:")
            sorted_features = sorted(
                results.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for feature, importance in sorted_features[:5]:
                logger.info(f"{feature}: {importance:.4f}")
            
            # Print statistical significance
            stats_metrics = calculate_statistical_metrics([results])
            logger.info("\nStatistical Significance:")
            logger.info(f"Returns t-stat: {stats_metrics['statistical_tests']['returns_t_stat']:.2f} (p={stats_metrics['statistical_tests']['returns_p_value']:.4f})")
            logger.info(f"Sharpe t-stat: {stats_metrics['statistical_tests']['sharpe_t_stat']:.2f} (p={stats_metrics['statistical_tests']['sharpe_p_value']:.4f})")
            
            # Print stability metrics
            logger.info("\nStability Metrics:")
            logger.info(f"Returns Variation Ratio: {stats_metrics['stability_metrics']['returns_var_ratio']:.2f}")
            logger.info(f"Max Drawdown Ratio: {stats_metrics['stability_metrics']['max_drawdown_ratio']:.2f}")
            logger.info(f"Returns Skewness: {stats_metrics['stability_metrics']['returns_skew']:.2f}")
            logger.info(f"Returns Kurtosis: {stats_metrics['stability_metrics']['returns_kurtosis']:.2f}")
        
        logger.info("\nOptimization complete! Check plots/ and results/ directories for detailed analysis.")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise

if __name__ == "__main__":
    main() 