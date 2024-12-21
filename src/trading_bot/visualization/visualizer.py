"""Visualization utilities for trading bot."""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from pathlib import Path

from ..utils.logger import logger

class Visualizer:
    """Base class for visualization components."""
    
    def __init__(self, output_dir: str = "plots"):
        """Initialize visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_figure(self, fig: go.Figure, filename: str) -> None:
        """Save figure to file."""
        try:
            file_path = self.output_dir / filename
            fig.write_html(str(file_path))
            logger.info(f"Saved plot to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save plot: {e}")

class BacktestVisualizer(Visualizer):
    """Visualizer for backtesting results."""
    
    def __init__(self, data: pd.DataFrame, results: Dict[str, Any], output_dir: str = "plots"):
        """Initialize backtest visualizer.
        
        Args:
            data: Historical market data
            results: Backtesting results
            output_dir: Directory to save plots
        """
        super().__init__(output_dir)
        self.data = data
        self.results = results
        
    def plot_trades(self, show_indicators: bool = True) -> go.Figure:
        """Plot trade entries and exits with price data."""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add price candlesticks
        fig.add_trace(
            go.Candlestick(
                x=self.data.index,
                open=self.data['Open'],
                high=self.data['High'],
                low=self.data['Low'],
                close=self.data['Close'],
                name="Price"
            )
        )
        
        # Add volume bars
        fig.add_trace(
            go.Bar(
                x=self.data.index,
                y=self.data['Volume'],
                name="Volume",
                opacity=0.3
            ),
            secondary_y=True
        )
        
        # Add trade markers
        if 'trades' in self.results:
            trades = pd.DataFrame(self.results['trades'])
            
            # Buy markers
            fig.add_trace(
                go.Scatter(
                    x=trades[trades['Side'] == 'Buy'].index,
                    y=trades[trades['Side'] == 'Buy']['Price'],
                    mode='markers',
                    name='Buy',
                    marker=dict(
                        symbol='triangle-up',
                        size=15,
                        color='green'
                    )
                )
            )
            
            # Sell markers
            fig.add_trace(
                go.Scatter(
                    x=trades[trades['Side'] == 'Sell'].index,
                    y=trades[trades['Side'] == 'Sell']['Price'],
                    mode='markers',
                    name='Sell',
                    marker=dict(
                        symbol='triangle-down',
                        size=15,
                        color='red'
                    )
                )
            )
        
        # Add technical indicators
        if show_indicators and 'indicators' in self.results:
            for name, values in self.results['indicators'].items():
                fig.add_trace(
                    go.Scatter(
                        x=self.data.index,
                        y=values,
                        name=name,
                        line=dict(dash='dash')
                    )
                )
        
        fig.update_layout(
            title="Trading Activity",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2_title="Volume"
        )
        
        return fig
    
    def plot_equity_curve(self) -> go.Figure:
        """Plot equity curve and drawdown."""
        fig = make_subplots(rows=2, cols=1)
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.results['equity_curve'],
                name="Equity"
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.results['drawdown'],
                name="Drawdown",
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Performance Metrics",
            xaxis_title="Date",
            yaxis_title="Equity",
            yaxis2_title="Drawdown %"
        )
        
        return fig
    
    def plot_returns_distribution(self) -> go.Figure:
        """Plot distribution of returns."""
        returns = pd.Series(self.results['trade_returns'])
        
        fig = go.Figure()
        
        # Histogram of returns
        fig.add_trace(
            go.Histogram(
                x=returns,
                name="Returns",
                nbinsx=50,
                histnorm='probability'
            )
        )
        
        # Add KDE curve
        if len(returns) > 1:
            kde_x = np.linspace(returns.min(), returns.max(), 100)
            kde = returns.plot.kde()
            fig.add_trace(
                go.Scatter(
                    x=kde_x,
                    y=kde.get_lines()[0].get_data()[1],
                    name="KDE",
                    line=dict(color='red')
                )
            )
        
        fig.update_layout(
            title="Returns Distribution",
            xaxis_title="Return %",
            yaxis_title="Probability"
        )
        
        return fig

class DashboardApp:
    """Real-time trading dashboard."""
    
    def __init__(self, port: int = 8050):
        """Initialize dashboard.
        
        Args:
            port: Port to run dashboard on
        """
        self.app = Dash(__name__)
        self.port = port
        self.latest_data = None
        self.latest_results = None
        
        # Create layout
        self.app.layout = html.Div([
            html.H1("Trading Bot Dashboard"),
            
            # Market data and position
            html.Div([
                html.H2("Market Data and Position"),
                dcc.Graph(id='market-graph'),
                dcc.Interval(
                    id='market-interval',
                    interval=60*1000,  # Update every minute
                    n_intervals=0
                )
            ]),
            
            # Performance metrics
            html.Div([
                html.H2("Performance Metrics"),
                dcc.Graph(id='performance-graph'),
                dcc.Interval(
                    id='performance-interval',
                    interval=60*1000,
                    n_intervals=0
                )
            ])
        ])
        
        self._register_callbacks()
    
    def update_data(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Update dashboard data."""
        self.latest_data = data
        self.latest_results = results
    
    def _register_callbacks(self) -> None:
        """Register Dash callbacks."""
        
        @self.app.callback(
            Output('market-graph', 'figure'),
            Input('market-interval', 'n_intervals')
        )
        def update_market_graph(_):
            if self.latest_data is None:
                return go.Figure()
                
            visualizer = BacktestVisualizer(self.latest_data, self.latest_results or {})
            return visualizer.plot_trades(show_indicators=True)
        
        @self.app.callback(
            Output('performance-graph', 'figure'),
            Input('performance-interval', 'n_intervals')
        )
        def update_performance_graph(_):
            if self.latest_data is None or self.latest_results is None:
                return go.Figure()
                
            visualizer = BacktestVisualizer(self.latest_data, self.latest_results)
            return visualizer.plot_equity_curve()
    
    def run(self) -> None:
        """Run the dashboard server."""
        self.app.run_server(debug=True, port=self.port) 