"""Visualization utilities for backtesting results."""

from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..utils.logger import logger

class Plotter:
    """Plotter for backtesting results visualization."""
    
    def __init__(self):
        # initialization code here
        pass
    
    # rest of the class implementation

class BacktestPlotter:
    """Plotter for backtesting results visualization."""
    
    def __init__(self, data: pd.DataFrame, results: Dict[str, Any]):
        """Initialize plotter with data and results.
        
        Args:
            data: OHLCV data used in backtesting.
            results: Backtesting results.
        """
        self.data = data
        self.results = results
        self._validate_data()
        
    def _validate_data(self) -> None:
        """Validate input data and results."""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
            
    def plot_trades(self, show_indicators: bool = True) -> go.Figure:
        """Plot trading results with candlestick chart and trades.
        
        Args:
            show_indicators: Whether to show technical indicators.
            
        Returns:
            Plotly figure object.
        """
        try:
            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3]
            )
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=self.data.index,
                    open=self.data.Open,
                    high=self.data.High,
                    low=self.data.Low,
                    close=self.data.Close,
                    name='OHLC'
                ),
                row=1, col=1
            )
            
            # Add volume bars
            fig.add_trace(
                go.Bar(
                    x=self.data.index,
                    y=self.data.Volume,
                    name='Volume'
                ),
                row=2, col=1
            )
            
            # Add trades
            trades = self.results.get('trades', [])
            for trade in trades:
                # Entry point
                fig.add_trace(
                    go.Scatter(
                        x=[trade.entry_time],
                        y=[trade.entry_price],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=10,
                            color='green'
                        ),
                        name='Entry'
                    ),
                    row=1, col=1
                )
                
                # Exit point
                if trade.exit_price:
                    fig.add_trace(
                        go.Scatter(
                            x=[trade.exit_time],
                            y=[trade.exit_price],
                            mode='markers',
                            marker=dict(
                                symbol='triangle-down',
                                size=10,
                                color='red'
                            ),
                            name='Exit'
                        ),
                        row=1, col=1
                    )
                    
            # Add technical indicators if available and requested
            if show_indicators and hasattr(self.results, '_strategy'):
                strategy = self.results._strategy
                if hasattr(strategy, 'rsi'):
                    fig.add_trace(
                        go.Scatter(
                            x=self.data.index,
                            y=strategy.rsi,
                            name='RSI'
                        ),
                        row=2, col=1
                    )
                    
                if hasattr(strategy, 'macd'):
                    fig.add_trace(
                        go.Scatter(
                            x=self.data.index,
                            y=strategy.macd,
                            name='MACD'
                        ),
                        row=2, col=1
                    )
                    
                if hasattr(strategy, 'signal'):
                    fig.add_trace(
                        go.Scatter(
                            x=self.data.index,
                            y=strategy.signal,
                            name='Signal'
                        ),
                        row=2, col=1
                    )
                    
            # Update layout
            fig.update_layout(
                title='Trading Results',
                yaxis_title='Price',
                yaxis2_title='Volume/Indicators',
                xaxis_rangeslider_visible=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting trades: {e}")
            raise
            
    def plot_equity_curve(self) -> go.Figure:
        """Plot equity curve.
        
        Returns:
            Plotly figure object.
        """
        try:
            # Create figure
            fig = go.Figure()
            
            # Add equity curve
            equity = self.results.get('_equity_curve', pd.Series())
            fig.add_trace(
                go.Scatter(
                    x=equity.index,
                    y=equity.Equity,
                    name='Equity',
                    line=dict(color='blue')
                )
            )
            
            # Add drawdown
            if 'DrawdownPct' in equity.columns:
                fig.add_trace(
                    go.Scatter(
                        x=equity.index,
                        y=equity.DrawdownPct,
                        name='Drawdown %',
                        line=dict(color='red'),
                        yaxis='y2'
                    )
                )
                
            # Update layout
            fig.update_layout(
                title='Equity Curve and Drawdown',
                yaxis_title='Equity',
                yaxis2=dict(
                    title='Drawdown %',
                    overlaying='y',
                    side='right'
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
            raise
            
    def plot_returns_distribution(self) -> go.Figure:
        """Plot distribution of returns.
        
        Returns:
            Plotly figure object.
        """
        try:
            # Calculate returns
            equity = self.results.get('_equity_curve', pd.Series())
            returns = equity.Equity.pct_change().dropna()
            
            # Create figure
            fig = go.Figure()
            
            # Add histogram
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    name='Returns',
                    nbinsx=50,
                    histnorm='probability'
                )
            )
            
            # Update layout
            fig.update_layout(
                title='Distribution of Returns',
                xaxis_title='Return',
                yaxis_title='Probability',
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting returns distribution: {e}")
            raise 