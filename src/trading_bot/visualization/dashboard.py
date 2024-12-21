#!/usr/bin/env python3
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional
import threading
import webbrowser
from ..utils.logger import logger

class DashboardApp:
    """Trading bot dashboard application."""
    
    def __init__(self, port: int = 8050, host: str = '0.0.0.0', update_interval: int = 60000):
        """Initialize dashboard components.
        
        Args:
            port: Port number for the dashboard server
            host: Host address for the dashboard server
            update_interval: Update interval in milliseconds
        """
        self.dashboard = Dashboard(host=host, port=port, update_interval=update_interval)
        self.port = port
        
    def run(self):
        """Run the dashboard server."""
        try:
            self.dashboard.app.run_server(
                host=self.dashboard.host,
                port=self.dashboard.port,
                debug=False
            )
        except Exception as e:
            logger.error(f"Dashboard server error: {e}")
            raise
            
    def update_data(self, data: pd.DataFrame, results: Dict[str, Any]) -> None:
        """Update dashboard with new data and results."""
        self.dashboard.update_data(data, results)

class Dashboard:
    """Unified dashboard combining real-time visualization and ML metrics."""
    
    def __init__(self, host='0.0.0.0', port=8050, update_interval=60000):
        self.app = Dash(__name__)
        self.host = host
        self.port = port
        self.update_interval = update_interval
        
        # Data storage
        self.data = None
        self.trades = []
        self.ml_metrics = None
        
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Create unified dashboard layout."""
        self.app.layout = html.Div([
            html.H1('Trading Bot Dashboard', style={'textAlign': 'center'}),
            
            # Market Data Section
            self._create_market_section(),
            
            # Trading Metrics Section
            self._create_metrics_section(),
            
            # Strategy Performance Section
            self._create_strategy_section(),
            
            # ML Performance Section
            self._create_ml_section()
        ])
    
    def _create_market_section(self):
        """Create market data visualization section."""
        return html.Div([
            html.H2("Market Data and Position"),
            dcc.Graph(id='market-graph'),
            dcc.Interval(
                id='market-interval',
                interval=self.update_interval,
                n_intervals=0
            )
        ])
    
    def _create_metrics_section(self):
        """Create trading metrics section."""
        return html.Div([
            html.Div([
                html.H4('Trading Metrics'),
                self._create_metric_table([
                    ('Equity', 'equity'),
                    ('Position', 'position'),
                    ('PnL', 'pnl'),
                    ('Win Rate', 'win-rate')
                ])
            ], style={'width': '33%', 'display': 'inline-block'}),
            
            html.Div([
                html.H4('Technical Indicators'),
                self._create_metric_table([
                    ('RSI', 'rsi'),
                    ('MACD', 'macd'),
                    ('Signal', 'signal')
                ])
            ], style={'width': '33%', 'display': 'inline-block'}),
            
            html.Div([
                html.H4('Market Data'),
                self._create_metric_table([
                    ('Price', 'price'),
                    ('Volume', 'volume'),
                    ('24h Change', 'change')
                ])
            ], style={'width': '33%', 'display': 'inline-block'})
        ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between'})
    
    def _create_metric_table(self, metrics):
        """Helper to create metric tables."""
        return html.Table([
            html.Tr([html.Td(f'{label}:'), html.Td(id=id_)])
            for label, id_ in metrics
        ])
    
    def _create_strategy_section(self):
        """Create strategy performance section."""
        return html.Div([
            html.H2("Strategy Performance"),
            dcc.Graph(id='strategy-graph'),
            dcc.Interval(
                id='strategy-interval',
                interval=self.update_interval,
                n_intervals=0
            )
        ])
    
    def _create_ml_section(self):
        """Create ML performance section."""
        return html.Div([
            html.H2("ML Model Performance"),
            dcc.Graph(id='ml-graph'),
            dcc.Interval(
                id='ml-interval',
                interval=self.update_interval,
                n_intervals=0
            )
        ])

    def _setup_callbacks(self):
        """Set up all dashboard callbacks."""
        self._setup_market_callbacks()
        self._setup_strategy_callbacks()
        self._setup_ml_callbacks()
    
    def _setup_market_callbacks(self):
        """Set up market data callbacks."""
        @self.app.callback(
            [Output('market-graph', 'figure')] + 
            [Output(id_, 'children') for id_ in [
                'equity', 'position', 'pnl', 'win-rate',
                'rsi', 'macd', 'signal', 'price', 'volume', 'change'
            ]],
            [Input('market-interval', 'n_intervals')]
        )
        def update_market_data(_):
            if self.data is None:
                return [go.Figure()] + ["N/A"] * 10
            
            try:
                fig = self._create_market_figure()
                metrics = self._get_latest_metrics()
                
                return [fig] + [
                    f"${metrics['equity']:.2f}",
                    f"{metrics['position']:.8f} BTC",
                    f"${metrics['pnl']:.2f}",
                    f"{metrics['win_rate']:.1f}%",
                    f"{metrics['rsi']:.1f}",
                    f"{metrics['macd']:.8f}",
                    f"{metrics['signal']:.8f}",
                    f"${metrics['price']:.2f}",
                    f"{metrics['volume']:.2f}",
                    f"{metrics['change']:.2f}%"
                ]
            except Exception as e:
                logger.error(f"Error updating market data: {e}")
                return [go.Figure()] + ["Error"] * 10

    def _create_market_figure(self):
        """Create market data visualization figure."""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add candlestick chart
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
                name="Volume"
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title="Market Data",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            yaxis2_title="Volume"
        )
        
        return fig

    def update_data(self, data: pd.DataFrame, trades: List[Dict[str, Any]], 
                   ml_metrics: Optional[Dict[str, Any]] = None):
        """Update dashboard data."""
        self.data = data
        self.trades = trades
        self.ml_metrics = ml_metrics

    def run(self, open_browser=True):
        """Run the dashboard server."""
        if open_browser:
            threading.Timer(1.5, lambda: webbrowser.open(
                f'http://{self.host}:{self.port}'
            )).start()
        
        logger.info(f"Starting dashboard at http://{self.host}:{self.port}")
        self.app.run_server(host=self.host, port=self.port, debug=False) 