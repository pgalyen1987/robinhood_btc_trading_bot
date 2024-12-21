"""Real-time data visualization using Dash."""

from typing import Dict, Any, Optional, List
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..utils.logger import logger

class DataViewer:
    """Real-time data visualization dashboard."""
    
    def __init__(self, update_interval: int = 5000):
        """Initialize data viewer.
        
        Args:
            update_interval: Dashboard update interval in milliseconds.
        """
        self.app = dash.Dash(__name__)
        self.update_interval = update_interval
        self.data = pd.DataFrame()
        self.trades = []
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self) -> None:
        """Set up dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.H1('Trading Bot Dashboard', style={'textAlign': 'center'}),
            
            # Price chart
            html.Div([
                dcc.Graph(id='price-chart'),
            ], style={'width': '100%', 'display': 'inline-block'}),
            
            # Trading metrics
            html.Div([
                html.Div([
                    html.H4('Trading Metrics'),
                    html.Table([
                        html.Tr([html.Td('Equity:'), html.Td(id='equity')]),
                        html.Tr([html.Td('Position:'), html.Td(id='position')]),
                        html.Tr([html.Td('PnL:'), html.Td(id='pnl')]),
                        html.Tr([html.Td('Win Rate:'), html.Td(id='win-rate')])
                    ])
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                # Technical indicators
                html.Div([
                    html.H4('Technical Indicators'),
                    html.Table([
                        html.Tr([html.Td('RSI:'), html.Td(id='rsi')]),
                        html.Tr([html.Td('MACD:'), html.Td(id='macd')]),
                        html.Tr([html.Td('Signal:'), html.Td(id='signal')])
                    ])
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                # Market data
                html.Div([
                    html.H4('Market Data'),
                    html.Table([
                        html.Tr([html.Td('Price:'), html.Td(id='price')]),
                        html.Tr([html.Td('Volume:'), html.Td(id='volume')]),
                        html.Tr([html.Td('24h Change:'), html.Td(id='change')])
                    ])
                ], style={'width': '30%', 'display': 'inline-block'})
            ], style={'width': '100%', 'display': 'flex', 'justifyContent': 'space-between'}),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            )
        ])
        
    def _setup_callbacks(self) -> None:
        """Set up dashboard callbacks."""
        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('equity', 'children'),
             Output('position', 'children'),
             Output('pnl', 'children'),
             Output('win-rate', 'children'),
             Output('rsi', 'children'),
             Output('macd', 'children'),
             Output('signal', 'children'),
             Output('price', 'children'),
             Output('volume', 'children'),
             Output('change', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            try:
                # Create price chart
                fig = self._create_price_chart()
                
                # Get latest metrics
                metrics = self._get_latest_metrics()
                
                return (
                    fig,
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
                )
                
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                return dash.no_update
                
    def _create_price_chart(self) -> go.Figure:
        """Create price chart with indicators.
        
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
            
            # Add trades if available
            for trade in self.trades:
                # Entry point
                fig.add_trace(
                    go.Scatter(
                        x=[trade['entry_time']],
                        y=[trade['entry_price']],
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
                
                # Exit point if trade is closed
                if trade.get('exit_price'):
                    fig.add_trace(
                        go.Scatter(
                            x=[trade['exit_time']],
                            y=[trade['exit_price']],
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
                    
            # Update layout
            fig.update_layout(
                title='Real-time Trading Data',
                yaxis_title='Price',
                yaxis2_title='Volume',
                xaxis_rangeslider_visible=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price chart: {e}")
            raise
            
    def _get_latest_metrics(self) -> Dict[str, float]:
        """Get latest trading metrics.
        
        Returns:
            Dictionary of current metrics.
        """
        try:
            if self.data.empty:
                return {
                    'equity': 0.0,
                    'position': 0.0,
                    'pnl': 0.0,
                    'win_rate': 0.0,
                    'rsi': 0.0,
                    'macd': 0.0,
                    'signal': 0.0,
                    'price': 0.0,
                    'volume': 0.0,
                    'change': 0.0
                }
                
            latest = self.data.iloc[-1]
            
            return {
                'equity': self.get_equity(),
                'position': self.get_position_size(),
                'pnl': self.get_pnl(),
                'win_rate': self.calculate_win_rate(),
                'rsi': latest.get('RSI', 0.0),
                'macd': latest.get('MACD', 0.0),
                'signal': latest.get('Signal', 0.0),
                'price': latest.Close,
                'volume': latest.Volume,
                'change': self.calculate_24h_change()
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            raise
            
    def update_data(self, data: pd.DataFrame, trades: List[Dict[str, Any]]) -> None:
        """Update dashboard data.
        
        Args:
            data: New OHLCV data.
            trades: List of trades.
        """
        self.data = data
        self.trades = trades
        
    def run(self, host: str = 'localhost', port: int = 8050, debug: bool = False) -> None:
        """Run the dashboard server.
        
        Args:
            host: Server host.
            port: Server port.
            debug: Whether to run in debug mode.
        """
        try:
            self.app.run_server(host=host, port=port, debug=debug)
        except Exception as e:
            logger.error(f"Dashboard server error: {e}")
            raise
            
    # Helper methods for metrics calculation
    def get_equity(self) -> float:
        """Get current equity."""
        # Implement equity calculation
        return 0.0
        
    def get_position_size(self) -> float:
        """Get current position size."""
        # Implement position size calculation
        return 0.0
        
    def get_pnl(self) -> float:
        """Get current profit/loss."""
        # Implement PnL calculation
        return 0.0
        
    def calculate_win_rate(self) -> float:
        """Calculate win rate from trades."""
        if not self.trades:
            return 0.0
            
        closed_trades = [t for t in self.trades if t.get('exit_price')]
        if not closed_trades:
            return 0.0
            
        winning_trades = sum(1 for t in closed_trades if t.get('pnl', 0) > 0)
        return (winning_trades / len(closed_trades)) * 100
        
    def calculate_24h_change(self) -> float:
        """Calculate 24-hour price change."""
        if len(self.data) < 24:
            return 0.0
            
        current_price = self.data.Close.iloc[-1]
        price_24h_ago = self.data.Close.iloc[-24]
        return ((current_price - price_24h_ago) / price_24h_ago) * 100 