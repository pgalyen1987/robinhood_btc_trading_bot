import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Heading,
  Stat,
  StatLabel,
  StatNumber,
  StatHelpText,
  useToast,
} from '@chakra-ui/react';
import { useWebSocket } from '../context/WebSocketContext';
import TradingChart from '../components/TradingChart';
import MetricsPanel from '../components/MetricsPanel';
import PositionsTable from '../components/PositionsTable';
import { fetchHistoricalData } from '../api/trading';

interface MarketData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface Position {
  symbol: string;
  size: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  pnlPercent: number;
}

const Dashboard: React.FC = () => {
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [metrics, setMetrics] = useState({
    equity: 0,
    dailyPnL: 0,
    winRate: 0,
    sharpeRatio: 0,
  });
  
  const { lastMessage } = useWebSocket();
  const toast = useToast();
  
  useEffect(() => {
    // Load initial historical data
    const loadHistoricalData = async () => {
      try {
        const data = await fetchHistoricalData();
        setMarketData(data);
      } catch (error) {
        toast({
          title: 'Error loading data',
          description: error.message,
          status: 'error',
          duration: 5000,
          isClosable: true,
        });
      }
    };
    
    loadHistoricalData();
  }, []);
  
  useEffect(() => {
    if (lastMessage) {
      const update = JSON.parse(lastMessage.data);
      
      switch (update.type) {
        case 'market_data':
          setMarketData(prev => [...prev, update.data]);
          break;
        case 'positions':
          setPositions(update.data);
          break;
        case 'metrics':
          setMetrics(update.data);
          break;
        default:
          console.log('Unknown update type:', update.type);
      }
    }
  }, [lastMessage]);
  
  return (
    <Box>
      <Heading mb={6}>Trading Dashboard</Heading>
      
      {/* Performance Metrics */}
      <Grid templateColumns="repeat(4, 1fr)" gap={4} mb={6}>
        <Stat>
          <StatLabel>Total Equity</StatLabel>
          <StatNumber>${metrics.equity.toLocaleString()}</StatNumber>
          <StatHelpText>Current Portfolio Value</StatHelpText>
        </Stat>
        <Stat>
          <StatLabel>Daily P&L</StatLabel>
          <StatNumber color={metrics.dailyPnL >= 0 ? 'green.500' : 'red.500'}>
            ${metrics.dailyPnL.toLocaleString()}
          </StatNumber>
          <StatHelpText>24h Change</StatHelpText>
        </Stat>
        <Stat>
          <StatLabel>Win Rate</StatLabel>
          <StatNumber>{metrics.winRate.toFixed(1)}%</StatNumber>
          <StatHelpText>Success Rate</StatHelpText>
        </Stat>
        <Stat>
          <StatLabel>Sharpe Ratio</StatLabel>
          <StatNumber>{metrics.sharpeRatio.toFixed(2)}</StatNumber>
          <StatHelpText>Risk-Adjusted Returns</StatHelpText>
        </Stat>
      </Grid>
      
      {/* Trading Chart */}
      <Box mb={6} bg="white" p={4} borderRadius="lg" boxShadow="sm">
        <TradingChart data={marketData} />
      </Box>
      
      {/* Metrics and Positions */}
      <Grid templateColumns="1fr 2fr" gap={6}>
        <MetricsPanel metrics={metrics} />
        <PositionsTable positions={positions} />
      </Grid>
    </Box>
  );
};

export default Dashboard; 