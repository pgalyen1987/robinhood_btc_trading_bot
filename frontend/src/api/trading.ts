import axios from 'axios';
import { API_URL } from '../config';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface HistoricalDataParams {
  symbol?: string;
  interval?: string;
  days?: number;
}

export interface OptimizationParams {
  symbol?: string;
  days?: number;
}

export const fetchHistoricalData = async (params: HistoricalDataParams = {}) => {
  try {
    const response = await api.get('/data/historical', { params });
    return response.data.data;
  } catch (error) {
    console.error('Error fetching historical data:', error);
    throw new Error('Failed to fetch historical data');
  }
};

export const runOptimization = async (params: OptimizationParams = {}) => {
  try {
    const response = await api.get('/strategy/optimize', { params });
    return response.data.optimization_result;
  } catch (error) {
    console.error('Error running optimization:', error);
    throw new Error('Failed to run strategy optimization');
  }
};

export const getServerStatus = async () => {
  try {
    const response = await api.get('/');
    return response.data;
  } catch (error) {
    console.error('Error checking server status:', error);
    throw new Error('Failed to connect to server');
  }
}; 