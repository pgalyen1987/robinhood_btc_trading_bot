import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ChakraProvider, Box, VStack } from '@chakra-ui/react';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Optimization from './pages/Optimization';
import Settings from './pages/Settings';
import { WebSocketProvider } from './context/WebSocketContext';
import { API_URL, WS_URL } from './config';

function App() {
  return (
    <ChakraProvider>
      <WebSocketProvider url={WS_URL}>
        <Router>
          <Box minH="100vh" bg="gray.50">
            <Navbar />
            <Box p={4}>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/optimization" element={<Optimization />} />
                <Route path="/settings" element={<Settings />} />
              </Routes>
            </Box>
          </Box>
        </Router>
      </WebSocketProvider>
    </ChakraProvider>
  );
}

export default App; 