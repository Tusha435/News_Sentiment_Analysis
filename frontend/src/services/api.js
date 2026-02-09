/**
 * API service for communicating with the FastAPI backend.
 */

import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
  headers: { 'Content-Type': 'application/json' },
});

// Stock endpoints
export const searchStocks = (query) =>
  api.get(`/stocks/search?q=${encodeURIComponent(query)}`);

export const getStockPrices = (symbol, days = 365) =>
  api.get(`/stocks/${symbol}/prices?days=${days}`);

export const getCompanyOverview = (symbol) =>
  api.get(`/stocks/${symbol}/overview`);

export const getFundamentals = (symbol) =>
  api.get(`/stocks/${symbol}/fundamentals`);

// Sentiment endpoints
export const analyzeSentiment = (symbol, days = 30, maxArticles = 20) =>
  api.get(`/sentiment/${symbol}/analyze?days=${days}&max_articles=${maxArticles}`);

// Prediction endpoints
export const getPredictions = (symbol, modelType = 'hybrid') =>
  api.get(`/predictions/${symbol}?model_type=${modelType}`);

export const trainModel = (symbol, modelType = 'hybrid', days = 730) =>
  api.post('/predictions/train', { symbol, model_type: modelType, days });

export const getDashboard = (symbol, days = 365) =>
  api.get(`/predictions/${symbol}/dashboard?days=${days}`);

export default api;
