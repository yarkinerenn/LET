// API Configuration
// Uses environment variable REACT_APP_API_URL if set, otherwise defaults to relative path /api
// In production with nginx proxy, use relative path /api
// For development, use http://localhost:5000
export const API_URL = process.env.REACT_APP_API_URL || (process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:5000');

