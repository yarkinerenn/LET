import axios from "axios";

const API_URL = "http://127.0.0.1:5000";

export const register = async (username: string, password: string) => {
    return axios.post(`${API_URL}/register`, { username, password });
};

export const login = async (username: string, password: string) => {
    return axios.post(`${API_URL}/login`, { username, password });
};