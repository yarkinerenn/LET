import axios from "axios";
import { API_URL } from "./config";

export const register = async (username: string, password: string) => {
    return axios.post(`${API_URL}/register`, { username, password });
};

export const login = async (username: string, password: string) => {
    return axios.post(`${API_URL}/login`, { username, password });
};