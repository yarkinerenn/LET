export interface User {
    id: number;
    username: string;
}

export interface AuthContextType {
    user: User | null;
    login: (user: User) => void;
    logout: () => void;
}
export interface Classification {
    id: string;
    text: string;
    label: string;
    score: number;
    timestamp: string;
}