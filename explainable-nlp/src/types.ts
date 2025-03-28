export interface User {
    id: number;
    username: string;
}

export interface AuthContextType {
    user: User | null;
    login: (user: User) => void;
    logout: () => void;
    loading: boolean;

}
export interface Classification {
    id: string;
    text: string;
    label: string;
    score: number;
    timestamp: string;
}