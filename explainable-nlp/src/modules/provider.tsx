import React, { createContext, useContext, useState } from "react";

interface ProviderContextType {
    providerex: string;
    setProviderex: React.Dispatch<React.SetStateAction<string>>;
    modelex: string;
    setModelex: React.Dispatch<React.SetStateAction<string>>;
    provider: string;
    setProvider: React.Dispatch<React.SetStateAction<string>>;
    model: string;
    setModel: React.Dispatch<React.SetStateAction<string>>;
}

const ProviderContext = createContext<ProviderContextType | undefined>(undefined);

export const ProviderContextProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [providerex, setProviderex] = useState<string>("");  // 'openai' or 'groq'
    const [modelex, setModelex] = useState<string>("");
    const [provider, setProvider] = useState<string>("");  // 'openai' or 'groq'
    const [model, setModel] = useState<string>("");

    return (
        <ProviderContext.Provider value={{ provider, setProvider,providerex, setProviderex, model, setModel, modelex, setModelex }}>
            {children}
        </ProviderContext.Provider>
    );
};

export const useProvider = (): ProviderContextType => {
    const context = useContext(ProviderContext);
    if (!context) {
        throw new Error("useProvider must be used within a ProviderContextProvider");
    }
    return context;
};