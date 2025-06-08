import React, { createContext, useContext, useState, ReactNode } from "react";
import { login as apiLogin, register as apiRegister } from "../services/api";

interface AuthContextType {
  token: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  register: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [token, setToken] = useState<string | null>(null);

  const login = async (username: string, password: string) => {
    const res = await apiLogin(username, password);
    if (res.token) {
      setToken(res.token);
      return true;
    }
    return false;
  };

  const register = async (username: string, password: string) => {
    const res = await apiRegister(username, password);
    if (res.token) {
      setToken(res.token);
      return true;
    }
    return false;
  };

  const logout = () => setToken(null);

  return (
    <AuthContext.Provider value={{ token, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}; 