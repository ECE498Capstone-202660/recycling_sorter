import React, { createContext, useContext, useState, useEffect, ReactNode, useMemo, useCallback } from "react";
import AsyncStorage from '@react-native-async-storage/async-storage';
import { login as apiLogin, register as apiRegister } from "../services/api";

interface AuthContextType {
  token: string | null;
  isBootstrapped: boolean;
  login: (username: string, password: string) => Promise<boolean>;
  register: (
    username: string,
    password: string,
    profile?: { email?: string | null; first_name?: string | null; last_name?: string | null }
  ) => Promise<boolean>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [token, setToken] = useState<string | null>(null);
  const [isBootstrapped, setIsBootstrapped] = useState(false);

  // Load token from AsyncStorage on mount
  useEffect(() => {
    (async () => {
      try {
        const storedToken = await AsyncStorage.getItem("token");
        if (storedToken) setToken(storedToken);
      } finally {
        setIsBootstrapped(true);
      }
    })();
  }, []);

  // Save token to AsyncStorage whenever it changes
  useEffect(() => {
    if (!isBootstrapped) return;
    if (token) {
      AsyncStorage.setItem("token", token);
    } else {
      AsyncStorage.removeItem("token");
    }
  }, [token]);

  const login = useCallback(async (username: string, password: string) => {
    const res = await apiLogin(username, password);
    if (res.token) {
      setToken(res.token);
      return true;
    }
    return false;
  }, []);

  const register = useCallback(
    async (
      username: string,
      password: string,
      profile?: { email?: string | null; first_name?: string | null; last_name?: string | null }
    ) => {
      const res = await apiRegister(username, password, profile);
    if (res.token) {
      setToken(res.token);
      return true;
    }
    return false;
    },
    []
  );

  const logout = useCallback(() => setToken(null), []);

  const value = useMemo(
    () => ({ token, isBootstrapped, login, register, logout }),
    [token, isBootstrapped, login, register, logout]
  );

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}; 
