import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { loginAccount, logoutAccount, registerAccount, setUnauthorizedHandler } from "./api";
import { loadSettings } from "./storage";
import type { AuthUser } from "./types";

type AuthSession = {
  user: AuthUser;
  token: string;
  expires_at: number;
};

type AuthContextValue = {
  user: AuthUser | null;
  token: string;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string, displayName: string) => Promise<void>;
  setUser: (user: AuthUser) => void;
  logout: () => Promise<void>;
};

const STORAGE_KEY = "production-rag-auth-session";
const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<AuthSession | null>(() => loadAuthSession());

  useEffect(() => {
    setUnauthorizedHandler(() => {
      localStorage.removeItem(STORAGE_KEY);
      setSession(null);
    });
    return () => setUnauthorizedHandler(null);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      user: session?.user ?? null,
      token: session?.token ?? "",
      async login(username, password) {
        const response = await loginAccount(loadSettings(), { username, password });
        persistAuthSession(response);
        setSession(response);
      },
      async register(username, password, displayName) {
        const response = await registerAccount(loadSettings(), {
          username,
          password,
          displayName,
        });
        persistAuthSession(response);
        setSession(response);
      },
      setUser(user) {
        setSession((current) => {
          if (!current) return current;
          const next = { ...current, user };
          persistAuthSession(next);
          return next;
        });
      },
      async logout() {
        const current = loadAuthSession();
        if (current?.token) {
          await logoutAccount({ ...loadSettings(), token: current.token });
        }
        localStorage.removeItem(STORAGE_KEY);
        setSession(null);
      },
    }),
    [session],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within AuthProvider");
  }
  return context;
}

function loadAuthSession(): AuthSession | null {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    const session = JSON.parse(raw) as AuthSession;
    if (!session.token || !session.user || session.expires_at <= Date.now()) {
      localStorage.removeItem(STORAGE_KEY);
      return null;
    }
    return session;
  } catch {
    localStorage.removeItem(STORAGE_KEY);
    return null;
  }
}

function persistAuthSession(session: AuthSession) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
}
