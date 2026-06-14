import { useEffect, useState } from "react";
import { AuthPage } from "./app/AuthPage";
import { WorkspacePage } from "./app/WorkspacePage";
import { AuthProvider } from "./lib/AuthContext";

export function App() {
  const [pathname, setPathname] = useState(() => window.location.pathname);

  useEffect(() => {
    function handlePopState() {
      setPathname(window.location.pathname);
    }
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  function navigate(path: string, { replace = false }: { replace?: boolean } = {}) {
    if (window.location.pathname === path) return;
    const method = replace ? "replaceState" : "pushState";
    window.history[method](null, "", path);
    setPathname(path);
  }

  const authMode = pathname === "/register" ? "register" : pathname === "/login" ? "login" : null;

  return (
    <AuthProvider>
      {authMode ? (
        <AuthPage mode={authMode} onNavigate={navigate} />
      ) : (
        <WorkspacePage onNavigate={navigate} />
      )}
    </AuthProvider>
  );
}
