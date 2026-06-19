import { useEffect, useState } from "react";
import { AuthPage } from "./app/AuthPage";
import { ArchitecturePage, MarkdownPage } from "./app/ArchitecturePage";
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
  const isArchitecture = pathname === "/architecture";
  const isProjectEvaluation = pathname === "/project-evaluation" || pathname === "/PROJECT_EVALUATION.md";

  return (
    <AuthProvider>
      {authMode ? (
        <AuthPage mode={authMode} onNavigate={navigate} />
      ) : isArchitecture ? (
        <ArchitecturePage onBack={() => navigate("/", { replace: true })} />
      ) : isProjectEvaluation ? (
        <MarkdownPage
          documentPath="/PROJECT_EVALUATION.md"
          loadError="# 加载失败\n无法加载项目测评文档。"
          onBack={() => navigate("/architecture", { replace: true })}
          title="项目测评"
        />
      ) : (
        <WorkspacePage onNavigate={navigate} />
      )}
    </AuthProvider>
  );
}
