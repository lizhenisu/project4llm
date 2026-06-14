import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { DatabaseZap } from "lucide-react";
import { useAuth } from "../lib/AuthContext";

type AuthMode = "login" | "register";

export function AuthPage({
  mode,
  onNavigate,
}: {
  mode: AuthMode;
  onNavigate: (path: string, options?: { replace?: boolean }) => void;
}) {
  const auth = useAuth();
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const isRegister = mode === "register";

  useEffect(() => {
    if (auth.user && auth.token) {
      onNavigate("/", { replace: true });
    }
  }, [auth.user, auth.token, onNavigate]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      if (isRegister) {
        await auth.register(username, password, displayName);
      } else {
        await auth.login(username, password);
      }
      onNavigate("/", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "操作失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="auth-page">
      <form className="auth-page-panel" onSubmit={submit}>
        <div className="auth-page-brand">
          <span className="brand-mark">
            <DatabaseZap size={22} />
          </span>
          <span>Production RAG</span>
        </div>
        <div className="auth-page-heading">
          <h1>{isRegister ? "注册账号" : "登录账号"}</h1>
          <p>{isRegister ? "创建账号后进入知识库工作台。" : "登录后继续使用知识库工作台。"}</p>
        </div>
        <label>
          用户名
          <input
            value={username}
            onChange={(event) => setUsername(event.target.value)}
            autoComplete="username"
            autoFocus
          />
        </label>
        {isRegister ? (
          <label>
            显示名称
            <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} autoComplete="name" />
          </label>
        ) : null}
        <label>
          密码
          <input
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            type="password"
            autoComplete={isRegister ? "new-password" : "current-password"}
          />
        </label>
        {error ? <p className="error-text">{error}</p> : null}
        <button className="auth-page-submit" type="submit" disabled={submitting || !username.trim() || !password}>
          {submitting ? "处理中..." : isRegister ? "注册并登录" : "登录"}
        </button>
        <p className="auth-page-switch">
          {isRegister ? "已有账号？" : "没有账号？"}
          <button type="button" onClick={() => onNavigate(isRegister ? "/login" : "/register")}>
            {isRegister ? "去登录" : "去注册"}
          </button>
        </p>
      </form>
    </main>
  );
}
