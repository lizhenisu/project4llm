import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ArrowLeft } from "lucide-react";

export function ArchitecturePage({ onBack }: { onBack: () => void }) {
  const [content, setContent] = useState("");

  useEffect(() => {
    fetch("/ARCHITECTURE.md")
      .then((res) => res.text())
      .then(setContent)
      .catch(() => setContent("# 加载失败\n无法加载架构文档。"));
  }, []);

  return (
    <div className="architecture-page">
      <div className="architecture-header">
        <button className="icon-button" type="button" aria-label="返回" onClick={onBack}>
          <ArrowLeft size={18} />
        </button>
        <h1>系统架构</h1>
      </div>
      <article className="architecture-content">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </article>
    </div>
  );
}
