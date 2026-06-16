import { createElement, useEffect, useState } from "react";
import type { ReactNode } from "react";
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

  useEffect(() => {
    if (!content) return;
    scrollToArchitectureHash();
    window.addEventListener("hashchange", scrollToArchitectureHash);
    return () => window.removeEventListener("hashchange", scrollToArchitectureHash);
  }, [content]);

  return (
    <div className="architecture-page">
      <div className="architecture-header">
        <button className="icon-button" type="button" aria-label="返回" onClick={onBack}>
          <ArrowLeft size={18} />
        </button>
        <h1>系统架构</h1>
      </div>
      <article className="architecture-content">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>{content}</ReactMarkdown>
      </article>
    </div>
  );
}

const markdownComponents = {
  h1: Heading("h1"),
  h2: Heading("h2"),
  h3: Heading("h3"),
  h4: Heading("h4"),
  h5: Heading("h5"),
  h6: Heading("h6"),
};

function Heading(tag: "h1" | "h2" | "h3" | "h4" | "h5" | "h6") {
  return function MarkdownHeading({ children }: { children?: ReactNode }) {
    const text = textFromChildren(children);
    return createElement(tag, { id: architectureSlug(text) }, children);
  };
}

function scrollToArchitectureHash() {
  const hash = decodeURIComponent(window.location.hash.replace(/^#/, ""));
  if (!hash) return;
  window.requestAnimationFrame(() => {
    const target = document.getElementById(hash);
    const scroller = document.querySelector<HTMLElement>(".architecture-content");
    if (!target || !scroller) return;
    const targetTop = target.getBoundingClientRect().top - scroller.getBoundingClientRect().top + scroller.scrollTop;
    scroller.scrollTo({ top: Math.max(0, targetTop - 12), behavior: "auto" });
  });
}

function architectureSlug(text: string) {
  return text
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s-]/gu, "")
    .replace(/\s+/g, "-");
}

function textFromChildren(children: ReactNode): string {
  if (typeof children === "string" || typeof children === "number") {
    return String(children);
  }
  if (Array.isArray(children)) {
    return children.map(textFromChildren).join("");
  }
  if (children && typeof children === "object" && "props" in children) {
    const element = children as { props?: { children?: ReactNode } };
    return textFromChildren(element.props?.children);
  }
  return "";
}
