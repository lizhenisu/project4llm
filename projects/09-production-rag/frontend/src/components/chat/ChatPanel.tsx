import { ArrowRight, Bot, Check, Copy, ImagePlus, MoreVertical, ThumbsDown, ThumbsUp, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import type { ChatMessage, Citation, SourceItem } from "../../lib/types";
import { EmptyState } from "../ui/EmptyState";

type Props = {
  messages: ChatMessage[];
  selectedSources: SourceItem[];
  authenticated: boolean;
  busy: boolean;
  conversationTitle: string;
  typingMessageId: string | null;
  onTypingComplete: () => void;
  onAsk: (query: string, imageDataUrl?: string | null) => void;
  onFeedback: (message: ChatMessage, rating: 1 | -1) => void;
  onDeleteConversation: () => void;
};

export function ChatPanel({
  messages,
  selectedSources,
  authenticated,
  busy,
  conversationTitle,
  typingMessageId,
  onTypingComplete,
  onAsk,
  onFeedback,
  onDeleteConversation,
}: Props) {
  const [draft, setDraft] = useState("");
  const [attachedImage, setAttachedImage] = useState<string | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [inputHeight, setInputHeight] = useState(46);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const canSend = (draft.trim().length > 0 || Boolean(attachedImage)) && (authenticated ? selectedSources.length > 0 : true) && !busy;

  useEffect(() => {
    function handleClickOutside() {
      setMenuOpen(false);
    }
    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  }, []);


  function submit() {
    if (!canSend) return;
    onAsk(draft.trim() || "请根据这张图片检索相关资料并回答。", attachedImage);
    setDraft("");
    setAttachedImage(null);
  }

  function attachImage(file: File | undefined) {
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = () => setAttachedImage(typeof reader.result === "string" ? reader.result : null);
    reader.readAsDataURL(file);
  }

  function handlePointerDown(event: React.PointerEvent<HTMLDivElement>) {
    const target = event.currentTarget;
    const startY = event.clientY;
    const startHeight = textareaRef.current?.getBoundingClientRect().height || inputHeight;
    target.setPointerCapture(event.pointerId);

    function onPointerMove(e: PointerEvent) {
      const deltaY = startY - e.clientY; // Dragging UP means smaller Y, positive delta
      setInputHeight(Math.max(46, Math.min(600, startHeight + deltaY)));
    }

    function onPointerUp(e: PointerEvent) {
      target.releasePointerCapture(e.pointerId);
      target.removeEventListener("pointermove", onPointerMove);
      target.removeEventListener("pointerup", onPointerUp);
      document.body.style.cursor = "";
    }

    document.body.style.cursor = "row-resize";
    target.addEventListener("pointermove", onPointerMove);
    target.addEventListener("pointerup", onPointerUp);
  }

  return (
    <section className="panel chat-panel">
      <div className="panel-header">
        <h2 title={conversationTitle}>对话</h2>
        <div className="chat-header-actions">
          <div className="chat-menu">
            <button
              className="icon-button"
              type="button"
              aria-label="更多"
              title="更多"
              onClick={(e) => {
                e.stopPropagation();
                setMenuOpen((open) => !open);
              }}
            >
              <MoreVertical size={18} />
            </button>
            {menuOpen ? (
              <div className="chat-menu-popover">
                <button
                  type="button"
                  disabled={messages.length === 0}
                  onClick={() => {
                    setMenuOpen(false);
                    onDeleteConversation();
                  }}
                >
                  删除对话记录
                </button>
                <p>只有您自己能看到对话记录。</p>
              </div>
            ) : null}
          </div>
        </div>
      </div>
      <div className="chat-scroll">
        {messages.length === 0 ? (
          selectedSources.length === 0 ? (
            <EmptyState
              icon={<Bot size={34} />}
              title="让我们开始构建知识库..."
              text="添加来源后，你可以基于资料提问、生成摘要和创建思维导图。"
            />
          ) : (
            <Overview sources={selectedSources} onAsk={onAsk} />
          )
        ) : (
          <div className="message-list">
            {messages.map((message) =>
              message.role === "user" ? (
                <div className="user-message" key={message.id}>
                  {message.content}
                </div>
              ) : (
                <AssistantMessage
                  key={message.id}
                  message={message}
                  typing={message.id === typingMessageId && message.status === "done"}
                  onTypingComplete={onTypingComplete}
                  onFeedback={onFeedback}
                />
              ),
            )}
          </div>
        )}
      </div>
      <div className="chat-input" style={{ position: "relative" }}>
        <div
          className="chat-input-resizer"
          onPointerDown={handlePointerDown}
        />
        {attachedImage ? (
          <div className="chat-image-attachment">
            <img src={attachedImage} alt="待发送图片" />
            <button type="button" aria-label="移除图片" onClick={() => setAttachedImage(null)}>
              <X size={14} />
            </button>
          </div>
        ) : null}
        <textarea
          id="chat-input-textarea"
          name="chat-message"
          ref={textareaRef}
          value={draft}
          placeholder={authenticated ? (selectedSources.length ? "提问或创作内容" : "请先添加并选择来源") : "登录后即可发送"}
          onChange={(event) => setDraft(event.target.value)}
          style={{ minHeight: `${inputHeight}px`, maxHeight: '600px' }}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
        />
        <span>{selectedSources.length} 个来源</span>
        <input
          ref={imageInputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(event) => {
            attachImage(event.target.files?.[0]);
            event.currentTarget.value = "";
          }}
        />
        <button type="button" aria-label="上传图片提问" disabled={!authenticated || busy} onClick={() => imageInputRef.current?.click()}>
          <ImagePlus size={20} />
        </button>
        <button type="button" aria-label="发送消息" disabled={!canSend} onClick={submit}>
          <ArrowRight size={22} />
        </button>
      </div>
    </section>
  );
}

function Overview({ sources, onAsk }: { sources: SourceItem[]; onAsk: (query: string) => void }) {
  const title = sources.length === 1 ? sources[0].title.replace(/\.[^.]+$/, "") : "选中来源知识库";
  const suggestions = ["总结这些资料的核心内容", "有哪些关键事实值得关注？", "基于这些资料生成后续行动清单"];
  return (
    <div className="overview">
      <Bot size={42} />
      <h1>{title}</h1>
      <p>
        {sources.length} 个来源 · {new Date().toLocaleDateString("zh-CN")}
      </p>
      <div className="overview-copy">
        当前知识库已选中 {sources.length} 个来源。你可以直接提问，系统会检索证据、生成回答并附上引用。
      </div>
      <div className="suggestions">
        {suggestions.map((item) => (
          <button type="button" key={item} onClick={() => onAsk(item)}>
            {item}
          </button>
        ))}
      </div>
    </div>
  );
}

function AssistantMessage({
  message,
  typing,
  onTypingComplete,
  onFeedback,
}: {
  message: ChatMessage;
  typing: boolean;
  onTypingComplete: () => void;
  onFeedback: (message: ChatMessage, rating: 1 | -1) => void;
}) {
  const [copied, setCopied] = useState(false);
  const feedbackRating = message.feedbackRating ?? null;
  
  const { text, done } = useTypewriter(message.content, typing);

  const showControls = message.status !== "sending" && (message.status === "failed" || done);
  const className = [
    "assistant-message",
    message.status === "failed" ? "failed" : "",
    message.status === "sending" ? "sending" : "",
    typing && !done ? "typing" : "",
  ]
    .filter(Boolean)
    .join(" ");

  useEffect(() => {
    if (typing && done) {
      onTypingComplete();
    }
  }, [done, onTypingComplete, typing]);

  return (
    <article className={className}>
      <ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeKatex]}>{text}</ReactMarkdown>
      {typing && !done ? <span className="type-caret" aria-hidden="true" /> : null}
      {showControls && message.citations?.length ? <Citations message={message} /> : null}
      {showControls ? (
        <div className="message-actions">
          <button type="button" onClick={() => {
            navigator.clipboard?.writeText(message.content);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
          }}>
            {copied ? <Check size={16} style={{ color: "var(--green)" }} /> : <Copy size={16} />}
          </button>
          <button 
            type="button" 
            onClick={() => {
              onFeedback(message, 1);
            }}
            style={{ color: feedbackRating === 1 ? "var(--green)" : "inherit" }}
          >
            <ThumbsUp size={16} fill={feedbackRating === 1 ? "currentColor" : "none"} />
          </button>
          <button 
            type="button" 
            onClick={() => {
              onFeedback(message, -1);
            }}
            style={{ color: feedbackRating === -1 ? "var(--danger)" : "inherit" }}
          >
            <ThumbsDown size={16} fill={feedbackRating === -1 ? "currentColor" : "none"} />
          </button>
        </div>
      ) : null}
    </article>
  );
}

function useTypewriter(content: string, enabled: boolean) {
  const [visibleLength, setVisibleLength] = useState(() => (enabled ? 0 : content.length));
  const step = useMemo(() => Math.max(2, Math.ceil(content.length / 140)), [content.length]);

  useEffect(() => {
    if (!enabled) {
      setVisibleLength(content.length);
      return;
    }
    setVisibleLength(0);
  }, [content, enabled]);

  useEffect(() => {
    if (!enabled || visibleLength >= content.length) {
      return;
    }
    const timer = window.setTimeout(() => {
      setVisibleLength((length) => Math.min(content.length, length + step));
    }, 16);
    return () => window.clearTimeout(timer);
  }, [content.length, enabled, step, visibleLength]);

  return {
    text: content.slice(0, visibleLength),
    done: visibleLength >= content.length,
  };
}

function Citations({ message }: { message: ChatMessage }) {
  return (
    <div className="citations">
      {message.citations?.map((citation, index) => {
        const images = citationImages(citation);
        return (
          <details key={`${citation.doc_id}-${citation.chunk_index}`}>
            <summary>
              {index + 1}. {formatCitationSummary(citation)}
            </summary>
            <p>{citation.text || citation.text_preview || citation.source_uri}</p>
            {images.length ? (
              <div className="citation-images">
                {images.map((image) => (
                  <figure key={image.url}>
                    <img src={image.url} alt={image.title || "引用图片"} />
                    {image.title ? <figcaption>{image.title}</figcaption> : null}
                  </figure>
                ))}
              </div>
            ) : null}
          </details>
        );
      })}
    </div>
  );
}

function formatCitationSummary(citation: Citation) {
  const title = citationDisplayTitle(citation);
  const location = citationLocation(citation);
  const score = citation.rerank_score ?? citation.score;
  const scoreLabel = citation.rerank_score == null ? "检索分数" : "重排分数";
  const parts = [title, location, `${scoreLabel} ${formatScore(score)}`].filter(Boolean);
  return parts.join(" · ");
}

function formatScore(score: number) {
  return Number.isFinite(score) ? score.toFixed(3) : "未知";
}

function citationDisplayTitle(citation: Citation) {
  const path = metadataString(citation.metadata, "relative_path") || citation.source_uri;
  const filename = filenameFromPath(path);
  if (filename) return filename;
  return citation.title.replace(/\s+p\d+$/i, "");
}

function citationLocation(citation: Citation) {
  const start = metadataNumber(citation.metadata, "page_start") ?? metadataNumber(citation.metadata, "page_no");
  const end = metadataNumber(citation.metadata, "page_end");
  if (start && end && end !== start) return `第 ${start}-${end} 页`;
  if (start) return `第 ${start} 页`;

  const titlePage = citation.title.match(/\bp(\d+)$/i)?.[1];
  if (titlePage) return `第 ${titlePage} 页`;

  const docPage = citation.doc_id.match(/\/page-(\d+)$/i)?.[1];
  return docPage ? `第 ${docPage} 页` : "";
}

function metadataString(metadata: Record<string, unknown>, key: string) {
  const value = metadata[key];
  return typeof value === "string" && value.trim() ? value.trim() : "";
}

function metadataNumber(metadata: Record<string, unknown>, key: string) {
  const value = metadata[key];
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && /^\d+$/.test(value)) return Number(value);
  return null;
}

function citationImages(citation: Citation) {
  const images: Array<{ title: string; url: string }> = [];
  const blocks = citation.metadata.display_blocks;
  if (Array.isArray(blocks)) {
    for (const block of blocks) {
      if (!isRecord(block) || block.type !== "image") continue;
      const url = typeof block.url === "string" ? block.url : "";
      if (!isImageUrl(url)) continue;
      const title = typeof block.title === "string" ? block.title : "";
      images.push({ title, url });
    }
  }

  const imageUrl = metadataString(citation.metadata, "image_url") || metadataString(citation.metadata, "image_uri");
  if (isImageUrl(imageUrl) && images.every((image) => image.url !== imageUrl)) {
    images.push({ title: citationDisplayTitle(citation), url: imageUrl });
  }
  return images;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isImageUrl(value: string) {
  return (
    /^data:image\/[a-z0-9.+-]+;base64,/i.test(value) ||
    /^https?:\/\//i.test(value) ||
    value.startsWith("/source-assets/") ||
    value.startsWith("/api/source-assets/")
  );
}

function filenameFromPath(path: string) {
  const filename = path.split(/[\\/]/).filter(Boolean).pop();
  if (!filename) return "";
  try {
    return decodeURIComponent(filename);
  } catch {
    return filename;
  }
}
