import { ArrowDown, ArrowRight, Bot, Check, ChevronRight, Circle, Copy, ImagePlus, Loader2, MoreVertical, ThumbsDown, ThumbsUp, X } from "lucide-react";
import { memo, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import type { ChatMessage, Citation, RagProgressStage, SourceItem } from "../../lib/types";
import { EmptyState } from "../ui/EmptyState";

const markdownRemarkPlugins = [remarkMath];
const markdownRehypePlugins = [rehypeKatex];
const INITIAL_VISIBLE_MESSAGE_COUNT = 80;
const MESSAGE_VISIBLE_INCREMENT = 80;
const CHAT_IMAGE_MAX_DIMENSION = 1600;
const CHAT_IMAGE_INLINE_LIMIT_BYTES = 512 * 1024;
const CHAT_IMAGE_JPEG_QUALITY = 0.86;

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
  const [imageProcessing, setImageProcessing] = useState(false);
  const [imageAttachError, setImageAttachError] = useState("");
  const [previewImage, setPreviewImage] = useState<{ url: string; title: string } | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [inputHeight, setInputHeight] = useState(46);
  const [showJumpLatest, setShowJumpLatest] = useState(false);
  const [jumpLatestClosing, setJumpLatestClosing] = useState(false);
  const [visibleMessageCount, setVisibleMessageCount] = useState(INITIAL_VISIBLE_MESSAGE_COUNT);
  const wasNearBottomRef = useRef(true);
  const programmaticScrollRef = useRef(false);
  const jumpLatestTimerRef = useRef<number | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const onTypingCompleteRef = useLatestRef(onTypingComplete);
  const onFeedbackRef = useLatestRef(onFeedback);
  const canSend = (draft.trim().length > 0 || Boolean(attachedImage)) && !busy && !imageProcessing;

  const handleTypingComplete = useMemo(() => () => onTypingCompleteRef.current(), [onTypingCompleteRef]);
  const handleFeedback = useMemo(
    () => (message: ChatMessage, rating: 1 | -1) => onFeedbackRef.current(message, rating),
    [onFeedbackRef],
  );
  const visibleMessages = messages.slice(-visibleMessageCount);
  const hiddenMessageCount = Math.max(0, messages.length - visibleMessages.length);

  useEffect(() => {
    function handleClickOutside() {
      setMenuOpen(false);
    }
    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  }, []);

  useEffect(() => {
    return () => {
      if (jumpLatestTimerRef.current !== null) {
        window.clearTimeout(jumpLatestTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const scrollEl = scrollRef.current;
    if (!scrollEl) return;
    if (wasNearBottomRef.current) {
      bottomRef.current?.scrollIntoView({ block: "end" });
    }
    updateJumpLatestState();
  }, [messages, typingMessageId]);

  useEffect(() => {
    setVisibleMessageCount((current) => {
      const minimum = Math.min(INITIAL_VISIBLE_MESSAGE_COUNT, messages.length || INITIAL_VISIBLE_MESSAGE_COUNT);
      return Math.min(Math.max(current, minimum), Math.max(messages.length, INITIAL_VISIBLE_MESSAGE_COUNT));
    });
  }, [messages.length]);

  function isNearBottom() {
    const scrollEl = scrollRef.current;
    if (!scrollEl) return true;
    return scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight < 72;
  }

  function updateJumpLatestState() {
    const nearBottom = isNearBottom();
    wasNearBottomRef.current = nearBottom;
    if (programmaticScrollRef.current) {
      if (nearBottom) {
        programmaticScrollRef.current = false;
        setShowJumpLatest(false);
        setJumpLatestClosing(false);
        if (jumpLatestTimerRef.current !== null) {
          window.clearTimeout(jumpLatestTimerRef.current);
          jumpLatestTimerRef.current = null;
        }
      }
      return;
    }
    setShowJumpLatest(messages.length > 0 && !nearBottom);
  }

  function jumpToLatest() {
    const scrollEl = scrollRef.current;
    if (!scrollEl) return;
    programmaticScrollRef.current = true;
    wasNearBottomRef.current = true;
    setJumpLatestClosing(true);
    scrollEl.scrollTo({ top: scrollEl.scrollHeight, behavior: "smooth" });
    if (jumpLatestTimerRef.current !== null) {
      window.clearTimeout(jumpLatestTimerRef.current);
    }
    jumpLatestTimerRef.current = window.setTimeout(() => {
      scrollEl.scrollTop = scrollEl.scrollHeight;
      programmaticScrollRef.current = false;
      setShowJumpLatest(false);
      setJumpLatestClosing(false);
      jumpLatestTimerRef.current = null;
    }, 900);
  }


  function submit() {
    if (!canSend) return;
    onAsk(draft.trim() || "请根据这张图片检索相关资料并回答。", attachedImage);
    setDraft("");
    setAttachedImage(null);
    setImageAttachError("");
  }

  async function attachImage(file: File | undefined) {
    if (!file || !file.type.startsWith("image/")) return;
    setImageAttachError("");
    setImageProcessing(true);
    try {
      setAttachedImage(await prepareChatImageAttachment(file));
    } catch {
      setAttachedImage(null);
      setImageAttachError("图片处理失败，请换一张图片。");
    } finally {
      setImageProcessing(false);
    }
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
      <div className="chat-scroll" ref={scrollRef} onScroll={updateJumpLatestState}>
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
            {hiddenMessageCount > 0 ? (
              <button
                className="message-list-more"
                type="button"
                onClick={() =>
                  setVisibleMessageCount((current) => Math.min(messages.length, current + MESSAGE_VISIBLE_INCREMENT))
                }
              >
                显示更早消息（{hiddenMessageCount}）
              </button>
            ) : null}
            {visibleMessages.map((message) =>
              message.role === "user" ? (
                <UserMessage key={message.id} message={message} onPreviewImage={setPreviewImage} />
              ) : (
                <AssistantMessage
                  key={message.id}
                  message={message}
                  typing={message.id === typingMessageId && message.status === "done"}
                  onTypingComplete={handleTypingComplete}
                  onFeedback={handleFeedback}
                  onPreviewImage={setPreviewImage}
                />
              ),
            )}
            <div ref={bottomRef} aria-hidden="true" />
          </div>
        )}
      </div>
      {showJumpLatest || jumpLatestClosing ? (
        <div
          className="jump-latest-anchor"
          style={{ bottom: `${inputHeight + (attachedImage ? 102 : 72)}px` }}
        >
          <button
            className={`jump-latest-button${jumpLatestClosing ? " is-hiding" : ""}`}
            type="button"
            aria-label="回到最新对话"
            title="回到最新对话"
            disabled={jumpLatestClosing}
            onClick={jumpToLatest}
          >
            <ArrowDown size={22} />
          </button>
        </div>
      ) : null}
      <div className="chat-input" style={{ position: "relative" }}>
        <div
          className="chat-input-resizer"
          onPointerDown={handlePointerDown}
        />
        {attachedImage ? (
          <div className="chat-image-attachment">
            <button
              type="button"
              className="attachment-preview-button"
              aria-label="预览待发送图片"
              onClick={() => setPreviewImage({ url: attachedImage, title: "待发送图片" })}
            >
              <img src={attachedImage} alt="待发送图片" decoding="async" />
            </button>
            <button className="attachment-remove-button" type="button" aria-label="移除图片" onClick={() => setAttachedImage(null)}>
              <X size={14} />
            </button>
          </div>
        ) : null}
        <textarea
          id="chat-input-textarea"
          name="chat-message"
          ref={textareaRef}
          value={draft}
          placeholder={authenticated ? "提问或创作内容" : "登录后即可发送"}
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
        {imageProcessing ? <small className="chat-input-hint">正在处理图片...</small> : null}
        {imageAttachError ? <small className="chat-input-error">{imageAttachError}</small> : null}
        <input
          ref={imageInputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(event) => {
            void attachImage(event.target.files?.[0]);
            event.currentTarget.value = "";
          }}
        />
        <div className="chat-input-actions">
          <button type="button" aria-label="上传图片提问" disabled={!authenticated || busy || imageProcessing} onClick={() => imageInputRef.current?.click()}>
            <ImagePlus size={20} />
          </button>
          <button type="button" aria-label="发送消息" disabled={!canSend} onClick={submit}>
            <ArrowRight size={22} />
          </button>
        </div>
      </div>
      {previewImage ? <ImagePreview image={previewImage} onClose={() => setPreviewImage(null)} /> : null}
    </section>
  );
}

function useLatestRef<T>(value: T) {
  const ref = useRef(value);
  useEffect(() => {
    ref.current = value;
  }, [value]);
  return ref;
}

async function prepareChatImageAttachment(file: File) {
  const original = await readFileAsDataUrl(file);
  const image = await loadImageElement(original);
  const longestSide = Math.max(image.naturalWidth, image.naturalHeight);
  if (longestSide <= CHAT_IMAGE_MAX_DIMENSION && file.size <= CHAT_IMAGE_INLINE_LIMIT_BYTES) {
    return original;
  }
  const scale = Math.min(1, CHAT_IMAGE_MAX_DIMENSION / longestSide);
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("Canvas is unavailable.");
  }
  context.drawImage(image, 0, 0, width, height);
  return canvas.toDataURL("image/jpeg", CHAT_IMAGE_JPEG_QUALITY);
}

function readFileAsDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") {
        resolve(reader.result);
      } else {
        reject(new Error("Unable to read image."));
      }
    };
    reader.onerror = () => reject(reader.error || new Error("Unable to read image."));
    reader.readAsDataURL(file);
  });
}

function loadImageElement(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("Unable to decode image."));
    image.src = src;
  });
}

const UserMessage = memo(function UserMessage({
  message,
  onPreviewImage,
}: {
  message: ChatMessage;
  onPreviewImage: (image: { url: string; title: string }) => void;
}) {
  return (
    <div className="user-message">
      {message.imageDataUrl ? (
        <button
          className="message-image-thumb"
          type="button"
          aria-label="查看发送的图片"
          onClick={() => onPreviewImage({ url: message.imageDataUrl!, title: "发送的图片" })}
        >
          <img src={message.imageDataUrl} alt="发送的图片" loading="lazy" decoding="async" />
        </button>
      ) : null}
      <div>{message.content}</div>
    </div>
  );
});

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

const AssistantMessage = memo(function AssistantMessage({
  message,
  typing,
  onTypingComplete,
  onFeedback,
  onPreviewImage,
}: {
  message: ChatMessage;
  typing: boolean;
  onTypingComplete: () => void;
  onFeedback: (message: ChatMessage, rating: 1 | -1) => void;
  onPreviewImage: (image: { url: string; title: string }) => void;
}) {
  const [copied, setCopied] = useState(false);
  const [ragOpen, setRagOpen] = useState(false);
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
      {showControls && message.ragProgress?.length ? (
        <button
          className={`rag-summary-toggle${ragOpen ? " is-open" : ""}`}
          type="button"
          aria-expanded={ragOpen}
          onClick={() => setRagOpen((open) => !open)}
        >
          <span>{formatRagThoughtLabel(message.ragProgress)}</span>
          <ChevronRight size={15} />
        </button>
      ) : null}
      {showControls && ragOpen && message.ragProgress?.length ? (
        <div className="rag-inline-trace">
          <RagProgressTimeline stages={message.ragProgress} />
        </div>
      ) : null}
      {message.status === "sending" && message.ragProgress?.length ? (
        <RagProgressTimeline stages={message.ragProgress} />
      ) : (
        <ReactMarkdown remarkPlugins={markdownRemarkPlugins} rehypePlugins={markdownRehypePlugins}>{text}</ReactMarkdown>
      )}
      {typing && !done ? <span className="type-caret" aria-hidden="true" /> : null}
      {showControls && message.citations?.length ? <Citations message={message} onPreviewImage={onPreviewImage} /> : null}
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
});

const RagProgressTimeline = memo(function RagProgressTimeline({ stages }: { stages: RagProgressStage[] }) {
  const activeStage =
    stages.find((stage) => stage.status === "active") ??
    stages.find((stage) => stage.status === "pending") ??
    stages[stages.length - 1];
  return (
    <div className="rag-progress" aria-live="polite">
      <div className="rag-progress-header">
        <span>RAG 调用链</span>
        <strong>{activeStage?.label || "准备回答"}</strong>
      </div>
      <div className="rag-progress-track">
        {stages.map((stage) => {
          const meta = formatStageMeta(stage);
          return (
            <div className={`rag-progress-step ${stage.status}`} key={stage.stage}>
            <span className="rag-progress-node" aria-hidden="true">
              {stage.status === "done" ? (
                <Check size={14} />
              ) : stage.status === "active" ? (
                <Loader2 size={14} />
              ) : (
                <Circle size={10} />
              )}
            </span>
            <span className="rag-progress-copy">
              <span className="rag-progress-title">
                {stage.label}
                {meta ? <em>{meta}</em> : null}
              </span>
              <span className="rag-progress-detail">{stage.detail}</span>
            </span>
            </div>
          );
        })}
      </div>
    </div>
  );
});

function formatStageMeta(stage: RagProgressStage) {
  const parts: string[] = [];
  if (typeof stage.latency_ms === "number") {
    parts.push(`${Math.max(1, Math.round(stage.latency_ms))}ms`);
  }
  if (typeof stage.candidate_count === "number") {
    parts.push(`${stage.candidate_count} 候选`);
  } else if (typeof stage.reranked_count === "number") {
    parts.push(`${stage.reranked_count} 重排`);
  } else if (typeof stage.context_count === "number") {
    parts.push(`${stage.context_count} 证据`);
  }
  return parts.join(" · ");
}

function totalRagLatency(stages: RagProgressStage[]) {
  return stages.reduce((sum, stage) => sum + (typeof stage.latency_ms === "number" ? stage.latency_ms : 0), 0);
}

function formatRagThoughtLabel(stages: RagProgressStage[]) {
  const duration = formatRagDuration(totalRagLatency(stages));
  return duration ? `已思考 ${duration}` : "已思考";
}

function formatRagDuration(ms: number) {
  if (!Number.isFinite(ms) || ms <= 0) {
    return "";
  }
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(ms >= 10_000 ? 0 : 1)}s`;
  }
  return `${Math.max(1, Math.round(ms))}ms`;
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

function Citations({
  message,
  onPreviewImage,
}: {
  message: ChatMessage;
  onPreviewImage: (image: { url: string; title: string }) => void;
}) {
  return (
    <div className="citations">
      {message.citations?.map((citation, index) => {
        const images = citationImages(citation);
        return (
          <details key={`${citation.doc_id}-${citation.chunk_index}`} open={images.length > 0}>
            <summary>
              {index + 1}. {formatCitationSummary(citation)}
            </summary>
            <p>{citation.text || citation.text_preview || citation.source_uri}</p>
            {images.length ? (
              <div className="citation-images">
                {images.map((image) => (
                  <figure key={image.url}>
                    <button
                      className="citation-image-thumb"
                      type="button"
                      aria-label={`查看${image.title || "引用图片"}`}
                      onClick={() => onPreviewImage({ url: image.url, title: image.title || "引用图片" })}
                    >
                      <img src={image.url} alt={image.title || "引用图片"} loading="lazy" decoding="async" />
                    </button>
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

function ImagePreview({ image, onClose }: { image: { url: string; title: string }; onClose: () => void }) {
  return (
    <div className="modal-backdrop image-preview-backdrop" role="presentation" onMouseDown={onClose}>
      <div
        className="image-preview-dialog"
        role="dialog"
        aria-modal="true"
        aria-label={image.title}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <button className="close-button" type="button" aria-label="关闭图片预览" onClick={onClose}>
          <X size={18} />
        </button>
        <img src={image.url} alt={image.title} decoding="async" />
      </div>
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
