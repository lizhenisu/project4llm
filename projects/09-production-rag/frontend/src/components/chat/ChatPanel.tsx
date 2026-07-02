import { ArrowDown, ArrowRight, Bot, Check, ChevronRight, Circle, Copy, History, ImagePlus, Loader2, Maximize, Minimize, MoreHorizontal, PencilLine, Plus, ThumbsDown, ThumbsUp, Trash2, X } from "lucide-react";
import { memo, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";
import type { ChatMessage, Citation, ConversationListItem, RagProgressStage, SourceItem } from "../../lib/types";
import { EmptyState } from "../ui/EmptyState";
import { ProtectedImage } from "../ui/ProtectedImage";

const markdownRemarkPlugins = [remarkMath];
const markdownRehypePlugins = [rehypeKatex];
const INITIAL_VISIBLE_MESSAGE_COUNT = 80;
const MESSAGE_VISIBLE_INCREMENT = 80;
const CHAT_IMAGE_MAX_DIMENSION = 1600;
const CHAT_IMAGE_INLINE_LIMIT_BYTES = 512 * 1024;
const CHAT_IMAGE_JPEG_QUALITY = 0.86;

type Props = {
  messages: ChatMessage[];
  conversations: ConversationListItem[];
  activeConversationId: string | null;
  chromeCollapsed: boolean;
  selectedSources: SourceItem[];
  authenticated: boolean;
  busy: boolean;
  conversationTitle: string;
  typingMessageId: string | null;
  assetToken?: string;
  assetApiBaseUrl?: string;
  onTypingComplete: () => void;
  onAsk: (query: string, imageDataUrl?: string | null) => void;
  onFeedback: (message: ChatMessage, rating: 1 | -1) => void;
  onNewConversation: () => void;
  onOpenConversation: (conversationId: string) => Promise<void>;
  onRenameConversation: (conversationId: string, title: string) => Promise<void>;
  onDeleteConversation: (conversationId: string) => Promise<void>;
  onToggleChrome: () => void;
};

export function ChatPanel({
  messages,
  conversations,
  activeConversationId,
  chromeCollapsed,
  selectedSources,
  authenticated,
  busy,
  conversationTitle,
  typingMessageId,
  assetToken,
  assetApiBaseUrl,
  onTypingComplete,
  onAsk,
  onFeedback,
  onNewConversation,
  onOpenConversation,
  onRenameConversation,
  onDeleteConversation,
  onToggleChrome,
}: Props) {
  const [draft, setDraft] = useState("");
  const [attachedImage, setAttachedImage] = useState<string | null>(null);
  const [imageProcessing, setImageProcessing] = useState(false);
  const [imageAttachError, setImageAttachError] = useState("");
  const [previewImage, setPreviewImage] = useState<{ url: string; title: string } | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [conversationMenuId, setConversationMenuId] = useState<string | null>(null);
  const [conversationMenuPosition, setConversationMenuPosition] = useState<{ left: number; top: number } | null>(null);
  const [editingConversationId, setEditingConversationId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState("");
  const [historyBusyId, setHistoryBusyId] = useState<string | null>(null);
  const [historyError, setHistoryError] = useState("");
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
  const renameSubmittingRef = useRef<string | null>(null);
  const renameCancelledRef = useRef<string | null>(null);
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
      setConversationMenuId(null);
      setConversationMenuPosition(null);
    }
    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  }, []);

  useEffect(() => {
    if (!conversationMenuId) return;
    const closeFloatingMenu = () => {
      setConversationMenuId(null);
      setConversationMenuPosition(null);
    };
    window.addEventListener("resize", closeFloatingMenu);
    document.addEventListener("scroll", closeFloatingMenu, true);
    return () => {
      window.removeEventListener("resize", closeFloatingMenu);
      document.removeEventListener("scroll", closeFloatingMenu, true);
    };
  }, [conversationMenuId]);

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

  async function openConversation(conversationId: string) {
    setHistoryError("");
    setHistoryBusyId(conversationId);
    try {
      await onOpenConversation(conversationId);
      setHistoryOpen(false);
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "加载历史对话失败");
    } finally {
      setHistoryBusyId(null);
    }
  }

  async function submitConversationRename(conversationId: string, currentTitle: string) {
    if (renameSubmittingRef.current === conversationId) return;
    const normalizedTitle = editTitle.trim();
    if (!normalizedTitle || normalizedTitle === currentTitle) {
      setEditingConversationId(null);
      setEditTitle("");
      return;
    }
    renameSubmittingRef.current = conversationId;
    setHistoryError("");
    setHistoryBusyId(conversationId);
    try {
      await onRenameConversation(conversationId, normalizedTitle);
      setEditingConversationId(null);
      setEditTitle("");
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "重命名对话失败");
    } finally {
      renameSubmittingRef.current = null;
      setHistoryBusyId(null);
    }
  }

  async function removeConversation(conversationId: string) {
    setHistoryError("");
    setHistoryBusyId(conversationId);
    try {
      await onDeleteConversation(conversationId);
      setConversationMenuId(null);
      setConversationMenuPosition(null);
      setEditingConversationId(null);
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "删除对话失败");
    } finally {
      setHistoryBusyId(null);
    }
  }

  function toggleConversationMenu(event: React.MouseEvent<HTMLButtonElement>, conversationId: string) {
    event.stopPropagation();
    if (conversationMenuId === conversationId) {
      setConversationMenuId(null);
      setConversationMenuPosition(null);
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const menuWidth = 132;
    const menuHeight = 78;
    const viewportGap = 8;
    const itemGap = 5;
    const left = Math.min(
      window.innerWidth - menuWidth - viewportGap,
      Math.max(viewportGap, rect.right - menuWidth),
    );
    const opensDownward = rect.bottom + itemGap + menuHeight <= window.innerHeight - viewportGap;
    const top = opensDownward
      ? rect.bottom + itemGap
      : Math.max(viewportGap, rect.top - menuHeight - itemGap);
    setConversationMenuPosition({ left, top });
    setConversationMenuId(conversationId);
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
        <div className="chat-title-row">
          <h2 title={conversationTitle}>对话</h2>
          <button
            className="icon-button"
            type="button"
            aria-label={historyOpen ? "关闭历史对话" : "打开历史对话"}
            title={historyOpen ? "关闭历史对话" : "历史对话"}
            aria-expanded={historyOpen}
            onClick={() => {
              setConversationMenuId(null);
              setConversationMenuPosition(null);
              setHistoryOpen((open) => !open);
            }}
          >
            <History size={18} />
          </button>
        </div>
        <button
          className="icon-button chat-chrome-toggle"
          type="button"
          aria-label={chromeCollapsed ? "展开顶部栏和状态栏" : "折叠顶部栏和状态栏"}
          title={chromeCollapsed ? "展开顶部栏和状态栏" : "折叠顶部栏和状态栏"}
          aria-expanded={!chromeCollapsed}
          onClick={onToggleChrome}
        >
          {chromeCollapsed ? (
            <Minimize size={18} aria-hidden="true" />
          ) : (
            <Maximize size={18} aria-hidden="true" />
          )}
        </button>
      </div>
      <button
        type="button"
        className={`conversation-history-backdrop ${historyOpen ? "open" : ""}`}
        aria-label="关闭历史对话"
        tabIndex={historyOpen ? 0 : -1}
        onClick={() => {
          setConversationMenuId(null);
          setConversationMenuPosition(null);
          setHistoryOpen(false);
        }}
      />
      <aside className={`conversation-history-drawer ${historyOpen ? "open" : ""}`} aria-label="历史对话">
        <div className="conversation-history-header">
          <div>
            <strong>历史对话</strong>
            <small>{conversations.length} 条记录</small>
          </div>
          <button
            className="icon-button"
            type="button"
            aria-label="关闭历史对话"
            onClick={() => {
              setConversationMenuId(null);
              setConversationMenuPosition(null);
              setHistoryOpen(false);
            }}
          >
            <X size={18} />
          </button>
        </div>
        {historyError ? <p className="conversation-history-error" role="alert">{historyError}</p> : null}
        <div className="conversation-history-create">
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              onNewConversation();
              setHistoryOpen(false);
            }}
          >
            <Plus size={16} />
            开启新对话
          </button>
        </div>
        <div className="conversation-history-list">
          {conversations.length === 0 ? (
            <div className="conversation-history-empty">还没有历史对话</div>
          ) : (
            conversations.map((conversation) => {
              const editing = editingConversationId === conversation.id;
              const itemBusy = historyBusyId === conversation.id;
              return (
                <article
                  key={conversation.id}
                  className={`conversation-history-item ${activeConversationId === conversation.id ? "active" : ""}`}
                >
                  {editing ? (
                    <form
                      className="conversation-history-rename"
                      onSubmit={(event) => {
                        event.preventDefault();
                        void submitConversationRename(conversation.id, conversation.title);
                      }}
                    >
                      <input
                        value={editTitle}
                        aria-label={`重命名${conversation.title}`}
                        maxLength={200}
                        autoFocus
                        disabled={itemBusy}
                        onChange={(event) => setEditTitle(event.target.value)}
                        onBlur={() => {
                          if (renameCancelledRef.current === conversation.id) {
                            renameCancelledRef.current = null;
                            return;
                          }
                          void submitConversationRename(conversation.id, conversation.title);
                        }}
                        onKeyDown={(event) => {
                          if (event.key === "Escape") {
                            renameCancelledRef.current = conversation.id;
                            setEditingConversationId(null);
                            setEditTitle("");
                          }
                        }}
                      />
                    </form>
                  ) : (
                    <button
                      type="button"
                      className="conversation-history-main"
                      disabled={itemBusy}
                      onClick={() => void openConversation(conversation.id)}
                    >
                      <strong>{conversation.title}</strong>
                      <span>{formatConversationTime(conversation.updated_at)} · {conversation.message_count} 条消息</span>
                    </button>
                  )}
                  <div className="conversation-history-menu">
                    <button
                      className="row-icon"
                      type="button"
                      aria-label={`管理对话：${conversation.title}`}
                      disabled={itemBusy}
                      onClick={(event) => toggleConversationMenu(event, conversation.id)}
                    >
                      {itemBusy ? <Loader2 className="spin" size={16} /> : <MoreHorizontal size={17} />}
                    </button>
                    {conversationMenuId === conversation.id && conversationMenuPosition && typeof document !== "undefined"
                      ? createPortal(
                          <div
                            className="conversation-history-popover"
                            role="menu"
                            style={conversationMenuPosition}
                            onClick={(event) => event.stopPropagation()}
                          >
                            <button
                              type="button"
                              role="menuitem"
                              onClick={() => {
                                renameCancelledRef.current = null;
                                setEditingConversationId(conversation.id);
                                setEditTitle(conversation.title);
                                setConversationMenuId(null);
                                setConversationMenuPosition(null);
                              }}
                            >
                              <PencilLine size={15} />
                              重命名
                            </button>
                            <button
                              type="button"
                              role="menuitem"
                              className="danger"
                              onClick={() => void removeConversation(conversation.id)}
                            >
                              <Trash2 size={15} />
                              删除
                            </button>
                          </div>,
                          document.body,
                        )
                      : null}
                  </div>
                </article>
              );
            })
          )}
        </div>
      </aside>
      <div className="chat-scroll" ref={scrollRef} onScroll={updateJumpLatestState}>
        {messages.length === 0 ? (
          selectedSources.length === 0 ? (
            <EmptyState
              icon={<Bot size={34} />}
              title="直接开始对话"
              text="无需选择文档也可以提问；选择来源后，回答会结合知识库内容。"
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
                  assetToken={assetToken}
                  assetApiBaseUrl={assetApiBaseUrl}
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
      {previewImage ? (
        <ImagePreview
          image={previewImage}
          assetToken={assetToken}
          assetApiBaseUrl={assetApiBaseUrl}
          onClose={() => setPreviewImage(null)}
        />
      ) : null}
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

function formatConversationTime(timestamp: number) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(timestamp));
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
  assetToken,
  assetApiBaseUrl,
}: {
  message: ChatMessage;
  typing: boolean;
  onTypingComplete: () => void;
  onFeedback: (message: ChatMessage, rating: 1 | -1) => void;
  onPreviewImage: (image: { url: string; title: string }) => void;
  assetToken?: string;
  assetApiBaseUrl?: string;
}) {
  const [copied, setCopied] = useState(false);
  const [ragOpen, setRagOpen] = useState(false);
  const feedbackRating = message.feedbackRating ?? null;
  const interrupted = message.status === "sending"
    && message.content === "连接已中断，刷新页面后将自动恢复回答。";
  
  const { text, done } = useTypewriter(message.content, typing);

  const showControls = message.status !== "sending" && (message.status === "failed" || done);
  const className = [
    "assistant-message",
    message.status === "failed" ? "failed" : "",
    message.status === "sending" ? "sending" : "",
    interrupted ? "interrupted" : "",
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
      {message.status === "sending" && message.ragProgress?.length && !interrupted ? (
        <RagProgressTimeline stages={message.ragProgress} />
      ) : (
        <ReactMarkdown remarkPlugins={markdownRemarkPlugins} rehypePlugins={markdownRehypePlugins}>{text}</ReactMarkdown>
      )}
      {typing && !done ? <span className="type-caret" aria-hidden="true" /> : null}
      {showControls && message.citations?.length ? (
        <Citations
          message={message}
          onPreviewImage={onPreviewImage}
          assetToken={assetToken}
          assetApiBaseUrl={assetApiBaseUrl}
        />
      ) : null}
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
  assetToken,
  assetApiBaseUrl,
}: {
  message: ChatMessage;
  onPreviewImage: (image: { url: string; title: string }) => void;
  assetToken?: string;
  assetApiBaseUrl?: string;
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
                      <ProtectedImage
                        src={image.url}
                        token={assetToken}
                        apiBaseUrl={assetApiBaseUrl}
                        alt={image.title || "引用图片"}
                        loading="lazy"
                        decoding="async"
                      />
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

function ImagePreview({
  image,
  assetToken,
  assetApiBaseUrl,
  onClose,
}: {
  image: { url: string; title: string };
  assetToken?: string;
  assetApiBaseUrl?: string;
  onClose: () => void;
}) {
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
        <ProtectedImage
          src={image.url}
          token={assetToken}
          apiBaseUrl={assetApiBaseUrl}
          alt={image.title}
          decoding="async"
        />
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
