import { ArrowRight, Bot, Copy, MoreVertical, ThumbsDown, ThumbsUp } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import type { ChatMessage, SourceItem } from "../../lib/types";
import { EmptyState } from "../ui/EmptyState";

type Props = {
  messages: ChatMessage[];
  selectedSources: SourceItem[];
  busy: boolean;
  conversationTitle: string;
  typingMessageId: string | null;
  onTypingComplete: () => void;
  onAsk: (query: string) => void;
  onFeedback: (message: ChatMessage, rating: 1 | -1) => void;
  onDeleteConversation: () => void;
};

export function ChatPanel({
  messages,
  selectedSources,
  busy,
  conversationTitle,
  typingMessageId,
  onTypingComplete,
  onAsk,
  onFeedback,
  onDeleteConversation,
}: Props) {
  const [draft, setDraft] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const canSend = draft.trim().length > 0 && selectedSources.length > 0 && !busy;

  function submit() {
    if (!canSend) return;
    onAsk(draft.trim());
    setDraft("");
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
              onClick={() => setMenuOpen((open) => !open)}
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
      <div className="chat-input">
        <textarea
          value={draft}
          placeholder={selectedSources.length ? "提问或创作内容" : "请先添加并选择来源"}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
        />
        <span>{selectedSources.length} 个来源</span>
        <button type="button" disabled={!canSend} onClick={submit}>
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
      <ReactMarkdown>{text}</ReactMarkdown>
      {typing && !done ? <span className="type-caret" aria-hidden="true" /> : null}
      {showControls && message.citations?.length ? <Citations message={message} /> : null}
      {showControls ? (
        <div className="message-actions">
          <button type="button" onClick={() => navigator.clipboard?.writeText(message.content)}>
            <Copy size={16} />
          </button>
          <button type="button" onClick={() => onFeedback(message, 1)}>
            <ThumbsUp size={16} />
          </button>
          <button type="button" onClick={() => onFeedback(message, -1)}>
            <ThumbsDown size={16} />
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
      {message.citations?.map((citation, index) => (
        <details key={`${citation.doc_id}-${citation.chunk_index}`}>
          <summary>
            {index + 1}. {citation.title} · chunk {citation.chunk_index}
          </summary>
          <p>{citation.text_preview || citation.source_uri}</p>
        </details>
      ))}
    </div>
  );
}
