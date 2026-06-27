import { FileText, LoaderCircle, Plus, Sparkles, Trash2, Upload, X , MoreVertical } from "lucide-react";
import { useRef, useState, useEffect } from "react";
import { createPortal } from "react-dom";
import type { SourceContent, SourceItem } from "../../lib/types";
import { EmptyState } from "../ui/EmptyState";

const INITIAL_VISIBLE_SOURCE_COUNT = 80;
const SOURCE_VISIBLE_INCREMENT = 80;
const PROCESSING_STALE_WARNING_MS = 30 * 60 * 1000;

type Props = {
  sources: SourceItem[];
  onSourcesChange: (sources: SourceItem[]) => void;
  onUpload: (file: File) => void;
  onDeleteSource: (source: SourceItem) => void;
  onRetrySource?: (source: SourceItem) => void;
  onRenameSource?: (source: SourceItem, newTitle: string) => void;
  onOpenSource: (source: SourceItem) => void;
  activeContent: SourceContent | null;
  contentLoading: boolean;
  contentError: string;
  onCloseContent: () => void;
};

export function SourcePanel({
  sources,
  onSourcesChange,
  onUpload,
  onDeleteSource,
  onRetrySource,
  onRenameSource,
  onOpenSource,
  activeContent,
  contentLoading,
  contentError,
  onCloseContent,
}: Props) {
  const [uploadOpen, setUploadOpen] = useState(false);

  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const [menuPosition, setMenuPosition] = useState<{ top: number; right: number } | null>(null);
  const [editingSourceId, setEditingSourceId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [sourceToDelete, setSourceToDelete] = useState<SourceItem | null>(null);
  const [visibleSourceCount, setVisibleSourceCount] = useState(INITIAL_VISIBLE_SOURCE_COUNT);

  
  useEffect(() => {
    function handleClickOutside() {
      setMenuOpenId(null);
    }
    document.addEventListener("click", handleClickOutside);
    return () => document.removeEventListener("click", handleClickOutside);
  }, []);

  useEffect(() => {
    setVisibleSourceCount((current) => {
      const minimum = Math.min(INITIAL_VISIBLE_SOURCE_COUNT, sources.length || INITIAL_VISIBLE_SOURCE_COUNT);
      return Math.min(Math.max(current, minimum), Math.max(sources.length, INITIAL_VISIBLE_SOURCE_COUNT));
    });
  }, [sources.length]);

  const readySources = sources.filter((source) => source.status === "ready");
  const allSelected = readySources.length > 0 && readySources.every((source) => source.selected);
  const visibleSources = sources.slice(0, visibleSourceCount);
  const hiddenSourceCount = Math.max(0, sources.length - visibleSources.length);

  function toggleAll() {
    onSourcesChange(
      sources.map((source) => (source.status === "ready" ? { ...source, selected: !allSelected } : source)),
    );
  }

  
  function handleSaveTitle(source: SourceItem) {
    const newTitle = editingTitle.trim();
    if (newTitle && newTitle !== source.title && onRenameSource) {
      onRenameSource(source, newTitle);
    }
    setEditingSourceId(null);
  }

  function toggle(docId: string) {
    onSourcesChange(
      sources.map((source) =>
        sourceInstanceKey(source) === docId && source.status === "ready" ? { ...source, selected: !source.selected } : source,
      ),
    );
  }

  return (
    <aside className="panel source-panel">
      <div className="panel-header">
        <h2>来源</h2>
      </div>
      <button className="add-source-button" type="button" onClick={() => setUploadOpen(true)}>
        <Plus size={17} />
        添加来源
      </button>
      {sources.length === 0 ? (
        <EmptyState
          icon={<FileText size={28} />}
          title="已保存的来源将显示在此处"
          text="点击上方的添加来源上传 PDF、Markdown、TXT、CSV、TSV、HTML 等文件。"
        />
      ) : (
        <div className="source-list">
          <label className="source-select-all">
            <span>全选</span>
            <input type="checkbox" checked={allSelected} onChange={toggleAll} />
          </label>
          {visibleSources.map((source) => {
            const activeTask = source.status === "uploading" || source.status === "queued" || source.status === "processing";
            const progressDetail = sourceProgressDetail(source);
            const staleTask = source.status === "processing" && sourceStatusAgeMs(source) >= PROCESSING_STALE_WARNING_MS;
            const sourceKey = sourceInstanceKey(source);
            const isEditing = editingSourceId === sourceKey;
            return (
            <div className={`source-row status-${source.status}${activeTask ? " is-active-task" : ""}${staleTask ? " is-stale-task" : ""}${isEditing ? " is-editing" : ""}`} key={sourceKey}>
              <FileText className="file-type-icon" size={20} />
              
              <div className={`source-title${isEditing ? " is-editing" : ""}`}>
                {isEditing ? (
                  <input
                    className="inline-title-input"
                    autoFocus
                    value={editingTitle}
                    onChange={(e) => setEditingTitle(e.target.value)}
                    onBlur={() => handleSaveTitle(source)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleSaveTitle(source);
                      if (e.key === "Escape") setEditingSourceId(null);
                    }}
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <button
                    type="button"
                    style={{ background: "none", border: "none", padding: 0, textAlign: "left", cursor: "pointer", width: "100%", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                    disabled={source.status !== "ready"}
                    onClick={() => onOpenSource(source)}
                  >
                    <span>{source.title}</span>
                  </button>
                )}
                <small>
                  {source.source_type} · {source.chunk_count} chunks
                  {source.status !== "ready" ? ` · ${sourceStatusLabel(source.status)}` : ""}
                </small>
                {progressDetail ? (
                  <small className={`source-progress${staleTask ? " is-stale" : ""}`}>{progressDetail}</small>
                ) : null}
                {source.error ? <small className="error-text">{source.error}</small> : null}
              </div>

              <div className="source-menu-container">
                <button
                  type="button"
                  className="row-icon row-icon-more"
                  style={{ background: "none", border: "none", cursor: "pointer", padding: "4px" }}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (menuOpenId === sourceKey) {
                      setMenuOpenId(null);
                    } else {
                      const rect = e.currentTarget.getBoundingClientRect();
                      setMenuPosition({
                        top: rect.bottom + 4,
                        right: document.documentElement.clientWidth - rect.right,
                      });
                      setMenuOpenId(sourceKey);
                    }
                  }}
                >
                  <MoreVertical size={18} />
                </button>
                {menuOpenId === sourceKey && menuPosition && createPortal(
                  <>
                  <div className="dropdown-backdrop" onMouseDown={() => setMenuOpenId(null)} />
                  <div className="dropdown-menu" style={{ top: menuPosition.top, right: menuPosition.right }} onMouseDown={(e) => e.stopPropagation()}>
                    {source.status === "failed" && source.retryable && onRetrySource ? (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setMenuOpenId(null);
                          onRetrySource(source);
                        }}
                      >
                        重新处理
                      </button>
                    ) : null}
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setMenuOpenId(null);
                        setEditingSourceId(sourceKey);
                        setEditingTitle(source.title);
                      }}
                    >
                      重命名
                    </button>
                    <button
                      type="button"
                      className="danger"
                      onClick={(e) => {
                        e.stopPropagation();
                        setMenuOpenId(null);
                        setSourceToDelete(source);
                      }}
                    >
                      移除
                    </button>
                  </div>
                  </>,
                  document.body
                )}
              </div>

              <input
                type="checkbox"
                checked={Boolean(source.selected)}
                disabled={source.status !== "ready"}
                onChange={() => toggle(sourceKey)}
              />
            </div>
          );
          })}
          {hiddenSourceCount > 0 ? (
            <button
              className="source-list-more"
              type="button"
              onClick={() =>
                setVisibleSourceCount((current) => Math.min(sources.length, current + SOURCE_VISIBLE_INCREMENT))
              }
            >
              显示更多来源（{hiddenSourceCount}）
            </button>
          ) : null}
        </div>
      )}

      {sourceToDelete && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setSourceToDelete(null)}>
          <div className="delete-dialog" role="dialog" aria-modal="true" onMouseDown={(e) => e.stopPropagation()}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
              <h3 style={{ margin: 0, fontSize: "16px" }}>确认移除来源？</h3>
              <button className="icon-button" type="button" onClick={() => setSourceToDelete(null)}>
                <X size={18} />
              </button>
            </div>
            <p style={{ margin: "0 0 20px 0", fontSize: "14px", color: "var(--text-muted)", lineHeight: 1.5 }}>
              您确定要移除来源 "{sourceToDelete.title}" 吗？此操作不可撤销。
            </p>
            <div style={{ display: "flex", justifyContent: "flex-end", gap: "12px" }}>
              <button
                type="button"
                style={{ padding: "6px 12px", border: "1px solid var(--line)", background: "var(--surface)", borderRadius: "var(--radius)", cursor: "pointer" }}
                onClick={() => setSourceToDelete(null)}
              >
                取消
              </button>
              <button
                type="button"
                style={{ padding: "6px 12px", border: "none", background: "var(--danger)", color: "white", borderRadius: "var(--radius)", cursor: "pointer" }}
                onClick={() => {
                  onDeleteSource(sourceToDelete);
                  setSourceToDelete(null);
                }}
              >
                确认移除
              </button>
            </div>
          </div>
        </div>
      )}
      {uploadOpen ? <SourceUploadDialog onClose={() => setUploadOpen(false)} onUpload={onUpload} /> : null}
      {activeContent ? (
        <SourceReader
          content={activeContent}
          loading={contentLoading}
          error={contentError}
          onClose={onCloseContent}
        />
      ) : null}
    </aside>
  );
}

function sourceInstanceKey(source: SourceItem) {
  return `${source.doc_id}::${source.doc_version}`;
}

function sourceStatusLabel(status: SourceItem["status"]) {
  if (status === "queued") return "排队中";
  if (status === "processing") return "处理中";
  if (status === "uploading") return "上传中";
  if (status === "failed") return "失败";
  return "已就绪";
}

function sourceProgressDetail(source: SourceItem) {
  const attemptCount = Math.max(0, source.attempt_count || 0);
  if (source.status === "queued" && attemptCount > 0 && (source.next_attempt_at || 0) > 0) {
    return `等待自动重试 · 已尝试 ${attemptCount} 次`;
  }
  if (source.status === "failed" && source.dead_lettered) {
    return `已停止自动重试 · 共尝试 ${attemptCount} 次，可选择重新处理`;
  }
  if (source.status === "processing") {
    const age = sourceStatusAgeMs(source);
    if (age >= PROCESSING_STALE_WARNING_MS) {
      const attemptDetail = attemptCount > 0 ? `（第 ${attemptCount} 次尝试）` : "";
      return `处理时间已超过 30 分钟，疑似停滞，系统将自动尝试恢复${attemptDetail}`;
    }
    if (attemptCount > 1) return `第 ${attemptCount} 次处理尝试`;
  }
  return "";
}

function sourceStatusAgeMs(source: SourceItem) {
  const timestamp = source.status === "queued" ? source.created_at : source.updated_at;
  if (!timestamp || !Number.isFinite(timestamp)) return 0;
  return Math.max(0, Date.now() - timestamp);
}

function SourceReader({
  content,
  loading,
  error,
  onClose,
}: {
  content: SourceContent;
  loading: boolean;
  error: string;
  onClose: () => void;
}) {
  return (
    <div className="source-reader-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className="source-reader"
        role="dialog"
        aria-modal="true"
        aria-label={`${content.title} 原始内容`}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="source-reader-header">
          <h2>来源</h2>
          <button className="icon-button" type="button" title="关闭" onClick={onClose}>
            <X size={18} />
          </button>
        </header>
        <div className="source-reader-body">
          <h1>{content.title}</h1>
          <section className="source-guide-card">
            <div className="source-guide-title">
              <Sparkles size={20} />
              <h2>来源指南</h2>
              {loading ? <LoaderCircle className="spin-icon" size={18} /> : null}
            </div>
            {error ? <p className="error-text">{error}</p> : <p>{content.guide}</p>}
            {content.tags.length > 0 ? (
              <div className="source-tags">
                {content.tags.map((tag) => (
                  <span key={tag}>{tag}</span>
                ))}
              </div>
            ) : null}
          </section>
          <article className="source-document-text">
            {content.blocks?.length ? (
              content.blocks.map((block, index) =>
                block.type === "image" && block.url ? (
                  <figure className="source-document-image" key={`${index}-${block.url.slice(0, 32)}`}>
                    <img
                      src={block.url}
                      alt={block.title || block.page || "Document image"}
                      loading="lazy"
                      decoding="async"
                    />
                    {block.page || block.title ? (
                      <figcaption>{[block.page, block.title].filter(Boolean).join(" · ")}</figcaption>
                    ) : null}
                  </figure>
                ) : block.text ? (
                  <div className="source-document-block" key={`${index}-${block.text.slice(0, 16)}`}>
                    {block.text.split(/\n{2,}/).map((paragraph, paragraphIndex) => (
                      <p key={`${paragraphIndex}-${paragraph.slice(0, 16)}`}>{paragraph}</p>
                    ))}
                  </div>
                ) : null,
              )
            ) : content.text ? (
              content.text.split(/\n{2,}/).map((block, index) => <p key={`${index}-${block.slice(0, 16)}`}>{block}</p>)
            ) : loading ? (
              <p className="source-loading">正在加载原始内容...</p>
            ) : (
              <p className="source-loading">暂无可展示的原始内容。</p>
            )}
          </article>
        </div>
      </section>
    </div>
  );
}

function SourceUploadDialog({ onClose, onUpload }: { onClose: () => void; onUpload: (file: File) => void }) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);

  function handleFiles(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    onUpload(file);
    onClose();
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <div
        className="upload-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="source-upload-dialog-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <button className="close-button" type="button" onClick={onClose}>
          ×
        </button>
        <h2 id="source-upload-dialog-title">添加来源</h2>
        <p>上传文件后，系统会解析、切分并写入知识库。</p>
        <div
          className={`drop-zone ${dragging ? "dragging" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            handleFiles(event.dataTransfer.files);
          }}
        >
          <Upload size={28} />
          <strong>或拖放文件</strong>
          <span>PDF、Markdown、TXT、HTML、CSV、TSV · 最大 100MB</span>
          <div className="upload-actions">
            <button type="button" onClick={() => inputRef.current?.click()}>
              上传文件
            </button>
          </div>
          <input ref={inputRef} type="file" hidden onChange={(event) => handleFiles(event.target.files)} />
        </div>
      </div>
    </div>
  );
}
