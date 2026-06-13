import { FileText, LoaderCircle, Plus, Sparkles, Trash2, Upload, X } from "lucide-react";
import { useRef, useState } from "react";
import type { SourceContent, SourceItem } from "../../lib/types";
import { EmptyState } from "../ui/EmptyState";

type Props = {
  sources: SourceItem[];
  onSourcesChange: (sources: SourceItem[]) => void;
  onUpload: (file: File) => void;
  onDeleteSource: (source: SourceItem) => void;
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
  onOpenSource,
  activeContent,
  contentLoading,
  contentError,
  onCloseContent,
}: Props) {
  const [uploadOpen, setUploadOpen] = useState(false);
  const readySources = sources.filter((source) => source.status === "ready");
  const allSelected = readySources.length > 0 && readySources.every((source) => source.selected);

  function toggleAll() {
    onSourcesChange(
      sources.map((source) => (source.status === "ready" ? { ...source, selected: !allSelected } : source)),
    );
  }

  function toggle(docId: string) {
    onSourcesChange(sources.map((source) => (source.doc_id === docId ? { ...source, selected: !source.selected } : source)));
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
          {sources.map((source) => (
            <div className={`source-row status-${source.status}`} key={source.doc_id}>
              <FileText className="file-type-icon" size={20} />
              <button
                type="button"
                className="source-title"
                disabled={source.status !== "ready"}
                onClick={() => onOpenSource(source)}
              >
                <span>{source.title}</span>
                <small>
                  {source.source_type} · {source.chunk_count} chunks
                </small>
                {source.error ? <small className="error-text">{source.error}</small> : null}
              </button>
              <input
                type="checkbox"
                checked={Boolean(source.selected)}
                disabled={source.status !== "ready"}
                onChange={() => toggle(source.doc_id)}
              />
              <button
                className="row-icon danger"
                type="button"
                title="删除来源"
                disabled={source.status !== "ready"}
                onClick={() => onDeleteSource(source)}
              >
                <Trash2 size={16} />
              </button>
            </div>
          ))}
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
            {content.text ? (
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
      <div className="upload-dialog" role="dialog" aria-modal="true" onMouseDown={(event) => event.stopPropagation()}>
        <button className="close-button" type="button" onClick={onClose}>
          ×
        </button>
        <h2>添加来源</h2>
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
          <span>PDF、Markdown、TXT、HTML、CSV、TSV</span>
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
