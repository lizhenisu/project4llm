import { FileText, Globe2, Plus, Search, Sparkles, Trash2, Upload } from "lucide-react";
import { useRef, useState } from "react";
import type { SourceItem } from "../../lib/types";
import { EmptyState } from "../ui/EmptyState";

type Props = {
  sources: SourceItem[];
  onSourcesChange: (sources: SourceItem[]) => void;
  onUpload: (file: File) => void;
  onDeleteSource: (source: SourceItem) => void;
};

export function SourcePanel({ sources, onSourcesChange, onUpload, onDeleteSource }: Props) {
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
      <div className="source-search disabled-block">
        <span>在网络中搜索新来源</span>
        <div className="search-controls">
          <button type="button" disabled>
            <Globe2 size={16} />
            Web
          </button>
          <button type="button" disabled>
            <Sparkles size={16} />
            Fast Research
          </button>
          <Search className="muted-icon" size={20} />
        </div>
      </div>
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
              <button type="button" className="source-title" onClick={() => toggle(source.doc_id)}>
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
    </aside>
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
            <button type="button" disabled>
              粘贴文本
            </button>
            <button type="button" disabled>
              网站
            </button>
            <button type="button" disabled>
              云端硬盘
            </button>
          </div>
          <input ref={inputRef} type="file" hidden onChange={(event) => handleFiles(event.target.files)} />
        </div>
      </div>
    </div>
  );
}
