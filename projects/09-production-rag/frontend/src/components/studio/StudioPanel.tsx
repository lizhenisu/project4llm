import {
  ArrowLeft,
  ChevronRight,
  Download,
  Maximize2,
  Network,
  ThumbsDown,
  ThumbsUp,
  MoreVertical,
  FileText,
  RefreshCcw,
  Table2,
} from "lucide-react";
import {
  Background,
  Controls,
  MarkerType,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  Position,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { MindMapArtifact, MindMapNode, SourceItem } from "../../lib/types";
import { EmptyState } from "../ui/EmptyState";
import { buildMindMapLayout } from "./mindMapLayout";

type Props = {
  artifacts: MindMapArtifact[];
  sources: SourceItem[];
  selectedSources: SourceItem[];
  activeArtifact: MindMapArtifact | null;
  artifactGenerationLocked: boolean;
  artifactGenerationLockReason: string;
  onCreateMindMap: () => void;
  onCreateDataTable: () => void;
  onOpenArtifact: (artifact: MindMapArtifact) => void;
  onRenameArtifact: (artifact: MindMapArtifact, title: string) => void;
  onDeleteArtifact: (artifact: MindMapArtifact) => void;
  onOpenSource: (source: SourceItem) => void;
  onBack: () => void;
};

const tools = [
  { label: "思维导图", icon: Network, tone: "purple", kind: "mindmap" },
  { label: "数据表格", icon: Table2, tone: "cyan", kind: "table" },
];

export function StudioPanel({
  artifacts,
  sources,
  selectedSources,
  activeArtifact,
  artifactGenerationLocked,
  artifactGenerationLockReason,
  onCreateMindMap,
  onCreateDataTable,
  onOpenArtifact,
  onRenameArtifact,
  onDeleteArtifact,
  onOpenSource,
  onBack,
}: Props) {
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  const [menuPosition, setMenuPosition] = useState<{ top: number; right: number } | null>(null);
  const [editingArtifactId, setEditingArtifactId] = useState<string | null>(null);
  const [editingTitle, setEditingTitle] = useState("");
  const [artifactToDelete, setArtifactToDelete] = useState<MindMapArtifact | null>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const target = event.target as HTMLElement;
      if (!target.closest('.dropdown-menu') && !target.closest('.row-icon-more')) {
        setMenuOpenId(null);
      }
    }
    function handleScroll() {
      setMenuOpenId(null);
    }
    if (menuOpenId) {
      document.addEventListener("mousedown", handleClickOutside);
      window.addEventListener("scroll", handleScroll, true); // true = capture phase to catch scroll on any element
    }
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      window.removeEventListener("scroll", handleScroll, true);
    };
  }, [menuOpenId]);

  function handleSaveTitle(artifact: MindMapArtifact) {
    if (editingTitle.trim() && editingTitle !== artifact.title) {
      onRenameArtifact(artifact, editingTitle.trim());
    }
    setEditingArtifactId(null);
  }

  function getUniqueSourceCount(artifact: MindMapArtifact) {
    const parentSources = new Set<string>();
    artifact.source_doc_ids.forEach(docId => {
      const parent = sources.find(s => s.doc_id === docId || s.child_doc_ids?.includes(docId));
      if (parent) parentSources.add(parent.doc_id);
    });
    return parentSources.size;
  }

  if (activeArtifact?.artifact_type === "table") {
    return <TableDetail artifact={activeArtifact} sources={sources} onOpenSource={onOpenSource} onBack={onBack} />;
  }

  if (activeArtifact) {
    return <MindMapDetail artifact={activeArtifact} sources={sources} onOpenSource={onOpenSource} onBack={onBack} />;
  }

  return (
    <aside className="panel studio-panel">
      <div className="panel-header">
        <h2>Studio</h2>
      </div>
      <div className="tool-grid">
        {tools.map((tool) => {
          const Icon = tool.icon;
          const disabledReason =
            selectedSources.length === 0 ? "请先选择来源" : artifactGenerationLockReason || `生成${tool.label}`;
          return (
            <button
              className={`tool-card tone-${tool.tone}`}
              type="button"
              key={tool.label}
              disabled={selectedSources.length === 0 || artifactGenerationLocked}
              title={disabledReason}
              onClick={tool.kind === "table" ? onCreateDataTable : onCreateMindMap}
            >
              <Icon size={18} />
              <span>{tool.label}</span>
              <ChevronRight size={18} />
            </button>
          );
        })}
      </div>
      <div className="artifact-list">
        {artifacts.length === 0 ? (
          <EmptyState
            icon={<Network size={32} />}
            title="Studio 输出将保存在此处。"
            text="添加来源后，点击即可生成思维导图。"
          />
        ) : (
          artifacts.map((artifact) => (
            <div className={`artifact-row${artifact.status === "generating" ? " is-active-task" : ""}`} key={artifact.id}>
              {artifact.artifact_type === "table" ? (
                <Table2
                  className="tone-cyan"
                  size={22}
                  onClick={() => !editingArtifactId && onOpenArtifact(artifact)}
                  style={{ cursor: "pointer" }}
                />
              ) : (
                <Network
                  className="tone-purple"
                  size={22}
                  onClick={() => !editingArtifactId && onOpenArtifact(artifact)}
                  style={{ cursor: "pointer" }}
                />
              )}
              <div onClick={() => !editingArtifactId && onOpenArtifact(artifact)} style={{ cursor: "pointer", flex: 1, minWidth: 0 }}>
                {editingArtifactId === artifact.id ? (
                  <input
                    className="inline-title-input"
                    type="text"
                    // eslint-disable-next-line jsx-a11y/no-autofocus
                    autoFocus
                    value={editingTitle}
                    onChange={(e) => setEditingTitle(e.target.value)}
                    onBlur={() => handleSaveTitle(artifact)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleSaveTitle(artifact);
                      if (e.key === "Escape") setEditingArtifactId(null);
                    }}
                    onClick={(e) => e.stopPropagation()}
                  />
                ) : (
                  <strong>
                    {artifact.status === "generating"
                      ? `正在生成${artifact.artifact_type === "table" ? "数据表格" : "思维导图"}...`
                      : artifact.title}
                  </strong>
                )}
                <small>
                  {getUniqueSourceCount(artifact)} 个来源 · {formatTime(artifact.updated_at)}
                </small>
                {artifact.error ? <small className="error-text">{artifact.error}</small> : null}
              </div>
              <div className="artifact-menu-container">
                <button
                  type="button"
                  className="row-icon row-icon-more"
                  style={{ background: "none", border: "none", cursor: "pointer", padding: "4px" }}
                  onClick={(e) => {
                    e.stopPropagation();
                    if (menuOpenId === artifact.id) {
                      setMenuOpenId(null);
                    } else {
                      const rect = e.currentTarget.getBoundingClientRect();
                      setMenuPosition({
                        top: rect.bottom + 4,
                        right: document.documentElement.clientWidth - rect.right,
                      });
                      setMenuOpenId(artifact.id);
                    }
                  }}
                >
                  <MoreVertical size={18} />
                </button>
                {menuOpenId === artifact.id && menuPosition && createPortal(
                  <>
                  <div className="dropdown-backdrop" onMouseDown={() => setMenuOpenId(null)} />
                  <div className="dropdown-menu" style={{ top: menuPosition.top, right: menuPosition.right }} onMouseDown={(e) => e.stopPropagation()}>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setMenuOpenId(null);
                        setEditingArtifactId(artifact.id);
                        setEditingTitle(artifact.title);
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
                        setArtifactToDelete(artifact);
                      }}
                    >
                      删除
                    </button>
                  </div>
                  </>,
                  document.body
                )}
              </div>
            </div>
          ))
        )}
      </div>

      {artifactToDelete && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => setArtifactToDelete(null)}>
          <div className="settings-dialog" role="dialog" aria-modal="true" style={{ width: "400px", padding: "24px" }} onMouseDown={(e) => e.stopPropagation()}>
            <h2 style={{ marginTop: 0, marginBottom: "16px", fontSize: "18px" }}>确认删除</h2>
            <p style={{ color: "var(--text-muted)", marginBottom: "24px", lineHeight: 1.5 }}>
              您确定要删除思维导图 <strong>"{artifactToDelete.title}"</strong> 吗？此操作无法撤销。
            </p>
            <div className="dialog-actions">
              <button type="button" onClick={() => setArtifactToDelete(null)} style={{ padding: "8px 16px", background: "var(--surface-muted)", border: "1px solid var(--line)", borderRadius: "var(--radius)", cursor: "pointer" }}>
                取消
              </button>
              <button
                type="button"
                style={{ padding: "8px 16px", background: "var(--danger)", color: "white", border: "none", borderRadius: "var(--radius)", cursor: "pointer", fontWeight: 500 }}
                onClick={() => {
                  onDeleteArtifact(artifactToDelete);
                  setArtifactToDelete(null);
                }}
              >
                确认删除
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}

function TableDetail({
  artifact,
  sources,
  onOpenSource,
  onBack,
}: {
  artifact: MindMapArtifact;
  sources: SourceItem[];
  onOpenSource: (source: SourceItem) => void;
  onBack: () => void;
}) {
  const [showSources, setShowSources] = useState(false);
  const artifactSources = useMemo(() => {
    const parentSources = new Map<string, SourceItem>();
    artifact.source_doc_ids.forEach((docId) => {
      const parent = sources.find((source) => source.doc_id === docId || source.child_doc_ids?.includes(docId));
      if (parent) parentSources.set(parent.doc_id, parent);
    });
    return Array.from(parentSources.values());
  }, [artifact.source_doc_ids, sources]);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const target = event.target as HTMLElement;
      if (!target.closest(".source-popover-container")) {
        setShowSources(false);
      }
    }
    if (showSources) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showSources]);

  const table = artifact.table;

  return (
    <aside className="panel studio-panel table-detail">
      <div className="panel-header breadcrumb">
        <button type="button" onClick={onBack}>
          <ArrowLeft size={18} />
          Studio
        </button>
        <span>数据表格</span>
        <button type="button" className="row-icon" title="下载" onClick={() => downloadArtifact(artifact)}>
          <Download size={17} />
        </button>
      </div>
      <div className="table-title">
        <div>
          <h2>{artifact.title}</h2>
          {table?.summary ? <p>{table.summary}</p> : null}
        </div>
        <div className="source-popover-container" style={{ position: "relative" }}>
          <button type="button" onClick={() => setShowSources(!showSources)}>
            查看 {artifactSources.length} 个来源
          </button>
          {showSources ? (
            <div className="source-popover table-source-popover">
              <div className="popover-header">
                <div>
                  <FileText size={16} /> 来源
                </div>
              </div>
              <div className="popover-list">
                {artifactSources.map((source) => (
                  <button
                    type="button"
                    key={source.doc_id}
                    onClick={() => {
                      onOpenSource(source);
                      setShowSources(false);
                    }}
                  >
                    <FileText size={14} color="var(--danger)" />
                    <span>{source.title}</span>
                  </button>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      </div>
      <div className="table-artifact-scroll">
        {table && table.columns.length > 0 && table.rows.length > 0 ? (
          <>
            <h3 className="artifact-table-caption">{table.title}</h3>
            <table className="artifact-table">
              <thead>
                <tr>
                  {table.columns.map((column) => (
                    <th key={column}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row, rowIndex) => (
                  <tr key={`${rowIndex}-${row.join("|").slice(0, 24)}`}>
                    {table.columns.map((column, columnIndex) => (
                      <td key={`${column}-${columnIndex}`}>{row[columnIndex] || "未提及"}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        ) : (
          <EmptyState icon={<Table2 size={32} />} title="暂无可展示的数据表格。" text="请重新生成该表格。" />
        )}
      </div>
    </aside>
  );
}

function MindMapDetail({
  artifact,
  sources,
  onOpenSource,
  onBack,
}: {
  artifact: MindMapArtifact;
  sources: SourceItem[];
  onOpenSource: (source: SourceItem) => void;
  onBack: () => void;
}) {
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const [expandedNodeIds, setExpandedNodeIds] = useState<Set<string>>(() => new Set());
  const [showSources, setShowSources] = useState(false);
  const sourceButtonRef = useRef<HTMLButtonElement | null>(null);

  const artifactSources = useMemo(() => {
    const parentSources = new Map<string, SourceItem>();
    artifact.source_doc_ids.forEach(docId => {
      // Find a source where its doc_id is docId, or its child_doc_ids includes docId
      const parent = sources.find(s => s.doc_id === docId || s.child_doc_ids?.includes(docId));
      if (parent) {
        parentSources.set(parent.doc_id, parent);
      }
    });
    return Array.from(parentSources.values());
  }, [artifact.source_doc_ids, sources]);

  // Click outside to close popover
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const target = event.target as HTMLElement;
      if (!target.closest('.source-popover-container')) {
        setShowSources(false);
      }
    }
    if (showSources) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showSources]);

  async function toggleFullscreen() {
    const element = canvasRef.current;
    if (!element) return;
    if (document.fullscreenElement) {
      await document.exitFullscreen();
      return;
    }
    await element.requestFullscreen();
  }

  return (
    <aside className="panel studio-panel mindmap-detail">
      <div className="panel-header breadcrumb">
        <button type="button" onClick={onBack}>
          <ArrowLeft size={18} />
          Studio
        </button>
        <span>应用</span>
        <button type="button" className="row-icon" title="全屏" onClick={toggleFullscreen}>
          <Maximize2 size={18} />
        </button>
      </div>
      <div className="mindmap-title">
        <h2>{artifact.title}</h2>
        <div className="source-popover-container" style={{ position: "relative" }}>
          <button
            type="button"
            ref={sourceButtonRef}
            onClick={() => setShowSources(!showSources)}
            style={{ padding: "6px 12px", borderRadius: "16px", border: "1px solid var(--line)", background: "var(--surface)", cursor: "pointer", fontWeight: 500 }}
          >
            查看 {artifactSources.length} 个来源
          </button>

          {showSources && (
            <div
              className="source-popover"
              style={{
                position: "absolute",
                top: "100%",
                marginTop: "8px",
                right: 0,
                width: "280px",
                background: "var(--surface)",
                border: "1px solid var(--line)",
                borderRadius: "12px",
                boxShadow: "0 10px 25px rgba(0,0,0,0.1)",
                zIndex: 100,
                display: "flex",
                flexDirection: "column",
                overflow: "hidden"
              }}
            >
              <div className="popover-header" style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "12px 16px", borderBottom: "1px solid var(--line)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "6px", fontWeight: 600 }}>
                  <FileText size={16} /> 来源
                </div>
                <button type="button" onClick={(e) => e.stopPropagation()} style={{ background: "var(--surface-muted)", border: "none", borderRadius: "50%", width: "24px", height: "24px", display: "grid", placeItems: "center", cursor: "pointer", color: "var(--text)" }}>
                  <RefreshCcw size={14} />
                </button>
              </div>
              <div className="popover-list" style={{ maxHeight: "240px", overflowY: "auto", padding: "8px" }}>
                {artifactSources.map(src => (
                  <button
                    type="button"
                    key={src.doc_id}
                    onClick={() => {
                      onOpenSource(src);
                      setShowSources(false);
                    }}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "8px",
                      width: "100%",
                      padding: "8px",
                      background: "transparent",
                      border: "1px solid var(--line)",
                      borderRadius: "8px",
                      marginBottom: "6px",
                      cursor: "pointer",
                      textAlign: "left"
                    }}
                  >
                    <FileText size={14} color="var(--danger)" />
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1, fontSize: "13px" }}>
                      {src.title}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
      <div className="mindmap-canvas" ref={canvasRef}>
        <div className="mindmap-download-action">
          <button type="button" onClick={() => downloadArtifact(artifact)}>
            <Download size={17} />
          </button>
        </div>
        {artifact.root ? (
          <InteractiveMindMap
            root={artifact.root}
            expandedNodeIds={expandedNodeIds}
            onToggleNode={(nodeId) =>
              setExpandedNodeIds((current) => {
                const next = new Set(current);
                if (next.has(nodeId)) {
                  next.delete(nodeId);
                } else {
                  next.add(nodeId);
                }
                return next;
              })
            }
          />
        ) : (
          <p>暂无可展示的思维导图。</p>
        )}
      </div>
    </aside>
  );
}

function InteractiveMindMap({
  root,
  expandedNodeIds,
  onToggleNode,
}: {
  root: MindMapNode;
  expandedNodeIds: Set<string>;
  onToggleNode: (nodeId: string) => void;
}) {
  return (
    <ReactFlowProvider>
      <MindMapCanvas root={root} expandedNodeIds={expandedNodeIds} onToggleNode={onToggleNode} />
    </ReactFlowProvider>
  );
}

function MindMapCanvas({
  root,
  expandedNodeIds,
  onToggleNode,
}: {
  root: MindMapNode;
  expandedNodeIds: Set<string>;
  onToggleNode: (nodeId: string) => void;
}) {
  const { fitView } = useReactFlow();

  useEffect(() => {
    function handleFullscreenChange() {
      // Need a slight delay to allow the DOM/Canvas resize to complete before fitting
      setTimeout(() => {
        fitView({ padding: 0.22, duration: 450 });
      }, 50);
    }
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, [fitView]);

  const branches = root.children || [];
  const { nodes, edges } = useMemo(
    () => buildMindMapFlow(root, expandedNodeIds),
    [root, expandedNodeIds],
  );

  if (branches.length === 0) {
    return <div className="mindmap-empty-node">{root.label}</div>;
  }

  return (
    <ReactFlow
      className="mindmap-flow"
      nodes={nodes}
      edges={edges}
      nodesDraggable
      nodesConnectable={false}
      elementsSelectable
      fitView
      fitViewOptions={{ padding: 0.22, duration: 450 }}
      minZoom={0.35}
      maxZoom={1.8}
      onNodeClick={(_, node) => {
        if (node.data.canExpand) {
          onToggleNode(node.id);
        }
      }}
    >
      <Background color="#dce4f0" gap={26} />
      <Controls showInteractive={false} />
    </ReactFlow>
  );
}

type MindMapFlowNodeData = {
  label: string;
  canExpand: boolean;
  expanded: boolean;
};

function buildMindMapFlow(root: MindMapNode, expandedNodeIds: Set<string>): { nodes: Node<MindMapFlowNodeData>[]; edges: Edge[] } {
  const layoutNodes = buildMindMapLayout(root, expandedNodeIds);
  const nodes: Node<MindMapFlowNodeData>[] = layoutNodes.map((node) => ({
    id: node.id,
    type: "default",
    position: { x: node.x, y: node.y },
    sourcePosition: node.kind === "leaf" ? undefined : Position.Right,
    targetPosition: node.kind === "root" ? undefined : Position.Left,
    data: {
      label: node.label,
      canExpand: node.canExpand,
      expanded: node.expanded,
    },
    className: [
      "mindmap-flow-node",
      node.kind,
      node.kind === "root" ? "" : `tone-${node.toneIndex}`,
    ].filter(Boolean).join(" "),
    draggable: true,
  }));
  const edges: Edge[] = [];
  layoutNodes.forEach((node) => {
    if (node.parentId) {
      edges.push(makeMindMapEdge(node.parentId, node.id, `edge-${node.parentId}-${node.id}`));
    }
  });

  return { nodes, edges };
}

function makeMindMapEdge(source: string, target: string, id: string): Edge {
  return {
    id,
    source,
    target,
    type: "default",
    animated: false,
    markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16, color: "#6b7cff" },
    style: { stroke: "#6b7cff", strokeWidth: 2 },
  };
}

function downloadArtifact(artifact: MindMapArtifact) {
  const blob = new Blob([JSON.stringify(artifact, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${artifact.title}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function formatTime(value: number) {
  if (!value) return "刚刚";
  const delta = Date.now() - value;
  const minute = 60_000;
  if (delta < minute) return "刚刚";
  if (delta < 60 * minute) return `${Math.round(delta / minute)} 分钟前`;
  return new Date(value).toLocaleString("zh-CN");
}
