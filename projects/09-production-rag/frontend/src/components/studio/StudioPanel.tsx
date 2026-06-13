import {
  ArrowLeft,
  ChevronRight,
  Download,
  Maximize2,
  Network,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import {
  Background,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useMemo, useRef, useState } from "react";
import type { MindMapArtifact, MindMapNode, SourceItem } from "../../lib/types";
import { EmptyState } from "../ui/EmptyState";

type Props = {
  artifacts: MindMapArtifact[];
  selectedSources: SourceItem[];
  activeArtifact: MindMapArtifact | null;
  onCreateMindMap: () => void;
  onOpenArtifact: (artifact: MindMapArtifact) => void;
  onBack: () => void;
};

const tools = [
  { label: "思维导图", icon: Network, tone: "purple" },
];

export function StudioPanel({
  artifacts,
  selectedSources,
  activeArtifact,
  onCreateMindMap,
  onOpenArtifact,
  onBack,
}: Props) {
  if (activeArtifact) {
    return <MindMapDetail artifact={activeArtifact} onBack={onBack} />;
  }

  return (
    <aside className="panel studio-panel">
      <div className="panel-header">
        <h2>Studio</h2>
      </div>
      <div className="tool-grid">
        {tools.map((tool) => {
          const Icon = tool.icon;
          return (
            <button
              className={`tool-card tone-${tool.tone}`}
              type="button"
              key={tool.label}
              disabled={selectedSources.length === 0}
              title={selectedSources.length ? "生成思维导图" : "请先选择来源"}
              onClick={onCreateMindMap}
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
            <button className="artifact-row" type="button" key={artifact.id} onClick={() => onOpenArtifact(artifact)}>
              <Network size={22} />
              <span>
                <strong>{artifact.status === "generating" ? "正在生成思维导图..." : artifact.title}</strong>
                <small>
                  {artifact.source_doc_ids.length} 个来源 · {formatTime(artifact.updated_at)}
                </small>
                {artifact.error ? <small className="error-text">{artifact.error}</small> : null}
              </span>
            </button>
          ))
        )}
      </div>
    </aside>
  );
}

function MindMapDetail({ artifact, onBack }: { artifact: MindMapArtifact; onBack: () => void }) {
  const canvasRef = useRef<HTMLDivElement | null>(null);
  const [expandedNodeIds, setExpandedNodeIds] = useState<Set<string>>(() => new Set());

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
        <button type="button">查看 {artifact.source_doc_ids.length} 个来源</button>
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
      <div className="artifact-feedback">
        <button type="button">
          <ThumbsUp size={18} />
          优质内容
        </button>
        <button type="button">
          <ThumbsDown size={18} />
          劣质内容
        </button>
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
      <MiniMap nodeStrokeWidth={3} />
    </ReactFlow>
  );
}

type MindMapFlowNodeData = {
  label: string;
  canExpand: boolean;
  expanded: boolean;
};

function buildMindMapFlow(root: MindMapNode, expandedNodeIds: Set<string>): { nodes: Node<MindMapFlowNodeData>[]; edges: Edge[] } {
  const nodes: Node<MindMapFlowNodeData>[] = [
    {
      id: root.id,
      type: "default",
      position: { x: 0, y: 0 },
      data: { label: root.label, canExpand: false, expanded: false },
      className: "mindmap-flow-node root",
      draggable: true,
    },
  ];
  const edges: Edge[] = [];
  const branches = root.children || [];
  const rowGap = 82;
  const rootY = ((branches.length - 1) * rowGap) / 2;
  nodes[0].position.y = rootY;

  branches.forEach((branch, index) => {
    const expanded = expandedNodeIds.has(branch.id);
    const branchY = index * rowGap;
    nodes.push({
      id: branch.id,
      type: "default",
      position: { x: 330, y: branchY },
      data: {
        label: `${branch.label}${branch.children?.length ? ` ${expanded ? "⌄" : "›"}` : ""}`,
        canExpand: Boolean(branch.children?.length),
        expanded,
      },
      className: `mindmap-flow-node branch tone-${index % 6}`,
      draggable: true,
    });
    edges.push(makeMindMapEdge(root.id, branch.id, `edge-${root.id}-${branch.id}`));

    if (!expanded) {
      return;
    }
    const children = branch.children || [];
    const childStartY = branchY - ((children.length - 1) * 58) / 2;
    children.forEach((child, childIndex) => {
      nodes.push({
        id: child.id,
        type: "default",
        position: { x: 690, y: childStartY + childIndex * 58 },
        data: {
          label: child.label,
          canExpand: false,
          expanded: false,
        },
        className: `mindmap-flow-node leaf tone-${childIndex % 6}`,
        draggable: true,
      });
      edges.push(makeMindMapEdge(branch.id, child.id, `edge-${branch.id}-${child.id}`));
    });
  });
  return { nodes, edges };
}

function makeMindMapEdge(source: string, target: string, id: string): Edge {
  return {
    id,
    source,
    target,
    type: "smoothstep",
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
