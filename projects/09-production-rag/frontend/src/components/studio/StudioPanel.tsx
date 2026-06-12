import {
  ArrowLeft,
  BarChart3,
  ChevronRight,
  Download,
  FileQuestion,
  FileText,
  Flashlight,
  Maximize2,
  Network,
  Presentation,
  Rows3,
  ThumbsDown,
  ThumbsUp,
  Video,
  Volume2,
} from "lucide-react";
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
  { label: "音频概览", icon: Volume2, disabled: true, tone: "blue" },
  { label: "演示文稿", icon: Presentation, disabled: true, tone: "olive" },
  { label: "视频概览", icon: Video, disabled: true, tone: "green" },
  { label: "思维导图", icon: Network, disabled: false, tone: "purple" },
  { label: "报告", icon: FileText, disabled: true, tone: "gold" },
  { label: "闪卡", icon: Flashlight, disabled: true, tone: "red" },
  { label: "测验", icon: FileQuestion, disabled: true, tone: "cyan" },
  { label: "信息图", icon: BarChart3, disabled: true, tone: "pink" },
  { label: "数据表格", icon: Rows3, disabled: true, tone: "indigo" },
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
              disabled={tool.disabled || (!tool.disabled && selectedSources.length === 0)}
              title={tool.disabled ? "即将支持" : selectedSources.length ? "生成思维导图" : "请先选择来源"}
              onClick={tool.disabled ? undefined : onCreateMindMap}
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
  return (
    <aside className="panel studio-panel mindmap-detail">
      <div className="panel-header breadcrumb">
        <button type="button" onClick={onBack}>
          <ArrowLeft size={18} />
          Studio
        </button>
        <span>应用</span>
        <button type="button" className="row-icon" title="全屏">
          <Maximize2 size={18} />
        </button>
      </div>
      <div className="mindmap-title">
        <h2>{artifact.title}</h2>
        <button type="button">查看 {artifact.source_doc_ids.length} 个来源</button>
      </div>
      <div className="mindmap-canvas">
        <div className="canvas-tools">
          <button type="button">⌖</button>
          <button type="button">+</button>
          <button type="button">−</button>
          <button type="button" onClick={() => downloadArtifact(artifact)}>
            <Download size={17} />
          </button>
        </div>
        {artifact.root ? <MindMapTree node={artifact.root} depth={0} /> : <p>暂无可展示的思维导图。</p>}
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

function MindMapTree({ node, depth }: { node: MindMapNode; depth: number }) {
  return (
    <div className={`mindmap-node depth-${Math.min(depth, 3)}`}>
      <div className="node-label">{node.label}</div>
      {node.children?.length ? (
        <div className="node-children">
          {node.children.map((child) => (
            <MindMapTree node={child} depth={depth + 1} key={child.id} />
          ))}
        </div>
      ) : null}
    </div>
  );
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
