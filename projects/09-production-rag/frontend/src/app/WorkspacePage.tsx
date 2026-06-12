import { useEffect, useMemo, useState } from "react";
import { Bot, Grid3X3, Plus, Settings as SettingsIcon, Share2 } from "lucide-react";
import { ChatPanel } from "../components/chat/ChatPanel";
import { SourcePanel } from "../components/sources/SourcePanel";
import { StudioPanel } from "../components/studio/StudioPanel";
import { IconButton } from "../components/ui/IconButton";
import { SettingsDialog } from "./SettingsDialog";
import {
  createMindMap,
  deleteSource,
  health,
  listArtifacts,
  listSources,
  queryRag,
  sendFeedback,
  uploadSource,
} from "../lib/api";
import { defaultSettings, loadSettings, saveSettings } from "../lib/storage";
import type { ChatMessage, MindMapArtifact, Settings, SourceItem } from "../lib/types";

export function WorkspacePage() {
  const [settings, setSettings] = useState<Settings>(() => loadSettings());
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [artifacts, setArtifacts] = useState<MindMapArtifact[]>([]);
  const [activeArtifact, setActiveArtifact] = useState<MindMapArtifact | null>(null);
  const [status, setStatus] = useState("未连接");
  const [busy, setBusy] = useState(false);

  const selectedSources = useMemo(() => sources.filter((source) => source.selected), [sources]);
  const selectedDocIds = selectedSources.map((source) => source.doc_id);

  useEffect(() => {
    saveSettings(settings);
    void refresh(settings);
  }, [settings]);

  async function refresh(nextSettings = settings) {
    try {
      await health(nextSettings);
      setStatus("API 已连接");
      const [sourceRows, artifactRows] = await Promise.all([
        listSources(nextSettings),
        listArtifacts(nextSettings),
      ]);
      setSources((current) => mergeSelectedState(sourceRows, current));
      setArtifacts(artifactRows);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "连接失败");
    }
  }

  async function handleUpload(file: File) {
    const temp: SourceItem = {
      doc_id: `upload-${Date.now()}`,
      title: file.name,
      source_type: file.name.split(".").pop() || "file",
      source_uri: file.name,
      doc_version: 1,
      chunk_count: 0,
      acl_groups: settings.aclGroups,
      status: "uploading",
      current: false,
      selected: false,
    };
    setSources((items) => [temp, ...items]);
    try {
      const uploaded = await uploadSource(settings, file);
      setSources((items) => mergeSelectedState(uploaded, items.filter((item) => item.doc_id !== temp.doc_id)));
      await refresh();
    } catch (error) {
      setSources((items) =>
        items.map((item) =>
          item.doc_id === temp.doc_id
            ? { ...item, status: "failed", error: error instanceof Error ? error.message : "上传失败" }
            : item,
        ),
      );
    }
  }

  async function handleDeleteSource(source: SourceItem) {
    setSources((items) => items.filter((item) => item.doc_id !== source.doc_id));
    try {
      await deleteSource(settings, source.doc_id);
      await refresh();
    } catch (error) {
      setSources((items) => [
        { ...source, status: "failed", error: error instanceof Error ? error.message : "删除失败" },
        ...items,
      ]);
    }
  }

  async function handleAsk(query: string) {
    if (!query.trim()) return;
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
      status: "done",
    };
    const pending: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "正在检索资料并生成回答...",
      status: "sending",
    };
    setMessages((items) => [...items, userMessage, pending]);
    setBusy(true);
    try {
      const response = await queryRag(settings, {
        query,
        docIds: selectedDocIds,
        history: messages.map((message) => `${message.role}: ${message.content}`).slice(-8),
      });
      setMessages((items) =>
        items.map((item) =>
          item.id === pending.id
            ? {
                ...item,
                content: response.answer,
                requestId: response.request_id,
                citations: response.citations,
                status: "done",
              }
            : item,
        ),
      );
    } catch (error) {
      setMessages((items) =>
        items.map((item) =>
          item.id === pending.id
            ? { ...item, content: error instanceof Error ? error.message : "回答失败", status: "failed" }
            : item,
        ),
      );
    } finally {
      setBusy(false);
    }
  }

  async function handleFeedback(message: ChatMessage, rating: 1 | -1) {
    if (!message.requestId) return;
    await sendFeedback(settings, message.requestId, rating, selectedDocIds);
  }

  async function handleCreateMindMap() {
    if (selectedDocIds.length === 0) return;
    const title = selectedSources.length === 1 ? `${selectedSources[0].title} 思维导图` : "选中来源思维导图";
    const placeholder: MindMapArtifact = {
      id: `pending-${Date.now()}`,
      title,
      status: "generating",
      tenant_id: settings.tenantId,
      source_doc_ids: selectedDocIds,
      created_at: Date.now(),
      updated_at: Date.now(),
      root: null,
    };
    setArtifacts((items) => [placeholder, ...items]);
    try {
      const artifact = await createMindMap(settings, title, selectedDocIds);
      setArtifacts((items) => [artifact, ...items.filter((item) => item.id !== placeholder.id)]);
      setActiveArtifact(artifact);
    } catch (error) {
      setArtifacts((items) =>
        items.map((item) =>
          item.id === placeholder.id
            ? { ...item, status: "failed", error: error instanceof Error ? error.message : "生成失败" }
            : item,
        ),
      );
    }
  }

  return (
    <div className="workspace-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            <Bot size={22} />
          </span>
          <span>未命名的知识库</span>
        </div>
        <div className="topbar-actions">
          <button className="primary-pill" type="button" disabled>
            <Plus size={16} />
            创建知识库
          </button>
          <IconButton label="分享" disabled>
            <Share2 size={18} />
          </IconButton>
          <IconButton label="设置" onClick={() => setSettingsOpen(true)}>
            <SettingsIcon size={18} />
          </IconButton>
          <IconButton label="应用" disabled>
            <Grid3X3 size={18} />
          </IconButton>
          <div className="avatar" aria-label="用户头像" />
        </div>
      </header>
      <main className="workspace-grid">
        <SourcePanel
          sources={sources}
          onSourcesChange={setSources}
          onUpload={handleUpload}
          onDeleteSource={handleDeleteSource}
        />
        <ChatPanel
          messages={messages}
          selectedSources={selectedSources}
          busy={busy}
          onAsk={handleAsk}
          onFeedback={handleFeedback}
        />
        <StudioPanel
          artifacts={artifacts}
          selectedSources={selectedSources}
          activeArtifact={activeArtifact}
          onCreateMindMap={handleCreateMindMap}
          onOpenArtifact={setActiveArtifact}
          onBack={() => setActiveArtifact(null)}
        />
      </main>
      <footer className="statusbar">{status}</footer>
      <SettingsDialog
        open={settingsOpen}
        settings={settings}
        onClose={() => setSettingsOpen(false)}
        onSave={(next) => setSettings({ ...defaultSettings, ...next })}
      />
    </div>
  );
}

function mergeSelectedState(next: SourceItem[], current: SourceItem[]) {
  const selected = new Map(current.map((item) => [item.doc_id, item.selected ?? item.current]));
  return next.map((item) => ({
    ...item,
    selected: selected.get(item.doc_id) ?? item.current,
  }));
}
