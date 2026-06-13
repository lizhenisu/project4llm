import { useEffect, useMemo, useRef, useState } from "react";
import type { Dispatch, FormEvent, PointerEvent as ReactPointerEvent, RefObject, SetStateAction } from "react";
import { Bot, LogIn, LogOut, Megaphone, Settings as SettingsIcon, Shield, UserRound } from "lucide-react";
import { ChatPanel } from "../components/chat/ChatPanel";
import { SourcePanel } from "../components/sources/SourcePanel";
import { StudioPanel } from "../components/studio/StudioPanel";
import { IconButton } from "../components/ui/IconButton";
import { SettingsDialog } from "./SettingsDialog";
import {
  createDataTable,
  createMindMap,
  deleteConversation,
  deleteSource,
  deleteArtifact,
  renameArtifact,
  getConversation,
  getArtifact,
  getSourceContent,
  health,
  listAdminUsers,
  listAnnouncements,
  listArtifacts,
  listConversations,
  listSources,
  publishAnnouncement,
  queryRag,
  saveConversation,
  sendFeedback,
  uploadSource,
} from "../lib/api";
import { useAuth } from "../lib/AuthContext";
import { defaultSettings, loadSettings, saveSettings } from "../lib/storage";
import type { Announcement, AuthUser, ChatMessage, MindMapArtifact, Settings, SourceContent, SourceItem } from "../lib/types";

type PanelLayout = {
  source: number;
  chat: number;
  studio: number;
};

type ResizeHandle = "source-chat" | "chat-studio";

const DEFAULT_LAYOUT: PanelLayout = { source: 24, chat: 46, studio: 30 };
const MINDMAP_LAYOUT: PanelLayout = { source: 20, chat: 38, studio: 42 };
const MIN_LAYOUT: PanelLayout = { source: 17, chat: 31, studio: 22 };
const ARTIFACT_GENERATION_COOLDOWN_MS = 4_000;

export function WorkspacePage() {
  const auth = useAuth();
  const [settings, setSettings] = useState<Settings>(() => loadSettings());
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const [authDialogMode, setAuthDialogMode] = useState<"login" | "register" | null>(null);
  const [adminOpen, setAdminOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const [announcements, setAnnouncements] = useState<Announcement[]>([]);
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [conversationTitle, setConversationTitle] = useState("未命名对话");
  const [artifacts, setArtifacts] = useState<MindMapArtifact[]>([]);
  const [activeArtifact, setActiveArtifact] = useState<MindMapArtifact | null>(null);
  const [activeSourceContent, setActiveSourceContent] = useState<SourceContent | null>(null);
  const [sourceContentError, setSourceContentError] = useState("");
  const [sourceContentLoading, setSourceContentLoading] = useState(false);
  const [status, setStatus] = useState("未连接");
  const [busy, setBusy] = useState(false);
  const [artifactGenerationBusy, setArtifactGenerationBusy] = useState(false);
  const [artifactGenerationReadyAt, setArtifactGenerationReadyAt] = useState(0);
  const [, setCooldownTick] = useState(0);
  const [panelLayout, setPanelLayout] = useState<PanelLayout>(DEFAULT_LAYOUT);
  const [typingMessageId, setTypingMessageId] = useState<string | null>(null);
  const gridRef = useRef<HTMLElement | null>(null);
  const studioListLayoutRef = useRef<PanelLayout | null>(null);
  const artifactGenerationBusyRef = useRef(false);
  const artifactGenerationReadyAtRef = useRef(0);

  const selectedSources = useMemo(() => sources.filter((source) => source.selected), [sources]);
  const selectedDocIds = useMemo(
    () =>
      selectedSources.flatMap((source) =>
        source.child_doc_ids && source.child_doc_ids.length > 0 ? source.child_doc_ids : [source.doc_id],
      ),
    [selectedSources],
  );
  const artifactCooldownRemainingMs = Math.max(0, artifactGenerationReadyAt - Date.now());
  const artifactGenerationLocked = artifactGenerationBusy || artifactCooldownRemainingMs > 0;
  const artifactGenerationLockReason = artifactGenerationBusy
    ? "正在生成上一个 Studio 项"
    : artifactCooldownRemainingMs > 0
      ? `请等待 ${Math.ceil(artifactCooldownRemainingMs / 1000)} 秒后再生成`
      : "";

  useEffect(() => {
    artifactGenerationReadyAtRef.current = artifactGenerationReadyAt;
    if (artifactGenerationBusy || artifactGenerationReadyAt <= Date.now()) return;
    const timer = window.setInterval(() => setCooldownTick((value) => value + 1), 250);
    return () => window.clearInterval(timer);
  }, [artifactGenerationBusy, artifactGenerationReadyAt]);

  useEffect(() => {
    saveSettings(settings);
    void refresh(settings);
  }, [settings]);

  useEffect(() => {
    setSettings((current) => {
      const next = {
        ...current,
        token: auth.token,
        tenantId: auth.user?.tenant_id || defaultSettings.tenantId,
        aclGroups: ["engineering"],
      };
      return settingsEqual(current, next) ? current : next;
    });
  }, [auth.token, auth.user?.tenant_id]);

  useEffect(() => {
    setSources([]);
    setMessages([]);
    setConversationId(null);
    setConversationTitle("未命名对话");
    setArtifacts([]);
    setActiveArtifact(null);
    setActiveSourceContent(null);
  }, [auth.user?.id]);

  async function refresh(nextSettings = settings) {
    try {
      await health(nextSettings);
      setStatus("API 已连接");
      const [sourceRows, artifactRows, announcementRows] = await Promise.all([
        listSources(nextSettings),
        listArtifacts(nextSettings),
        listAnnouncements(nextSettings),
      ]);
      setSources((current) => mergeSelectedState(sourceRows, current));
      setArtifacts(artifactRows);
      setAnnouncements(announcementRows);
      await loadLatestConversation(nextSettings, sourceRows);
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "连接失败");
    }
  }

  async function loadLatestConversation(nextSettings: Settings, sourceRows: SourceItem[]) {
    const rows = await listConversations(nextSettings);
    if (conversationId || rows.length === 0 || messages.length > 0) {
      return;
    }
    const latest = await getConversation(nextSettings, rows[0].id);
    setConversationId(latest.id);
    setConversationTitle(latest.title);
    setMessages(latest.messages.map(normalizeMessage));
    const selectedIds = new Set(latest.source_doc_ids);
    if (selectedIds.size > 0) {
      setSources((current) =>
        (current.length ? current : sourceRows).map((source) => ({
          ...source,
          selected:
            selectedIds.has(source.doc_id) || Boolean(source.child_doc_ids?.some((docId) => selectedIds.has(docId))),
        })),
      );
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
      setSources((items) => [
        ...uploaded.map(item => ({ ...item, selected: true })), // Auto-select newly uploaded items
        ...items.filter((item) => item.doc_id !== temp.doc_id)
      ]);
      const readyRows = await waitForSourcesReady(settings, uploaded);
      setSources((items) => mergeSelectedState(readyRows, items));
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
    if (activeSourceContent?.doc_id === source.doc_id) {
      setActiveSourceContent(null);
    }
    try {
      await deleteSource(settings, source.doc_id);
    } catch (error) {
      setSources((items) => [
        { ...source, status: "failed", error: error instanceof Error ? error.message : "删除失败" },
        ...items,
      ]);
    }
  }

  async function handleOpenSource(source: SourceItem) {
    if (source.status !== "ready") return;
    setSourceContentError("");
    setSourceContentLoading(true);
    setActiveSourceContent({
      doc_id: source.doc_id,
      title: source.title,
      source_type: source.source_type,
      source_uri: source.source_uri,
      doc_version: source.doc_version,
      child_doc_ids: source.child_doc_ids || [source.doc_id],
      guide: "正在加载来源内容...",
      tags: [],
      text: "",
    });
    try {
      const content = await getSourceContent(settings, source.doc_id);
      setActiveSourceContent(content);
    } catch (error) {
      setSourceContentError(error instanceof Error ? error.message : "加载来源内容失败");
    } finally {
      setSourceContentLoading(false);
    }
  }

  async function handleAsk(query: string) {
    if (!query.trim()) return;
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
      status: "done",
      created_at: Date.now(),
    };
    const pending: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: "正在检索资料并生成回答...",
      status: "sending",
      created_at: Date.now(),
    };
    const baseMessages = [...messages, userMessage, pending];
    setMessages(baseMessages);
    setTypingMessageId(null);
    setBusy(true);
    try {
      const response = await queryRag(settings, {
        query,
        docIds: selectedDocIds,
        history: messages.map((message) => `${message.role}: ${message.content}`).slice(-8),
      });
      const nextMessages: ChatMessage[] = baseMessages.map((item) =>
          item.id === pending.id
            ? {
                ...item,
                content: response.answer,
                requestId: response.request_id,
                citations: response.citations,
                status: "done" as const,
              }
            : item,
      );
      setMessages(nextMessages);
      setTypingMessageId(pending.id);
      await persistConversation(nextMessages);
    } catch (error) {
      const failedMessages: ChatMessage[] = baseMessages.map((item) =>
          item.id === pending.id
            ? { ...item, content: error instanceof Error ? error.message : "回答失败", status: "failed" as const }
            : item,
      );
      setMessages(failedMessages);
      setTypingMessageId(null);
      await persistConversation(failedMessages);
    } finally {
      setBusy(false);
    }
  }

  async function persistConversation(nextMessages: ChatMessage[]) {
    if (nextMessages.length === 0) return;
    const title = inferConversationTitle(nextMessages);
    const saved = await saveConversation(settings, {
      id: conversationId,
      title,
      messages: nextMessages,
      sourceDocIds: selectedDocIds,
    });
    setConversationId(saved.id);
    setConversationTitle(saved.title);
  }

  async function handleDeleteConversation() {
    if (conversationId) {
      await deleteConversation(settings, conversationId);
    }
    setConversationId(null);
    setConversationTitle("未命名对话");
    setMessages([]);
  }

  async function handleFeedback(message: ChatMessage, rating: 1 | -1) {
    if (!message.requestId) return;
    await sendFeedback(settings, message.requestId, rating, selectedDocIds);
  }

  function openArtifact(artifact: MindMapArtifact) {
    setPanelLayout((layout) => {
      if (!activeArtifact) {
        studioListLayoutRef.current = layout;
      }
      return layout.studio >= MINDMAP_LAYOUT.studio ? layout : MINDMAP_LAYOUT;
    });
    setActiveArtifact(artifact);
  }

  function backToStudioList() {
    setActiveArtifact(null);
    if (studioListLayoutRef.current) {
      setPanelLayout(studioListLayoutRef.current);
      studioListLayoutRef.current = null;
    }
  }

  function beginArtifactGeneration() {
    const now = Date.now();
    if (artifactGenerationBusyRef.current || now < artifactGenerationReadyAtRef.current) {
      return false;
    }
    const readyAt = now + ARTIFACT_GENERATION_COOLDOWN_MS;
    artifactGenerationBusyRef.current = true;
    artifactGenerationReadyAtRef.current = readyAt;
    setArtifactGenerationBusy(true);
    setArtifactGenerationReadyAt(readyAt);
    return true;
  }

  function finishArtifactGeneration() {
    artifactGenerationBusyRef.current = false;
    setArtifactGenerationBusy(false);
  }

  async function handleCreateMindMap() {
    if (selectedDocIds.length === 0 || !beginArtifactGeneration()) return;
    const sourceDocIds = [...selectedDocIds];
    const title = selectedSources.length === 1 ? `${selectedSources[0].title} 思维导图` : "选中来源思维导图";
    const pendingArtifact: MindMapArtifact = {
      id: `pending-${Date.now()}`,
      title,
      status: "generating",
      tenant_id: settings.tenantId,
      source_doc_ids: sourceDocIds,
      created_at: Date.now(),
      updated_at: Date.now(),
      artifact_type: "mindmap",
      root: null,
    };
    setArtifacts((items) => [pendingArtifact, ...items]);
    let trackedArtifactId = pendingArtifact.id;
    try {
      const artifact = await createMindMap(settings, title, sourceDocIds);
      trackedArtifactId = artifact.id;
      setArtifacts((items) => [artifact, ...items.filter((item) => item.id !== pendingArtifact.id)]);
      const readyArtifact = await waitForArtifact(settings, artifact.id);
      setArtifacts((items) => [readyArtifact, ...items.filter((item) => item.id !== artifact.id)]);
      openArtifact(readyArtifact);
    } catch (error) {
      setArtifacts((items) =>
        items.map((item) =>
          item.id === pendingArtifact.id || item.id === trackedArtifactId
            ? { ...item, status: "failed", error: error instanceof Error ? error.message : "生成失败" }
          : item,
        ),
      );
    } finally {
      finishArtifactGeneration();
    }
  }

  async function handleCreateDataTable() {
    if (selectedDocIds.length === 0 || !beginArtifactGeneration()) return;
    const sourceDocIds = [...selectedDocIds];
    const title = selectedSources.length === 1 ? `${selectedSources[0].title} 数据表格` : "选中来源数据表格";
    const pendingArtifact: MindMapArtifact = {
      id: `pending-table-${Date.now()}`,
      title,
      status: "generating",
      tenant_id: settings.tenantId,
      source_doc_ids: sourceDocIds,
      created_at: Date.now(),
      updated_at: Date.now(),
      artifact_type: "table",
      table: null,
    };
    setArtifacts((items) => [pendingArtifact, ...items]);
    let trackedArtifactId = pendingArtifact.id;
    try {
      const artifact = await createDataTable(settings, title, sourceDocIds);
      trackedArtifactId = artifact.id;
      setArtifacts((items) => [artifact, ...items.filter((item) => item.id !== pendingArtifact.id)]);
      const readyArtifact = await waitForArtifact(settings, artifact.id);
      setArtifacts((items) => [readyArtifact, ...items.filter((item) => item.id !== artifact.id)]);
      openArtifact(readyArtifact);
    } catch (error) {
      setArtifacts((items) =>
        items.map((item) =>
          item.id === pendingArtifact.id || item.id === trackedArtifactId
            ? { ...item, status: "failed", error: error instanceof Error ? error.message : "生成失败" }
          : item,
        ),
      );
    } finally {
      finishArtifactGeneration();
    }
  }

  async function handleRenameArtifact(artifact: MindMapArtifact, newTitle: string) {
    if (!newTitle.trim() || newTitle === artifact.title) return;
    try {
      await renameArtifact(settings, artifact.id, newTitle);
      setArtifacts((items) => items.map((item) => (item.id === artifact.id ? { ...item, title: newTitle } : item)));
      if (activeArtifact?.id === artifact.id) {
        setActiveArtifact({ ...activeArtifact, title: newTitle });
      }
    } catch (error) {
      console.error("Rename failed:", error);
    }
  }


  async function handleRenameSource(source: SourceItem, newTitle: string) {
    if (!newTitle.trim() || newTitle === source.title) return;
    try {
      // Mocking the backend call for now since there's no rename_source endpoint.
      // Update local state directly so the UI responds correctly.
      setSources((items) => items.map((item) => (item.doc_id === source.doc_id ? { ...item, title: newTitle } : item)));
    } catch (error) {
      console.error("Rename source failed:", error);
    }
  }

  async function handleDeleteArtifact(artifact: MindMapArtifact) {
    try {
      await deleteArtifact(settings, artifact.id);
      setArtifacts((items) => items.filter((item) => item.id !== artifact.id));
      if (activeArtifact?.id === artifact.id) {
        setActiveArtifact(null);
      }
    } catch (error) {
      console.error("Delete failed:", error);
    }
  }

  async function handleLogout() {
    setAccountMenuOpen(false);
    await auth.logout();
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
          <IconButton label="设置" onClick={() => setSettingsOpen(true)}>
            <SettingsIcon size={18} />
          </IconButton>
          <div className="account-menu-anchor">
            <button
              type="button"
              className="avatar"
              aria-label="用户头像"
              title={auth.user ? auth.user.display_name : "账户"}
              onClick={() => setAccountMenuOpen((open) => !open)}
            >
              {auth.user ? auth.user.display_name.slice(0, 1).toUpperCase() : ""}
            </button>
            {accountMenuOpen ? (
              <AccountMenu
                user={auth.user}
                onLogin={() => {
                  setAccountMenuOpen(false);
                  setAuthDialogMode("login");
                }}
                onRegister={() => {
                  setAccountMenuOpen(false);
                  setAuthDialogMode("register");
                }}
                onProfile={() => {
                  setAccountMenuOpen(false);
                  setProfileOpen(true);
                }}
                onAdmin={() => {
                  setAccountMenuOpen(false);
                  setAdminOpen(true);
                }}
                onLogout={handleLogout}
              />
            ) : null}
          </div>
        </div>
      </header>
      <AnnouncementBar announcement={announcements[0] ?? null} />
      <main
        className={`workspace-grid ${activeArtifact ? "mindmap-expanded" : ""}`}
        ref={gridRef}
        style={{
          gridTemplateColumns: `${panelLayout.source}fr 10px ${panelLayout.chat}fr 10px ${panelLayout.studio}fr`,
        }}
      >
        <SourcePanel
          sources={sources}
          onSourcesChange={setSources}
          onUpload={handleUpload}
          onDeleteSource={handleDeleteSource}
          onRenameSource={handleRenameSource}
          onOpenSource={handleOpenSource}
          activeContent={activeSourceContent}
          contentLoading={sourceContentLoading}
          contentError={sourceContentError}
          onCloseContent={() => {
            setActiveSourceContent(null);
            setSourceContentError("");
          }}
        />
        <ResizeDivider
          label="调整来源和对话宽度"
          onPointerDown={(event) => startPanelResize(event, "source-chat", gridRef, panelLayout, setPanelLayout)}
        />
        <ChatPanel
          messages={messages}
          selectedSources={selectedSources}
          busy={busy}
          conversationTitle={conversationTitle}
          typingMessageId={typingMessageId}
          onTypingComplete={() => setTypingMessageId(null)}
          onAsk={handleAsk}
          onFeedback={handleFeedback}
          onDeleteConversation={handleDeleteConversation}
        />
        <ResizeDivider
          label="调整对话和 Studio 宽度"
          onPointerDown={(event) => startPanelResize(event, "chat-studio", gridRef, panelLayout, setPanelLayout)}
        />
        <StudioPanel
          artifacts={artifacts}
          sources={sources}
          selectedSources={selectedSources}
          activeArtifact={activeArtifact}
          artifactGenerationLocked={artifactGenerationLocked}
          artifactGenerationLockReason={artifactGenerationLockReason}
          onCreateMindMap={handleCreateMindMap}
          onCreateDataTable={handleCreateDataTable}
          onOpenArtifact={openArtifact}
          onRenameArtifact={handleRenameArtifact}
          onDeleteArtifact={handleDeleteArtifact}
          onOpenSource={handleOpenSource}
          onBack={backToStudioList}
        />
      </main>
      <footer className="statusbar">{status}</footer>
      <SettingsDialog
        open={settingsOpen}
        settings={settings}
        onClose={() => setSettingsOpen(false)}
        onSave={(next) => setSettings({ ...defaultSettings, ...next })}
      />
      {authDialogMode ? (
        <AuthDialog mode={authDialogMode} onClose={() => setAuthDialogMode(null)} />
      ) : null}
      {adminOpen && auth.user?.role === "admin" ? (
        <AdminDialog
          settings={settings}
          onClose={() => setAdminOpen(false)}
          onAnnouncement={(announcement) => setAnnouncements((items) => [announcement, ...items])}
        />
      ) : null}
      {profileOpen && auth.user ? <ProfileDialog user={auth.user} onClose={() => setProfileOpen(false)} /> : null}
    </div>
  );
}

function AccountMenu({
  user,
  onLogin,
  onRegister,
  onProfile,
  onAdmin,
  onLogout,
}: {
  user: AuthUser | null;
  onLogin: () => void;
  onRegister: () => void;
  onProfile: () => void;
  onAdmin: () => void;
  onLogout: () => void;
}) {
  return (
    <div className="account-menu" role="menu">
      {user ? (
        <>
          <div className="account-card">
            <UserRound size={18} />
            <div>
              <strong>{user.display_name}</strong>
              <span>{user.username} · {user.role === "admin" ? "管理员" : "普通用户"}</span>
            </div>
          </div>
          <button type="button" role="menuitem" onClick={onProfile}>
            <UserRound size={16} />
            个人信息
          </button>
          {user.role === "admin" ? (
            <button type="button" role="menuitem" onClick={onAdmin}>
              <Shield size={16} />
              管理员控制台
            </button>
          ) : null}
          <button type="button" role="menuitem" onClick={onLogout}>
            <LogOut size={16} />
            登出
          </button>
        </>
      ) : (
        <>
          <button type="button" role="menuitem" onClick={onLogin}>
            <LogIn size={16} />
            登录
          </button>
          <button type="button" role="menuitem" onClick={onRegister}>
            <UserRound size={16} />
            注册
          </button>
        </>
      )}
    </div>
  );
}

function AnnouncementBar({ announcement }: { announcement: Announcement | null }) {
  if (!announcement) {
    return <div className="announcement-bar is-empty" aria-hidden="true" />;
  }
  return (
    <div className="announcement-bar">
      <Megaphone size={16} />
      <strong>{announcement.title}</strong>
      <span>{announcement.content}</span>
    </div>
  );
}

function AuthDialog({ mode, onClose }: { mode: "login" | "register"; onClose: () => void }) {
  const auth = useAuth();
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const isRegister = mode === "register";

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      if (isRegister) {
        await auth.register(username, password, displayName);
      } else {
        await auth.login(username, password);
      }
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "操作失败");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <form className="auth-dialog" role="dialog" aria-modal="true" onSubmit={submit} onMouseDown={(event) => event.stopPropagation()}>
        <h2>{isRegister ? "注册账号" : "登录账号"}</h2>
        <label>
          用户名
          <input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" />
        </label>
        {isRegister ? (
          <label>
            显示名称
            <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} autoComplete="name" />
          </label>
        ) : null}
        <label>
          密码
          <input
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            type="password"
            autoComplete={isRegister ? "new-password" : "current-password"}
          />
        </label>
        {error ? <p className="error-text">{error}</p> : null}
        <div className="dialog-actions">
          <button type="button" onClick={onClose}>取消</button>
          <button type="submit" disabled={submitting || !username.trim() || !password}>
            {submitting ? "处理中..." : isRegister ? "注册并登录" : "登录"}
          </button>
        </div>
      </form>
    </div>
  );
}

function ProfileDialog({ user, onClose }: { user: AuthUser; onClose: () => void }) {
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="profile-dialog" role="dialog" aria-modal="true" onMouseDown={(event) => event.stopPropagation()}>
        <div className="admin-dialog-header">
          <h2>个人信息</h2>
          <button type="button" onClick={onClose}>关闭</button>
        </div>
        <dl className="profile-list">
          <div>
            <dt>显示名称</dt>
            <dd>{user.display_name}</dd>
          </div>
          <div>
            <dt>用户名</dt>
            <dd>{user.username}</dd>
          </div>
          <div>
            <dt>角色</dt>
            <dd>{user.role === "admin" ? "管理员" : "普通用户"}</dd>
          </div>
          <div>
            <dt>数据空间</dt>
            <dd>{user.tenant_id}</dd>
          </div>
          <div>
            <dt>最近登录</dt>
            <dd>{user.last_login_at ? formatTime(user.last_login_at) : "当前会话"}</dd>
          </div>
        </dl>
      </section>
    </div>
  );
}

function AdminDialog({
  settings,
  onClose,
  onAnnouncement,
}: {
  settings: Settings;
  onClose: () => void;
  onAnnouncement: (announcement: Announcement) => void;
}) {
  const [users, setUsers] = useState<AuthUser[]>([]);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    listAdminUsers(settings)
      .then(setUsers)
      .catch((err) => setError(err instanceof Error ? err.message : "加载用户失败"));
  }, [settings]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    try {
      const announcement = await publishAnnouncement(settings, { title, content });
      onAnnouncement(announcement);
      setTitle("");
      setContent("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "发布公告失败");
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="admin-dialog" role="dialog" aria-modal="true" onMouseDown={(event) => event.stopPropagation()}>
        <div className="admin-dialog-header">
          <h2>管理员控制台</h2>
          <button type="button" onClick={onClose}>关闭</button>
        </div>
        <form className="announcement-form" onSubmit={submit}>
          <label>
            公告标题
            <input value={title} onChange={(event) => setTitle(event.target.value)} />
          </label>
          <label>
            公告内容
            <textarea value={content} onChange={(event) => setContent(event.target.value)} rows={4} />
          </label>
          <button type="submit" disabled={!title.trim() || !content.trim()}>发布公告</button>
        </form>
        {error ? <p className="error-text">{error}</p> : null}
        <div className="admin-users">
          <h3>用户列表</h3>
          <table>
            <thead>
              <tr>
                <th>用户名</th>
                <th>角色</th>
                <th>Tenant</th>
                <th>最近登录</th>
              </tr>
            </thead>
            <tbody>
              {users.map((user) => (
                <tr key={user.id}>
                  <td>{user.display_name} / {user.username}</td>
                  <td>{user.role === "admin" ? "管理员" : "普通用户"}</td>
                  <td>{user.tenant_id}</td>
                  <td>{user.last_login_at ? formatTime(user.last_login_at) : "未登录"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function normalizeMessage(message: ChatMessage): ChatMessage {
  return {
    ...message,
    requestId: message.requestId,
    citations: message.citations || [],
    status: message.status || "done",
  };
}

function inferConversationTitle(messages: ChatMessage[]) {
  const firstUser = messages.find((message) => message.role === "user")?.content.trim();
  return firstUser ? firstUser.slice(0, 40) : "未命名对话";
}

function settingsEqual(left: Settings, right: Settings) {
  return (
    left.apiBaseUrl === right.apiBaseUrl &&
    left.token === right.token &&
    left.tenantId === right.tenantId &&
    left.aclGroups.join(",") === right.aclGroups.join(",")
  );
}

function formatTime(value: number) {
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(value);
}

async function waitForArtifact(settings: Settings, artifactId: string): Promise<MindMapArtifact> {
  const deadline = Date.now() + 180_000;
  while (Date.now() < deadline) {
    await new Promise((resolve) => window.setTimeout(resolve, 1200));
    const artifact = await getArtifact(settings, artifactId);
    if (artifact.status === "ready") {
      return artifact;
    }
    if (artifact.status === "failed") {
      throw new Error(artifact.error || "生成失败");
    }
  }
  throw new Error("生成超时，请稍后在 Studio 列表中查看结果");
}

async function waitForSourcesReady(settings: Settings, pendingSources: SourceItem[]): Promise<SourceItem[]> {
  const pendingTitles = new Set(pendingSources.map((source) => source.title));
  const deadline = Date.now() + 180_000;
  while (Date.now() < deadline) {
    await new Promise((resolve) => window.setTimeout(resolve, 1500));
    const rows = await listSources(settings);
    const related = rows.filter((source) => pendingTitles.has(source.title));
    if (related.some((source) => source.status === "ready" || source.status === "failed")) {
      return rows;
    }
  }
  return listSources(settings);
}

function mergeSelectedState(next: SourceItem[], current: SourceItem[]) {
  const selected = new Map(current.map((item) => [item.doc_id, item.selected ?? item.current]));
  const merged = next.map((item) => ({
    ...item,
    selected: selected.get(item.doc_id) ?? item.current,
  }));
  const activeTasks = current.filter(
    (item) => item.status !== "ready" && !merged.some((nextItem) => nextItem.doc_id === item.doc_id),
  );
  return [...activeTasks, ...merged];
}

function ResizeDivider({
  label,
  onPointerDown,
}: {
  label: string;
  onPointerDown: (event: ReactPointerEvent<HTMLButtonElement>) => void;
}) {
  return (
    <button
      className="panel-resizer"
      type="button"
      aria-label={label}
      title={label}
      role="separator"
      aria-orientation="vertical"
      onPointerDown={onPointerDown}
    />
  );
}

function startPanelResize(
  event: ReactPointerEvent<HTMLButtonElement>,
  handle: ResizeHandle,
  gridRef: RefObject<HTMLElement | null>,
  initialLayout: PanelLayout,
  setPanelLayout: Dispatch<SetStateAction<PanelLayout>>,
) {
  const grid = gridRef.current;
  if (!grid) return;
  const bounds = grid.getBoundingClientRect();
  const startX = event.clientX;
  const pointerId = event.pointerId;
  const width = Math.max(1, bounds.width);
  const target = event.currentTarget;
  target.setPointerCapture(pointerId);
  document.body.classList.add("is-resizing-panels");

  function applyDelta(clientX: number) {
    const deltaPercent = ((clientX - startX) / width) * 100;
    setPanelLayout(normalizePanelLayout(applyResizeDelta(initialLayout, handle, deltaPercent)));
  }

  function onPointerMove(moveEvent: PointerEvent) {
    applyDelta(moveEvent.clientX);
  }

  function onPointerUp(upEvent: PointerEvent) {
    applyDelta(upEvent.clientX);
    document.body.classList.remove("is-resizing-panels");
    if (target.hasPointerCapture(pointerId)) {
      target.releasePointerCapture(pointerId);
    }
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup", onPointerUp);
    window.removeEventListener("pointercancel", onPointerUp);
  }

  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
  window.addEventListener("pointercancel", onPointerUp);
}

function applyResizeDelta(layout: PanelLayout, handle: ResizeHandle, delta: number): PanelLayout {
  if (handle === "source-chat") {
    return {
      ...layout,
      source: layout.source + delta,
      chat: layout.chat - delta,
    };
  }
  return {
    ...layout,
    chat: layout.chat + delta,
    studio: layout.studio - delta,
  };
}

function normalizePanelLayout(layout: PanelLayout): PanelLayout {
  const source = Math.max(MIN_LAYOUT.source, layout.source);
  const chat = Math.max(MIN_LAYOUT.chat, layout.chat);
  const studio = Math.max(MIN_LAYOUT.studio, layout.studio);
  const total = source + chat + studio;
  return {
    source: roundRatio((source / total) * 100),
    chat: roundRatio((chat / total) * 100),
    studio: roundRatio((studio / total) * 100),
  };
}

function roundRatio(value: number) {
  return Math.round(value * 10) / 10;
}
