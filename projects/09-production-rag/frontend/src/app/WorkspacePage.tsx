import { useEffect, useMemo, useRef, useState } from "react";
import type { Dispatch, FormEvent, PointerEvent as ReactPointerEvent, RefObject, SetStateAction } from "react";
import { ArrowLeft, Ban, Check, CheckCircle2, Copy, DatabaseZap, Eye, EyeOff, LogIn, LogOut, Megaphone, Settings as SettingsIcon, Shield, UserRound, X } from "lucide-react";
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
  renameSource,
  getConversation,
  getAdminSettings,
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
  changeCurrentPassword,
  saveConversation,
  sendFeedback,
  updateAdminUserStatus,
  updateCurrentUser,
  updateRegistrationEnabled,
  uploadSource,
} from "../lib/api";
import { useAuth } from "../lib/AuthContext";
import {
  DEFAULT_WORKSPACE_NAME,
  createUserWorkspaceRecord,
  defaultSettings,
  loadActiveWorkspaceId,
  loadSettings,
  loadUserWorkspaces,
  saveActiveWorkspaceId,
  saveSettings,
  saveWorkspaceName,
  saveWorkspaces,
} from "../lib/storage";
import type { AdminSettings, Announcement, AuthUser, ChatMessage, Conversation, MindMapArtifact, Settings, SourceContent, SourceItem, WorkspaceRecord } from "../lib/types";

type PanelLayout = {
  source: number;
  chat: number;
  studio: number;
};

type ResizeHandle = "source-chat" | "chat-studio";
type AppView = "workspace" | "profile" | "admin";

const DEFAULT_LAYOUT: PanelLayout = { source: 24, chat: 46, studio: 30 };
const MINDMAP_LAYOUT: PanelLayout = { source: 20, chat: 38, studio: 42 };
const MIN_LAYOUT: PanelLayout = { source: 17, chat: 31, studio: 22 };
const ARTIFACT_GENERATION_COOLDOWN_MS = 4_000;

export function WorkspacePage({ onNavigate }: { onNavigate: (path: string) => void }) {
  const auth = useAuth();
  const [settings, setSettings] = useState<Settings>(() => loadSettings());
  const [workspaces, setWorkspaces] = useState<WorkspaceRecord[]>(() =>
    loadUserWorkspaces(auth.user?.id ?? null),
  );
  const [activeWorkspaceId, setActiveWorkspaceId] = useState(() =>
    loadActiveWorkspaceId(loadUserWorkspaces(auth.user?.id ?? null), auth.user?.id ?? null),
  );
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [accountMenuOpen, setAccountMenuOpen] = useState(false);
  const [activeView, setActiveView] = useState<AppView>("workspace");
  const [announcementOpen, setAnnouncementOpen] = useState(false);
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
  const accountMenuRef = useRef<HTMLDivElement | null>(null);
  const studioListLayoutRef = useRef<PanelLayout | null>(null);
  const artifactGenerationBusyRef = useRef(false);
  const artifactGenerationReadyAtRef = useRef(0);
  const suppressAutoConversationLoadRef = useRef(false);
  const authTokenRef = useRef<string | null>(auth.token);
  const refreshRunRef = useRef(0);

  const isAuthenticated = Boolean(auth.user && auth.token);
  const activeWorkspace = workspaces.find((workspace) => workspace.id === activeWorkspaceId) ?? workspaces[0];
  const workspaceName = activeWorkspace?.name || DEFAULT_WORKSPACE_NAME;
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
    const userId = auth.user?.id ?? null;
    const scoped = workspaces.filter((w) => (w.user_id ?? null) === userId);
    saveWorkspaces(scoped.length > 0 ? scoped : loadUserWorkspaces(userId), userId);
  }, [workspaces, auth.user?.id]);

  useEffect(() => {
    const userId = auth.user?.id ?? null;
    saveActiveWorkspaceId(activeWorkspaceId, userId);
    saveWorkspaceName(workspaceName);
  }, [activeWorkspaceId, workspaceName, auth.user?.id]);

  useEffect(() => {
    if (!accountMenuOpen) return;
    function handleOutsideClick(event: MouseEvent) {
      if (!accountMenuRef.current?.contains(event.target as Node)) {
        setAccountMenuOpen(false);
      }
    }
    document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, [accountMenuOpen]);

  useEffect(() => {
    authTokenRef.current = auth.token;
    const userId = auth.user?.id ?? null;
    setSettings((current) => {
      const next = {
        ...current,
        token: auth.token,
        tenantId: auth.user?.tenant_id || defaultSettings.tenantId,
        aclGroups: ["engineering"],
      };
      return settingsEqual(current, next) ? current : next;
    });
    const userWorkspaces = loadUserWorkspaces(userId);
    setWorkspaces((prev) => {
      const existingIds = new Set(prev.map((w) => w.id));
      const incoming = userWorkspaces.filter((w) => !existingIds.has(w.id));
      if (incoming.length > 0) return userWorkspaces;
      const renamed = prev.map(
        (w) => userWorkspaces.find((uw) => uw.id === w.id) ?? { ...w, user_id: userId },
      );
      return renamed.length === prev.length ? prev : renamed;
    });
    setActiveWorkspaceId((prev) => {
      const activeId = loadActiveWorkspaceId(userWorkspaces, userId);
      return activeId || prev || userWorkspaces[0]?.id || "default-workspace";
    });
  }, [auth.token, auth.user?.tenant_id, auth.user?.id]);

  useEffect(() => {
    const userId = auth.user?.id ?? null;
    const userWorkspaces = loadUserWorkspaces(userId);
    setWorkspaces(userWorkspaces);
    setActiveWorkspaceId(loadActiveWorkspaceId(userWorkspaces, userId));
    setSources([]);
    setMessages([]);
    setConversationId(null);
    setConversationTitle("未命名对话");
    setArtifacts([]);
    setActiveArtifact(null);
    setActiveSourceContent(null);
    setActiveView("workspace");
    setBusy(false);
    setTypingMessageId(null);
    suppressAutoConversationLoadRef.current = false;
  }, [auth.user?.id]);

  useEffect(() => {
    const announcement = announcements[0];
    if (!announcement) {
      setAnnouncementOpen(false);
      return;
    }
    const audience = auth.user?.id ?? "guest";
    const key = announcementDismissKey(audience, announcement.id);
    setAnnouncementOpen(sessionStorage.getItem(key) !== "1");
  }, [announcements, auth.user?.id]);

  async function refresh(nextSettings = settings) {
    const runId = ++refreshRunRef.current;
    const isCurrentRefresh = () => runId === refreshRunRef.current && nextSettings.token === authTokenRef.current;
    try {
      await health(nextSettings);
      if (!nextSettings.token) {
        const announcementRows = await listAnnouncements(nextSettings);
        if (!isCurrentRefresh()) return;
        setSources([]);
        setArtifacts([]);
        setAnnouncements(announcementRows);
        setStatus("请先登录后使用知识库服务");
        return;
      }
      if (!isCurrentRefresh()) return;
      setStatus("API 已连接");
      const [sourceRows, artifactRows, announcementRows] = await Promise.all([
        listSources(nextSettings),
        listArtifacts(nextSettings),
        listAnnouncements(nextSettings),
      ]);
      if (!isCurrentRefresh()) return;
      setSources((current) => mergeSelectedState(sourceRows, current));
      setArtifacts(artifactRows);
      setAnnouncements(announcementRows);
      await loadLatestConversation(nextSettings, sourceRows, isCurrentRefresh);
    } catch (error) {
      if (!isCurrentRefresh()) return;
      setStatus(error instanceof Error ? error.message : "连接失败");
    }
  }

  async function loadLatestConversation(
    nextSettings: Settings,
    sourceRows: SourceItem[],
    isCurrentRefresh: () => boolean,
  ) {
    if (suppressAutoConversationLoadRef.current) {
      return;
    }
    const rows = await listConversations(nextSettings);
    if (!isCurrentRefresh()) return;
    if (conversationId || rows.length === 0 || messages.length > 0) {
      return;
    }
    const latest = await getConversation(nextSettings, rows[0].id);
    if (!isCurrentRefresh()) return;
    setConversationId(latest.id);
    setConversationTitle(latest.title);
    const latestMessages = latest.messages.map(normalizeMessage);
    setMessages(latestMessages);
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
    if (hasPendingAssistant(latestMessages)) {
      void resumePendingAnswer({ ...latest, messages: latestMessages }, nextSettings);
    }
  }

  async function handleUpload(file: File) {
    if (!isAuthenticated) {
      onNavigate("/login");
      return;
    }
    const sourceKeysBeforeUpload = new Set(sources.map(sourceStateKey));
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
      const readyRows = await waitForSourcesReady(settings, uploaded, sourceKeysBeforeUpload);
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
    const deletingKey = sourceStateKey(source);
    setSources((items) => items.filter((item) => sourceStateKey(item) !== deletingKey));
    if (activeSourceContent?.doc_id === source.doc_id && activeSourceContent.doc_version === source.doc_version) {
      setActiveSourceContent(null);
    }
    try {
      await deleteSource(settings, source.doc_id, source.doc_version);
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
      const content = await getSourceContent(settings, source.doc_id, source.doc_version);
      setActiveSourceContent(content);
    } catch (error) {
      setSourceContentError(error instanceof Error ? error.message : "加载来源内容失败");
    } finally {
      setSourceContentLoading(false);
    }
  }

  async function handleAsk(query: string) {
    if (!query.trim()) return;
    if (!isAuthenticated) {
      onNavigate("/login");
      return;
    }
    const requestToken = authTokenRef.current;
    const isCurrentSession = () => Boolean(requestToken) && authTokenRef.current === requestToken;
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
    let targetConversationId = conversationId;
    try {
      const savedPending = await persistConversation(baseMessages);
      if (!isCurrentSession()) return;
      targetConversationId = savedPending?.id ?? targetConversationId;
      const response = await queryRag(settings, {
        query,
        docIds: selectedDocIds,
        history: messages.map((message) => `${message.role}: ${message.content}`).slice(-8),
      });
      if (!isCurrentSession()) return;
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
      await persistConversation(nextMessages, targetConversationId, selectedDocIds);
    } catch (error) {
      if (!isCurrentSession()) return;
      if (isFetchInterrupted(error)) {
        return;
      }
      const failedMessages: ChatMessage[] = baseMessages.map((item) =>
          item.id === pending.id
            ? { ...item, content: error instanceof Error ? error.message : "回答失败", status: "failed" as const }
            : item,
      );
      setMessages(failedMessages);
      setTypingMessageId(null);
      await persistConversation(failedMessages, targetConversationId, selectedDocIds);
    } finally {
      if (isCurrentSession()) {
        setBusy(false);
      }
    }
  }

  async function resumePendingAnswer(conversation: Conversation, nextSettings: Settings) {
    const requestToken = nextSettings.token;
    const isCurrentSession = () => Boolean(requestToken) && authTokenRef.current === requestToken;
    const pendingIndex = conversation.messages.findIndex(
      (message) => message.role === "assistant" && message.status === "sending",
    );
    if (pendingIndex < 0) return;
    const userIndex = findPreviousUserMessageIndex(conversation.messages, pendingIndex);
    if (userIndex < 0) return;
    const pending = conversation.messages[pendingIndex];
    const userMessage = conversation.messages[userIndex];
    setBusy(true);
    setTypingMessageId(null);
    try {
      const response = await queryRag(nextSettings, {
        query: userMessage.content,
        docIds: conversation.source_doc_ids,
        history: conversation.messages
          .slice(0, userIndex)
          .map((message) => `${message.role}: ${message.content}`)
          .slice(-8),
      });
      if (!isCurrentSession()) return;
      const nextMessages: ChatMessage[] = conversation.messages.map((item, index) =>
        index === pendingIndex
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
      await persistConversation(nextMessages, conversation.id, conversation.source_doc_ids, nextSettings);
    } catch (error) {
      if (!isCurrentSession()) return;
      if (isFetchInterrupted(error)) {
        return;
      }
      const failedMessages: ChatMessage[] = conversation.messages.map((item, index) =>
        index === pendingIndex
          ? {
              ...item,
              content: error instanceof Error ? error.message : "回答恢复失败",
              status: "failed" as const,
            }
          : item,
      );
      setMessages(failedMessages);
      setTypingMessageId(null);
      await persistConversation(failedMessages, conversation.id, conversation.source_doc_ids, nextSettings);
    } finally {
      if (isCurrentSession()) {
        setBusy(false);
      }
    }
  }

  async function persistConversation(
    nextMessages: ChatMessage[],
    targetConversationId = conversationId,
    sourceDocIds = selectedDocIds,
    targetSettings = settings,
  ) {
    if (nextMessages.length === 0) return null;
    const title = inferConversationTitle(nextMessages);
    const saved = await saveConversation(targetSettings, {
      id: targetConversationId,
      title,
      messages: nextMessages,
      sourceDocIds,
    });
    setConversationId(saved.id);
    setConversationTitle(saved.title);
    return saved;
  }

  async function handleDeleteConversation() {
    if (conversationId) {
      await deleteConversation(settings, conversationId);
    }
    suppressAutoConversationLoadRef.current = true;
    setConversationId(null);
    setConversationTitle("未命名对话");
    setMessages([]);
  }

  async function handleFeedback(message: ChatMessage, rating: 1 | -1) {
    if (!message.requestId) return;
    await sendFeedback(settings, message.requestId, rating, selectedDocIds);
    const nextMessages = messages.map((item) =>
      item.id === message.id ? { ...item, feedbackRating: rating } : item,
    );
    setMessages(nextMessages);
    await persistConversation(nextMessages);
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
    const previousTitle = source.title;
    const renamingKey = sourceStateKey(source);
    setSources((items) =>
      items.map((item) => (sourceStateKey(item) === renamingKey ? { ...item, title: newTitle } : item)),
    );
    try {
      const renamed = await renameSource(settings, source.doc_id, newTitle, source.doc_version);
      setSources((items) =>
        items.map((item) => (sourceStateKey(item) === renamingKey ? { ...item, title: renamed.title } : item)),
      );
      if (activeSourceContent?.doc_id === source.doc_id && activeSourceContent.doc_version === source.doc_version) {
        setActiveSourceContent({ ...activeSourceContent, title: renamed.title });
      }
    } catch (error) {
      console.error("Rename source failed:", error);
      setSources((items) =>
        items.map((item) =>
          sourceStateKey(item) === renamingKey
            ? { ...item, title: previousTitle, error: error instanceof Error ? error.message : "重命名失败" }
            : item,
        ),
      );
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
    const anonWorkspaces = loadUserWorkspaces(null);
    saveWorkspaces(anonWorkspaces, null);
    setWorkspaces(anonWorkspaces);
    setActiveWorkspaceId(loadActiveWorkspaceId(anonWorkspaces, null));
    try {
      await auth.logout();
    } catch {
      // If logout API fails, still clear local state
      localStorage.removeItem("production-rag-auth-session");
      window.location.reload();
    }
  }

  function handleNewWorkspace() {
    if (!isAuthenticated || !auth.user?.id) return;
    const nextWorkspace = createUserWorkspaceRecord(`${DEFAULT_WORKSPACE_NAME} ${new Date().toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    })}`, auth.user.id);
    setWorkspaces((items) => [nextWorkspace, ...items]);
    setActiveWorkspaceId(nextWorkspace.id);
    setSources([]);
    setMessages([]);
    setConversationId(null);
    setConversationTitle("未命名对话");
    setArtifacts([]);
    setActiveArtifact(null);
    setActiveSourceContent(null);
  }

  function handleRenameWorkspace(name: string) {
    if (!name.trim() || !isAuthenticated) return;
    const nextName = name.trim();
    setWorkspaces((items) =>
      items.map((workspace) =>
        workspace.id === activeWorkspaceId ? { ...workspace, name: nextName, updated_at: Date.now() } : workspace,
      ),
    );
  }

  function handleSelectWorkspace(id: string) {
    if (id === activeWorkspaceId || !workspaces.some((workspace) => workspace.id === id)) return;
    setActiveWorkspaceId(id);
    setSources([]);
    setMessages([]);
    setConversationId(null);
    setConversationTitle("未命名对话");
    setArtifacts([]);
    setActiveArtifact(null);
    setActiveSourceContent(null);
    suppressAutoConversationLoadRef.current = false;
    void refresh(settings);
  }

  function closeAnnouncement() {
    const announcement = announcements[0];
    if (announcement) {
      sessionStorage.setItem(announcementDismissKey(auth.user?.id ?? "guest", announcement.id), "1");
    }
    setAnnouncementOpen(false);
  }

  return (
    <div className="workspace-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            <DatabaseZap size={22} />
          </span>
          <span>{workspaceName}</span>
        </div>
        <div className="topbar-actions">
          <IconButton label="设置" onClick={() => setSettingsOpen(true)}>
            <SettingsIcon size={18} />
          </IconButton>
          <div className="account-menu-anchor" ref={accountMenuRef}>
            <button
              type="button"
              className={`avatar ${auth.user?.avatar_url ? "has-image" : "no-image"}`}
              aria-label="用户头像"
              title={auth.user ? auth.user.display_name : "账户"}
              onClick={() => setAccountMenuOpen((open) => !open)}
            >
              {auth.user?.avatar_url ? (
                <img src={auth.user.avatar_url} alt="" />
              ) : auth.user ? (
                auth.user.display_name.slice(0, 1).toUpperCase()
              ) : (
                <UserRound size={17} />
              )}
            </button>
            {accountMenuOpen ? (
              <AccountMenu
                user={auth.user}
                onLogin={() => {
                  setAccountMenuOpen(false);
                  onNavigate("/login");
                }}
                onRegister={() => {
                  setAccountMenuOpen(false);
                  onNavigate("/register");
                }}
                onProfile={() => {
                  setAccountMenuOpen(false);
                  setActiveView("profile");
                }}
                onAdmin={() => {
                  setAccountMenuOpen(false);
                  setActiveView("admin");
                }}
                onLogout={handleLogout}
              />
            ) : null}
          </div>
        </div>
      </header>
      {activeView === "profile" && auth.user ? (
        <ProfilePage user={auth.user} settings={settings} onBack={() => setActiveView("workspace")} />
      ) : activeView === "admin" && auth.user?.role === "admin" ? (
        <AdminPage
          settings={settings}
          currentUser={auth.user}
          onBack={() => setActiveView("workspace")}
          onAnnouncement={(announcement) => {
            setAnnouncements((items) => [announcement, ...items]);
            setAnnouncementOpen(true);
          }}
        />
      ) : (
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
            authenticated={isAuthenticated}
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
      )}
      <footer className="statusbar">{status}</footer>
      <SettingsDialog
        open={settingsOpen}
        workspaceName={workspaceName}
        workspaces={workspaces}
        activeWorkspaceId={activeWorkspaceId}
        authenticated={isAuthenticated}
        onClose={() => setSettingsOpen(false)}
        onNewWorkspace={handleNewWorkspace}
        onRenameWorkspace={handleRenameWorkspace}
        onSelectWorkspace={handleSelectWorkspace}
      />
      {announcementOpen && announcements[0] ? (
        <AnnouncementModal announcement={announcements[0]} onClose={closeAnnouncement} />
      ) : null}
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
            <AccountAvatar user={user} />
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

function AccountAvatar({ user }: { user: AuthUser }) {
  return user.avatar_url ? (
    <img className="account-avatar-image" src={user.avatar_url} alt="" />
  ) : (
    <span className="account-avatar-fallback">{user.display_name.slice(0, 1).toUpperCase()}</span>
  );
}

function AnnouncementModal({ announcement, onClose }: { announcement: Announcement; onClose: () => void }) {
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="announcement-modal" role="dialog" aria-modal="true" onMouseDown={(event) => event.stopPropagation()}>
        <button className="close-button" type="button" aria-label="关闭公告" onClick={onClose}>
          <X size={18} />
        </button>
        <div className="announcement-modal-icon">
          <Megaphone size={22} />
        </div>
        <h2>{announcement.title}</h2>
        <p>{announcement.content}</p>
        <span>{announcement.author_name || "系统公告"} · {formatDateTime(announcement.created_at)}</span>
      </section>
    </div>
  );
}

function ProfilePage({ user, settings, onBack }: { user: AuthUser; settings: Settings; onBack: () => void }) {
  const auth = useAuth();
  const [username, setUsername] = useState(user.username);
  const [displayName, setDisplayName] = useState(user.display_name);
  const [avatarUrl, setAvatarUrl] = useState(user.avatar_url || "");
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [loginLinkVisible, setLoginLinkVisible] = useState(false);
  const [loginLinkCopied, setLoginLinkCopied] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [savingPassword, setSavingPassword] = useState(false);
  const loginLink = useMemo(() => buildLoginLink(auth.token), [auth.token]);

  useEffect(() => {
    setUsername(user.username);
    setDisplayName(user.display_name);
    setAvatarUrl(user.avatar_url || "");
  }, [user]);

  async function saveProfile(event: FormEvent) {
    event.preventDefault();
    setError("");
    setMessage("");
    setSavingProfile(true);
    try {
      const updated = await updateCurrentUser(settings, { username, displayName, avatarUrl });
      auth.setUser(updated);
      setMessage("个人信息已保存");
    } catch (err) {
      setError(err instanceof Error ? err.message : "保存失败");
    } finally {
      setSavingProfile(false);
    }
  }

  async function savePassword(event: FormEvent) {
    event.preventDefault();
    setError("");
    setMessage("");
    setSavingPassword(true);
    try {
      await changeCurrentPassword(settings, { currentPassword, newPassword });
      setCurrentPassword("");
      setNewPassword("");
      setMessage("密码已更新");
    } catch (err) {
      setError(err instanceof Error ? err.message : "修改密码失败");
    } finally {
      setSavingPassword(false);
    }
  }

  async function copyLoginLink() {
    if (!loginLink) return;
    await navigator.clipboard.writeText(loginLink);
    setLoginLinkCopied(true);
    window.setTimeout(() => setLoginLinkCopied(false), 1600);
  }

  return (
    <main className="account-page">
      <div className="page-heading">
        <button className="icon-button" type="button" aria-label="返回工作台" onClick={onBack}>
          <ArrowLeft size={18} />
        </button>
        <div>
          <h1>个人信息</h1>
          <p>管理账号资料、头像和登录密码。</p>
        </div>
      </div>
      <section className="account-page-grid">
        <form className="account-section" onSubmit={saveProfile}>
          <div className="profile-hero">
            <div className="profile-avatar">
              {avatarUrl ? <img src={avatarUrl} alt="" /> : <span>{displayName.slice(0, 1).toUpperCase() || "U"}</span>}
            </div>
            <div>
              <strong>{displayName || user.display_name}</strong>
              <span>{user.role === "admin" ? "管理员" : "普通用户"} · {user.status === "banned" ? "已封禁" : "正常"}</span>
            </div>
          </div>
          <label>
            用户名
            <input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" />
          </label>
          <label>
            显示名称
            <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} autoComplete="name" />
          </label>
          <label>
            头像地址
            <input value={avatarUrl} onChange={(event) => setAvatarUrl(event.target.value)} placeholder="https://..." />
          </label>
          <dl className="profile-list">
            <div>
              <dt>数据空间</dt>
              <dd>{user.tenant_id}</dd>
            </div>
            <div>
              <dt>注册日期</dt>
              <dd>{formatDateTime(user.created_at)}</dd>
            </div>
            <div>
              <dt>最近登录</dt>
              <dd>{user.last_login_at ? formatDateTime(user.last_login_at) : "当前会话"}</dd>
            </div>
          </dl>
          <button className="primary-pill" type="submit" disabled={savingProfile || !username.trim() || !displayName.trim()}>
            {savingProfile ? "保存中..." : "保存资料"}
          </button>
        </form>
        <form className="account-section" onSubmit={savePassword}>
          <h2>修改密码</h2>
          <label>
            当前密码
            <input type="password" value={currentPassword} onChange={(event) => setCurrentPassword(event.target.value)} autoComplete="current-password" />
          </label>
          <label>
            新密码
            <input type="password" value={newPassword} onChange={(event) => setNewPassword(event.target.value)} autoComplete="new-password" />
          </label>
          <button className="primary-pill" type="submit" disabled={savingPassword || !currentPassword || newPassword.length < 8}>
            {savingPassword ? "更新中..." : "更新密码"}
          </button>
          {message ? <p className="success-text">{message}</p> : null}
          {error ? <p className="error-text">{error}</p> : null}
        </form>
        <section className="account-section login-link-section">
          <h2>专属登录链接</h2>
          <p className="muted-text">请勿分享给别人。</p>
          <label>
            专属登录链接
            <div className="secret-field">
              <input
                readOnly
                value={loginLinkVisible ? loginLink : "********"}
                aria-label="专属登录链接"
              />
              <button
                type="button"
                className="icon-button"
                aria-label={loginLinkVisible ? "隐藏专属登录链接" : "显示专属登录链接"}
                onClick={() => setLoginLinkVisible((visible) => !visible)}
              >
                {loginLinkVisible ? <Eye size={16} /> : <EyeOff size={16} />}
              </button>
              <button
                type="button"
                className="icon-button"
                aria-label="复制专属登录链接"
                onClick={copyLoginLink}
                disabled={!loginLink}
              >
                {loginLinkCopied ? <Check size={16} /> : <Copy size={16} />}
              </button>
            </div>
          </label>
        </section>
      </section>
    </main>
  );
}

function AdminPage({
  settings,
  currentUser,
  onBack,
  onAnnouncement,
}: {
  settings: Settings;
  currentUser: AuthUser;
  onBack: () => void;
  onAnnouncement: (announcement: Announcement) => void;
}) {
  const [users, setUsers] = useState<AuthUser[]>([]);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [error, setError] = useState("");
  const [busyUserId, setBusyUserId] = useState<string | null>(null);
  const [adminSettings, setAdminSettings] = useState<AdminSettings | null>(null);
  const [settingsBusy, setSettingsBusy] = useState(false);

  useEffect(() => {
    Promise.all([listAdminUsers(settings), getAdminSettings(settings)])
      .then(([userRows, nextSettings]) => {
        setUsers(userRows);
        setAdminSettings(nextSettings);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "加载管理员控制台失败"));
  }, [settings]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    try {
      const announcement = await publishAnnouncement(settings, { title, content });
      onAnnouncement(announcement);
      setAdminSettings((current) =>
        current
          ? { ...current, latest_announcement: announcement }
          : { registration_enabled: true, latest_announcement: announcement },
      );
      setTitle("");
      setContent("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "发布公告失败");
    }
  }

  async function toggleUserStatus(user: AuthUser) {
    setError("");
    setBusyUserId(user.id);
    try {
      const nextStatus = user.status === "banned" ? "active" : "banned";
      const updated = await updateAdminUserStatus(settings, user.id, nextStatus);
      setUsers((items) => items.map((item) => (item.id === updated.id ? updated : item)));
    } catch (err) {
      setError(err instanceof Error ? err.message : "更新用户状态失败");
    } finally {
      setBusyUserId(null);
    }
  }

  async function toggleRegistrationEnabled() {
    if (!adminSettings) {
      return;
    }
    setError("");
    setSettingsBusy(true);
    try {
      const updated = await updateRegistrationEnabled(settings, !adminSettings.registration_enabled);
      setAdminSettings(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : "更新注册设置失败");
    } finally {
      setSettingsBusy(false);
    }
  }

  return (
    <main className="account-page admin-page">
      <div className="page-heading">
        <button className="icon-button" type="button" aria-label="返回工作台" onClick={onBack}>
          <ArrowLeft size={18} />
        </button>
        <div>
          <h1>管理员控制台</h1>
          <p>发布公告、查看用户并管理普通账号状态。</p>
        </div>
      </div>
      <section className="account-page-grid admin-page-grid">
        <section className="announcement-form admin-settings-card">
          <h2>系统设置</h2>
          <div className="admin-setting-row">
            <div>
              <strong>新用户注册</strong>
              <p>{adminSettings?.registration_enabled ? "当前允许新用户自行注册。" : "当前已关闭新用户注册。"}</p>
            </div>
            <button
              type="button"
              className={`toggle-switch ${adminSettings?.registration_enabled ? "is-on" : ""}`}
              role="switch"
              aria-checked={Boolean(adminSettings?.registration_enabled)}
              disabled={!adminSettings || settingsBusy}
              onClick={toggleRegistrationEnabled}
            >
              <span className="toggle-track">
                <span />
              </span>
              <span className="toggle-label">{adminSettings?.registration_enabled ? "允许注册" : "关闭注册"}</span>
            </button>
          </div>
        </section>
        <form className="announcement-form" onSubmit={submit}>
          <h2>发布公告</h2>
          <label>
            公告标题
            <input value={title} onChange={(event) => setTitle(event.target.value)} />
          </label>
          <label>
            公告内容
            <textarea value={content} onChange={(event) => setContent(event.target.value)} rows={4} />
          </label>
          <button type="submit" disabled={!title.trim() || !content.trim()}>发布公告</button>
          <div className="latest-announcement">
            <span>上一次公告</span>
            {adminSettings?.latest_announcement ? (
              <>
                <strong>{adminSettings.latest_announcement.title}</strong>
                <p>{adminSettings.latest_announcement.content}</p>
                <small>
                  {adminSettings.latest_announcement.author_name || "系统公告"} · {formatDateTime(adminSettings.latest_announcement.created_at)}
                </small>
              </>
            ) : (
              <p>暂无公告</p>
            )}
          </div>
        </form>
        {error ? <p className="error-text">{error}</p> : null}
        <div className="admin-users">
          <h3>用户列表</h3>
          <table>
            <thead>
              <tr>
                <th>用户名</th>
                <th>角色</th>
                <th>状态</th>
                <th>注册日期</th>
                <th>Tenant</th>
                <th>最近登录</th>
                <th>操作</th>
              </tr>
            </thead>
            <tbody>
              {users.map((user) => (
                <tr key={user.id}>
                  <td>{user.display_name} / {user.username}</td>
                  <td>{user.role === "admin" ? "管理员" : "普通用户"}</td>
                  <td>{user.status === "banned" ? "已封禁" : "正常"}</td>
                  <td>{formatDateTime(user.created_at)}</td>
                  <td>{user.tenant_id}</td>
                  <td>{user.last_login_at ? formatDateTime(user.last_login_at) : "未登录"}</td>
                  <td>
                    {user.role === "admin" || user.id === currentUser.id ? (
                      <span className="muted-text">不可操作</span>
                    ) : (
                      <button
                        className={user.status === "banned" ? "inline-action" : "inline-action danger"}
                        type="button"
                        disabled={busyUserId === user.id}
                        onClick={() => toggleUserStatus(user)}
                      >
                        {user.status === "banned" ? <CheckCircle2 size={15} /> : <Ban size={15} />}
                        {user.status === "banned" ? "解封" : "封禁"}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
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

function hasPendingAssistant(messages: ChatMessage[]) {
  return messages.some((message) => message.role === "assistant" && message.status === "sending");
}

function findPreviousUserMessageIndex(messages: ChatMessage[], fromIndex: number) {
  for (let index = fromIndex - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "user") {
      return index;
    }
  }
  return -1;
}

function isFetchInterrupted(error: unknown) {
  return error instanceof TypeError && error.message === "Failed to fetch";
}

function settingsEqual(left: Settings, right: Settings) {
  return (
    left.apiBaseUrl === right.apiBaseUrl &&
    left.token === right.token &&
    left.tenantId === right.tenantId &&
    left.aclGroups.join(",") === right.aclGroups.join(",")
  );
}

function buildLoginLink(token: string) {
  if (!token) return "";
  const { origin, pathname, search } = window.location;
  return `${origin}${pathname}${search}#token=${encodeURIComponent(token)}`;
}

function formatDateTime(value: number) {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(value);
}

function announcementDismissKey(audience: string, announcementId: string) {
  return `announcement-dismissed:${audience}:${announcementId}`;
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

async function waitForSourcesReady(
  settings: Settings,
  pendingSources: SourceItem[],
  sourceKeysBeforeUpload: Set<string>,
): Promise<SourceItem[]> {
  const pendingIds = new Set(pendingSources.map((source) => source.doc_id));
  const pendingUris = new Set(pendingSources.map((source) => source.source_uri));
  const pendingTitles = new Set(pendingSources.map((source) => source.title));
  const deadline = Date.now() + 180_000;
  while (Date.now() < deadline) {
    await new Promise((resolve) => window.setTimeout(resolve, 1500));
    const rows = await listSources(settings);
    const pendingRows = rows.filter((source) => pendingIds.has(source.doc_id));
    const newReadyRows = rows.filter(
      (source) =>
        source.status === "ready" &&
        !sourceKeysBeforeUpload.has(sourceStateKey(source)) &&
        (pendingUris.has(source.source_uri) || pendingTitles.has(source.title)),
    );
    if (pendingRows.some((source) => source.status === "failed") || newReadyRows.length > 0) {
      return rows;
    }
  }
  return listSources(settings);
}

function mergeSelectedState(next: SourceItem[], current: SourceItem[]) {
  const selected = new Map(current.map((item) => [sourceStateKey(item), item.selected ?? item.current]));
  const merged = next.map((item) => ({
    ...item,
    selected: selected.get(sourceStateKey(item)) ?? item.current,
  }));
  return merged;
}

function sourceStateKey(source: SourceItem) {
  return `${source.doc_id}::${source.doc_version}`;
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
