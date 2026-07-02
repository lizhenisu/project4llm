import { useEffect, useMemo, useRef, useState } from "react";
import type { Dispatch, FormEvent, PointerEvent as ReactPointerEvent, RefObject, SetStateAction } from "react";
import { ArrowLeft, Ban, Check, CheckCircle2, Copy, DatabaseZap, ExternalLink, Eye, EyeOff, Github, LogIn, LogOut, Megaphone, MoreHorizontal, PanelLeftClose, PanelLeftOpen, PencilLine, RefreshCw, Search, Settings as SettingsIcon, Shield, Trash2, UserRound, Users, X } from "lucide-react";
import { ChatPanel } from "../components/chat/ChatPanel";
import { SourcePanel } from "../components/sources/SourcePanel";
import { StudioPanel } from "../components/studio/StudioPanel";
import { IconButton } from "../components/ui/IconButton";
import { SettingsDialog } from "./SettingsDialog";
import {
  createDataTable,
  createMindMap,
  deleteAnnouncement,
  deleteConversation,
  deleteSource,
  deleteArtifact,
  renameArtifact,
  renameSource,
  retrySource,
  getConversation,
  getAdminSettings,
  getArtifact,
  getSourceContent,
  health,
  listAdminUsers,
  listAdminDeadLetters,
  listAdminIngestionAudit,
  listAnnouncements,
  listArtifacts,
  listConversations,
  listSources,
  publishAnnouncement,
  redriveAdminDeadLetters,
  queryRagStream,
  renameConversation,
  changeCurrentPassword,
  refreshLoginToken,
  saveConversation,
  sendFeedback,
  updateAdminUserStatus,
  updateAdminUsers,
  updateCurrentUser,
  updateRegistrationEnabled,
  uploadSource,
} from "../lib/api";
import { useAuth } from "../lib/AuthContext";
import {
  DEFAULT_WORKSPACE_NAME,
  addArtifactToWorkspace,
  addConversationToWorkspace,
  addSourcesToWorkspace,
  createUserWorkspaceRecord,
  defaultSettings,
  deleteWorkspace,
  hasWorkspaceArtifacts,
  hasWorkspaceConversations,
  hasWorkspaceSources,
  initializeEmptyWorkspaceData,
  loadActiveWorkspaceId,
  loadSettings,
  loadUserWorkspaces,
  loadWorkspaceArtifacts,
  loadWorkspaceConversations,
  loadWorkspaceSourceTitles,
  loadWorkspaceSources,
  removeConversationFromWorkspace,
  removeSourcesFromWorkspace,
  saveActiveWorkspaceId,
  saveSettings,
  saveWorkspaceSourceTitle,
  saveWorkspaceName,
  saveWorkspaces,
} from "../lib/storage";
import type { AdminDeadLetterTask, AdminIngestionAuditEvent, AdminSettings, Announcement, AuthUser, ChatMessage, Conversation, ConversationListItem, MindMapArtifact, RagProgressStage, Settings, SourceContent, SourceItem, WorkspaceRecord } from "../lib/types";

type PanelLayout = {
  source: number;
  chat: number;
  studio: number;
};

type ResizeHandle = "source-chat" | "chat-studio";
type AppView = "workspace" | "profile" | "admin";
type AdminSection = "settings" | "announcements" | "users" | "ingestion";
type AdminUserDraft = {
  profileNameEditAllowed: boolean;
  avatarEditAllowed: boolean;
};
const APP_VERSION = import.meta.env.VITE_APP_VERSION || "0.0.0";

const DEFAULT_LAYOUT: PanelLayout = { source: 24, chat: 46, studio: 30 };
const MINDMAP_LAYOUT: PanelLayout = { source: 20, chat: 38, studio: 42 };
const MIN_LAYOUT: PanelLayout = { source: 17, chat: 31, studio: 22 };
const ARTIFACT_GENERATION_COOLDOWN_MS = 4_000;
const ARTIFACT_STATUS_POLL_MS = 1_200;
const ARTIFACT_STATUS_POLL_HIDDEN_MS = 8_000;
const ARTIFACT_STATUS_POLL_MAX_MS = 10_000;
const SOURCE_READY_POLL_BASE_MS = 1_500;
const SOURCE_READY_POLL_MAX_MS = 8_000;
const MAX_FRONTEND_UPLOAD_CONCURRENCY = 2;
const SOURCE_LIST_FRONTEND_COALESCE_MS = 150;
type ArtifactKind = "mindmap" | "table";

let sourceListRequestSlot: { key: string; promise: Promise<SourceItem[]>; expiresAt: number } | null = null;

function initialRagProgress(enabled: boolean): RagProgressStage[] | undefined {
  if (!enabled) {
    return [
      {
        stage: "answer",
        label: "大模型直接回答",
        detail: "等待大模型接收问题。",
        status: "pending",
      },
    ];
  }
  return [
    { stage: "start", label: "接收问题", detail: "等待后端接收问题。", status: "pending" },
  ];
}

function mergeRagProgress(current: RagProgressStage[] | undefined, incoming: RagProgressStage): RagProgressStage[] {
  const stages = current?.length ? [...current] : [];
  const existingIndex = stages.findIndex((item) => item.stage === incoming.stage);
  if (existingIndex >= 0) {
    stages[existingIndex] = { ...stages[existingIndex], ...incoming };
    return stages;
  }
  return [...stages, incoming];
}

function listSourcesCoalesced(settings: Settings): Promise<SourceItem[]> {
  const key = `${settings.apiBaseUrl}|${settings.token}|${settings.tenantId}|${settings.aclGroups.join(",")}`;
  const now = Date.now();
  if (sourceListRequestSlot?.key === key && sourceListRequestSlot.expiresAt > now) {
    return sourceListRequestSlot.promise;
  }
  const promise = listSources(settings);
  sourceListRequestSlot = {
    key,
    promise,
    expiresAt: now + SOURCE_LIST_FRONTEND_COALESCE_MS,
  };
  void promise.finally(() => {
    if (sourceListRequestSlot?.promise === promise) {
      sourceListRequestSlot.expiresAt = Date.now() + SOURCE_LIST_FRONTEND_COALESCE_MS;
      window.setTimeout(() => {
        if (sourceListRequestSlot?.promise === promise && sourceListRequestSlot.expiresAt <= Date.now()) {
          sourceListRequestSlot = null;
        }
      }, SOURCE_LIST_FRONTEND_COALESCE_MS);
    }
  }).catch(() => undefined);
  return promise;
}

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
  const [conversations, setConversations] = useState<ConversationListItem[]>([]);
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
  const [chromeCollapsed, setChromeCollapsed] = useState(false);
  const [typingMessageId, setTypingMessageId] = useState<string | null>(null);
  const gridRef = useRef<HTMLElement | null>(null);
  const accountMenuRef = useRef<HTMLDivElement | null>(null);
  const studioListLayoutRef = useRef<PanelLayout | null>(null);
  const artifactGenerationBusyRef = useRef(false);
  const artifactGenerationReadyAtRef = useRef(0);
  const suppressAutoConversationLoadRef = useRef(false);
  const authTokenRef = useRef<string | null>(auth.token);
  const activeWorkspaceIdRef = useRef(activeWorkspaceId);
  const refreshRunRef = useRef(0);
  const activeUploadCountRef = useRef(0);
  const pendingUploadResolversRef = useRef<Array<() => void>>([]);

  const isAuthenticated = Boolean(auth.user && auth.token);
  const activeWorkspace = workspaces.find((workspace) => workspace.id === activeWorkspaceId) ?? workspaces[0];
  const displayWorkspaceName = isAuthenticated
    ? (activeWorkspace?.name || DEFAULT_WORKSPACE_NAME)
    : "未命名的知识库";
  const workspaceName = activeWorkspace?.name || DEFAULT_WORKSPACE_NAME;
  const selectedSources = useMemo(() => sources.filter((source) => source.selected), [sources]);
  const selectedDocIds = useMemo(
    () =>
      selectedSources.flatMap((source) =>
        source.child_doc_ids && source.child_doc_ids.length > 0 ? source.child_doc_ids : [source.doc_id],
      ),
    [selectedSources],
  );
  const hasServerGeneratingArtifact = useMemo(
    () => artifacts.some((artifact) => artifact.status === "generating" && !isLocalPendingArtifact(artifact.id)),
    [artifacts],
  );
  const artifactCooldownRemainingMs = Math.max(0, artifactGenerationReadyAt - Date.now());
  const artifactGenerationLocked = artifactGenerationBusy || hasServerGeneratingArtifact || artifactCooldownRemainingMs > 0;
  const artifactGenerationLockReason = artifactGenerationBusy || hasServerGeneratingArtifact
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
    activeWorkspaceIdRef.current = activeWorkspaceId;
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
    if (!isAuthenticated || !settings.token || !hasServerGeneratingArtifact) return;
    let cancelled = false;
    let timer: number | undefined;
    let failureCount = 0;
    const workspaceId = activeWorkspaceId;
    const token = settings.token;

    async function pollArtifacts() {
      try {
        const rows = await listArtifacts(settings, workspaceId);
        if (cancelled || token !== authTokenRef.current || workspaceId !== activeWorkspaceIdRef.current) return;
        failureCount = 0;
        setArtifacts((current) => mergePolledArtifacts(current, rows));
        setActiveArtifact((current) => {
          if (!current) return current;
          return rows.find((artifact) => artifact.id === current.id) ?? current;
        });
      } catch (error) {
        failureCount += 1;
        console.error("Artifact status polling failed:", error);
      } finally {
        if (!cancelled && token === authTokenRef.current && workspaceId === activeWorkspaceIdRef.current) {
          timer = window.setTimeout(pollArtifacts, artifactPollDelayMs(failureCount));
        }
      }
    }

    void pollArtifacts();
    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [activeWorkspaceId, hasServerGeneratingArtifact, isAuthenticated, settings]);

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
    setConversations([]);
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
    const dismissedKey = `announcement-dismissed:${key}`;
    const heartbeatKey = `announcement-heartbeat:${key}`;
    const heartbeatInterval = 5_000;
    const heartbeatGapMs = 8_000;

    // Emit heartbeat so we can detect when all tabs are closed.
    const beat = () => localStorage.setItem(heartbeatKey, String(Date.now()));
    beat();
    const interval = window.setInterval(beat, heartbeatInterval);

    const lastBeat = Number(localStorage.getItem(heartbeatKey) || 0);
    const browserSessionExpired = Date.now() - lastBeat > heartbeatGapMs;

    if (browserSessionExpired) {
      // New browser session — clear old tab-local dismissals and show announcement.
      sessionStorage.removeItem(dismissedKey);
      setAnnouncementOpen(true);
    } else if (sessionStorage.getItem(dismissedKey) === "1") {
      // Already dismissed in this tab session (e.g. after refresh).
      setAnnouncementOpen(false);
    } else {
      // Same browser session, first tab that loaded this announcement.
      setAnnouncementOpen(true);
    }

    return () => window.clearInterval(interval);
  }, [announcements, auth.user?.id]);

  // Persist user's explicit source selection to localStorage so refresh doesn't override it.
  useEffect(() => {
    if (isAuthenticated && activeWorkspaceId && sources.length > 0) {
      saveCachedSelection(activeWorkspaceId, sources);
    }
  }, [sources, isAuthenticated, activeWorkspaceId]);

  async function refresh(nextSettings = settings, workspaceId = activeWorkspaceId) {
    const runId = ++refreshRunRef.current;
    const isCurrentRefresh = () => runId === refreshRunRef.current && nextSettings.token === authTokenRef.current;
    try {
      await health(nextSettings);
      if (!nextSettings.token) {
        const announcementRows = await listAnnouncements(nextSettings);
        if (!isCurrentRefresh()) return;
        setSources([]);
        setArtifacts([]);
        setConversations([]);
        setAnnouncements(announcementRows);
        setStatus("请先登录后使用知识库服务");
        return;
      }
      if (!isCurrentRefresh()) return;
      setStatus("API 已连接");
      const [sourceRows, artifactRows, announcementRows, conversationRows] = await Promise.all([
        listSourcesCoalesced(nextSettings),
        listArtifacts(nextSettings, workspaceId),
        listAnnouncements(nextSettings),
        listConversations(nextSettings),
      ]);
      if (!isCurrentRefresh()) return;
      const wSources = loadWorkspaceSources(workspaceId);
      const visibleRows = hasWorkspaceSources(workspaceId)
        ? filterWorkspaceSources(sourceRowsForList(sourceRows), wSources)
        : sourceRowsForList(sourceRows);
      setSources((current) =>
        preservePendingSourceRows(
          mergeSelectedState(applyWorkspaceSourceTitles(visibleRows, workspaceId), current, workspaceId),
          current,
        ),
      );
      const wArtifacts = loadWorkspaceArtifacts(workspaceId);
      const visibleArtifacts = hasWorkspaceArtifacts(workspaceId)
        ? artifactRows.filter((a) => wArtifacts.includes(a.id))
        : artifactRows;
      const workspaceConversationIds = loadWorkspaceConversations(workspaceId);
      const visibleConversations = hasWorkspaceConversations(workspaceId)
        ? conversationRows.filter((row) => workspaceConversationIds.includes(row.id))
        : conversationRows;
      setArtifacts(visibleArtifacts);
      setConversations(visibleConversations);
      setAnnouncements(announcementRows);
      await loadLatestConversation(nextSettings, visibleRows, visibleConversations, isCurrentRefresh, workspaceId);
    } catch (error) {
      if (!isCurrentRefresh()) return;
      setStatus(error instanceof Error ? error.message : "连接失败");
    }
  }

  async function loadLatestConversation(
    nextSettings: Settings,
    sourceRows: SourceItem[],
    conversationRows: ConversationListItem[],
    isCurrentRefresh: () => boolean,
    workspaceId = activeWorkspaceId,
  ) {
    if (suppressAutoConversationLoadRef.current) {
      return;
    }
    if (!isCurrentRefresh()) return;
    if (conversationId || conversationRows.length === 0 || messages.length > 0) {
      return;
    }
    const latest = await getConversation(nextSettings, conversationRows[0].id);
    if (!isCurrentRefresh()) return;
    setConversationId(latest.id);
    setConversationTitle(latest.title);
    const latestMessages = latest.messages.map(normalizeMessage);
    setMessages(latestMessages);
    // Only restore selection from conversation history when there is no user cache yet.
    // If the user has already made explicit selections (cached), mergeSelectedState already
    // handled it in refresh(), so skip the conversation-based override.
    const cachedSelection = loadCachedSelection(workspaceId);
    if (cachedSelection === null) {
      const selectedIds = new Set(latest.source_doc_ids);
      if (selectedIds.size > 0) {
        setSources((current) =>
          (current.length ? current : sourceRows).map((source) => ({
            ...source,
            selected:
              source.status === "ready" && (selectedIds.has(source.doc_id) || Boolean(source.child_doc_ids?.some((docId) => selectedIds.has(docId)))),
          })),
        );
      }
    }
    if (hasPendingAssistant(latestMessages)) {
      void resumePendingAnswer({ ...latest, messages: latestMessages }, nextSettings, workspaceId);
    }
  }

  async function handleUpload(file: File) {
    if (!isAuthenticated) {
      onNavigate("/login");
      return;
    }
    const requestWorkspaceId = activeWorkspaceId;
    const isCurrentWorkspace = () => activeWorkspaceIdRef.current === requestWorkspaceId;
    const sourceKeysBeforeUpload = new Set(sources.map(sourceStateKey));
    const startsImmediately = activeUploadCountRef.current < MAX_FRONTEND_UPLOAD_CONCURRENCY;
    const temp: SourceItem = {
      doc_id: `upload-${crypto.randomUUID()}`,
      title: file.name,
      source_type: file.name.split(".").pop() || "file",
      source_uri: file.name,
      doc_version: 1,
      chunk_count: 0,
      acl_groups: settings.aclGroups,
      status: startsImmediately ? "uploading" : "queued",
      current: false,
      selected: false,
    };
    setSources((items) => [temp, ...items]);
    try {
      await acquireUploadSlot();
      if (isCurrentWorkspace()) {
        setSources((items) =>
          items.map((item) => (item.doc_id === temp.doc_id ? { ...item, status: "uploading" } : item)),
        );
      }
      const uploaded = await uploadSource(settings, file);
      if (isCurrentWorkspace()) {
        setSources((items) => [
          ...uploaded.map(item => ({ ...item, selected: item.status === "ready" })), // Auto-select only ready items
          ...items.filter((item) => item.doc_id !== temp.doc_id)
        ]);
      }
      // Track uploaded sources for the current workspace
      const uploadedIds = uploaded.map((s) => s.doc_id);
      addSourcesToWorkspace(requestWorkspaceId, uploadedIds);
      const readyRows = await waitForSourcesReady(
        settings,
        uploaded,
        sourceKeysBeforeUpload,
        (polledRows) => {
          if (!isCurrentWorkspace()) return;
          const workspaceSources = loadWorkspaceSources(requestWorkspaceId);
          const filteredRows = filterWorkspaceSources(sourceRowsForList(polledRows), workspaceSources);
          setSources((items) =>
            preservePendingSourceRows(
              mergeSelectedState(
                applyWorkspaceSourceTitles(filteredRows, requestWorkspaceId),
                items,
                requestWorkspaceId,
              ),
              items,
            ),
          );
        },
      );
      const resolvedUploads = resolveUploadedSources(readyRows, uploaded, sourceKeysBeforeUpload);
      if (resolvedUploads.length > 0) {
        addSourcesToWorkspace(requestWorkspaceId, sourceIdsForWorkspace(resolvedUploads));
      }
      const wSources = loadWorkspaceSources(requestWorkspaceId);
      const filteredReady = filterWorkspaceSources(sourceRowsForList(readyRows), wSources);
      if (isCurrentWorkspace()) {
        setSources((items) =>
          preservePendingSourceRows(
            mergeSelectedState(applyWorkspaceSourceTitles(filteredReady, requestWorkspaceId), items, requestWorkspaceId),
            items,
          ),
        );
      }
      // Auto-rename workspace if it's still the auto-generated default name
      const currentAutoNamed = activeWorkspace?.auto_named;
      if (currentAutoNamed && uploaded.length > 0) {
        try {
          const content = await getSourceContent(settings, uploaded[0].doc_id, uploaded[0].doc_version);
          if (content.suggested_title && content.suggested_title !== content.title) {
            setWorkspaces((prev) =>
              prev.map((w) =>
                w.id === requestWorkspaceId ? { ...w, name: content.suggested_title!, auto_named: false, updated_at: Date.now() } : w,
              ),
            );
          }
        } catch {
          // Ignore if content fetch fails; auto-rename is a best-effort feature
        }
      }
    } catch (error) {
      if (isCurrentWorkspace()) {
        setSources((items) =>
          items.map((item) =>
            item.doc_id === temp.doc_id
              ? { ...item, status: "failed", error: error instanceof Error ? error.message : "上传失败" }
              : item,
          ),
        );
      }
    } finally {
      releaseUploadSlot();
    }
  }

  async function acquireUploadSlot() {
    if (activeUploadCountRef.current < MAX_FRONTEND_UPLOAD_CONCURRENCY) {
      activeUploadCountRef.current += 1;
      return;
    }
    await new Promise<void>((resolve) => {
      pendingUploadResolversRef.current.push(() => {
        activeUploadCountRef.current += 1;
        resolve();
      });
    });
  }

  function releaseUploadSlot() {
    activeUploadCountRef.current = Math.max(0, activeUploadCountRef.current - 1);
    const next = pendingUploadResolversRef.current.shift();
    if (next) {
      next();
    }
  }

  async function handleDeleteSource(source: SourceItem) {
    const deletingKey = sourceStateKey(source);
    const sourceIds = sourceIdsForWorkspace([source]);
    setSources((items) => items.filter((item) => sourceStateKey(item) !== deletingKey));
    removeSourcesFromWorkspace(activeWorkspaceId, sourceIds);
    if (activeSourceContent?.doc_id === source.doc_id && activeSourceContent.doc_version === source.doc_version) {
      setActiveSourceContent(null);
    }
    try {
      if (!sourceReferencedByOtherWorkspace(activeWorkspaceId, sourceIds, workspaces)) {
        await deleteSource(settings, source.doc_id);
      }
    } catch (error) {
      addSourcesToWorkspace(activeWorkspaceId, sourceIds);
      setSources((items) => [
        { ...source, status: "failed", error: error instanceof Error ? error.message : "删除失败" },
        ...items,
      ]);
    }
  }

  async function handleRetrySource(source: SourceItem) {
    const retryingKey = sourceStateKey(source);
    setSources((items) =>
      items.map((item) =>
        sourceStateKey(item) === retryingKey
          ? { ...item, status: "queued", error: "", retryable: false, updated_at: Date.now() }
          : item,
      ),
    );
    try {
      const response = await retrySource(settings, source.doc_id);
      setSources((items) =>
        items.map((item) =>
          sourceStateKey(item) === retryingKey
            ? { ...response.source, selected: false }
            : item,
        ),
      );
    } catch (error) {
      setSources((items) =>
        items.map((item) =>
          sourceStateKey(item) === retryingKey
            ? {
                ...source,
                retryable: true,
                error: error instanceof Error ? error.message : "重新处理失败",
              }
            : item,
        ),
      );
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

  function createRagProgressFrameUpdater(
    matchesPendingMessage: (message: ChatMessage, index: number) => boolean,
    isCurrentWorkspace: () => boolean,
  ) {
    let frameId: number | null = null;
    let queuedStage: RagProgressStage | null = null;
    let queuedProgress: RagProgressStage[] = [];

    const flush = () => {
      frameId = null;
      const stage = queuedStage;
      if (!stage || !isCurrentWorkspace()) return;
      const progress = queuedProgress;
      setMessages((current) =>
        current.map((item, index) =>
          matchesPendingMessage(item, index)
            ? {
                ...item,
                content: stage.detail || item.content,
                ragProgress: progress,
              }
            : item,
        ),
      );
    };

    return {
      schedule(stage: RagProgressStage, progress: RagProgressStage[]) {
        queuedStage = stage;
        queuedProgress = progress;
        if (frameId !== null) return;
        frameId = window.requestAnimationFrame(flush);
      },
      cancel() {
        if (frameId !== null) {
          window.cancelAnimationFrame(frameId);
          frameId = null;
        }
        queuedStage = null;
        queuedProgress = [];
      },
    };
  }

  async function handleAsk(query: string, imageDataUrl?: string | null) {
    if (!query.trim()) return;
    if (!isAuthenticated) {
      onNavigate("/login");
      return;
    }
    const requestWorkspaceId = activeWorkspaceId;
    const requestConversationId = conversationId;
    const requestSelectedDocIds = [...selectedDocIds];
    const requestHistory = messages.map((message) => `${message.role}: ${message.content}`).slice(-8);
    const requestToken = authTokenRef.current;
    const isCurrentSession = () => Boolean(requestToken) && authTokenRef.current === requestToken;
    const isCurrentWorkspace = () => isCurrentSession() && activeWorkspaceIdRef.current === requestWorkspaceId;
    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: query,
      status: "done",
      created_at: Date.now(),
      imageDataUrl: imageDataUrl || null,
    };
    const pending: ChatMessage = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: requestSelectedDocIds.length > 0 ? "RAG 调用链启动中..." : "大模型正在思考...",
      requestId: crypto.randomUUID(),
      status: "sending",
      created_at: Date.now(),
      ragProgress: initialRagProgress(requestSelectedDocIds.length > 0 || Boolean(imageDataUrl)),
    };
    const baseMessages = [...messages, userMessage, pending];
    setMessages(baseMessages);
    setTypingMessageId(null);
    setBusy(true);
    let targetConversationId = requestConversationId;
    let latestRagProgress = pending.ragProgress || [];
    try {
      const savedPending = await persistConversation(
        baseMessages,
        requestConversationId,
        requestSelectedDocIds,
        settings,
        requestWorkspaceId,
      );
      if (!isCurrentSession()) return;
      targetConversationId = savedPending?.id ?? targetConversationId;
      const progressUpdater = createRagProgressFrameUpdater(
        (item) => item.id === pending.id,
        isCurrentWorkspace,
      );
      const updateRagProgress = (stage: RagProgressStage) => {
        latestRagProgress = mergeRagProgress(latestRagProgress, stage);
        progressUpdater.schedule(stage, latestRagProgress);
      };
      try {
        const response = await queryRagStream(settings, {
          query,
          requestId: pending.requestId!,
          docIds: requestSelectedDocIds,
          history: requestHistory,
          imageDataUrl,
          onEvent: (event) => {
            if (event.type === "stage") {
              updateRagProgress(event);
            }
          },
        });
        progressUpdater.cancel();
        if (!isCurrentSession()) return;
        const nextMessages: ChatMessage[] = baseMessages.map((item) =>
            item.id === pending.id
              ? {
                  ...item,
                  content: response.answer,
                  requestId: response.request_id,
                  citations: response.citations,
                  status: "done" as const,
                  ragProgress: latestRagProgress,
                }
              : item,
        );
        if (isCurrentWorkspace()) {
          setMessages(nextMessages);
          setTypingMessageId(pending.id);
        }
        await persistConversation(nextMessages, targetConversationId, requestSelectedDocIds, settings, requestWorkspaceId);
      } finally {
        progressUpdater.cancel();
      }
    } catch (error) {
      if (!isCurrentSession()) return;
      if (isFetchInterrupted(error)) {
        const interruptedMessages = markPendingAnswerInterrupted(baseMessages, pending.id, latestRagProgress);
        if (isCurrentWorkspace()) {
          setMessages(interruptedMessages);
          setTypingMessageId(null);
        }
        try {
          await persistConversation(
            interruptedMessages,
            targetConversationId,
            requestSelectedDocIds,
            settings,
            requestWorkspaceId,
          );
        } catch {
          // The initial sending state was already persisted before streaming began.
        }
        return;
      }
      const failedMessages: ChatMessage[] = baseMessages.map((item) =>
          item.id === pending.id
            ? {
                ...item,
                content: error instanceof Error ? error.message : "回答失败",
                status: "failed" as const,
                ragProgress: latestRagProgress,
              }
              : item,
      );
      if (isCurrentWorkspace()) {
        setMessages(failedMessages);
        setTypingMessageId(null);
      }
      await persistConversation(failedMessages, targetConversationId, requestSelectedDocIds, settings, requestWorkspaceId);
    } finally {
      if (isCurrentSession()) {
        setBusy(false);
      }
    }
  }

  async function resumePendingAnswer(conversation: Conversation, nextSettings: Settings, workspaceId = activeWorkspaceId) {
    const requestToken = nextSettings.token;
    const isCurrentSession = () => Boolean(requestToken) && authTokenRef.current === requestToken;
    const isCurrentWorkspace = () => isCurrentSession() && activeWorkspaceIdRef.current === workspaceId;
    const pendingIndex = conversation.messages.findIndex(
      (message) => message.role === "assistant" && message.status === "sending",
    );
    if (pendingIndex < 0) return;
    const userIndex = findPreviousUserMessageIndex(conversation.messages, pendingIndex);
    if (userIndex < 0) return;
    const pending = conversation.messages[pendingIndex];
    const userMessage = conversation.messages[userIndex];
    const recoveryRequestId = pending.requestId || crypto.randomUUID();
    const recoveryMessages = conversation.messages.map((message, index) =>
      index === pendingIndex ? { ...message, requestId: recoveryRequestId } : message,
    );
    setBusy(true);
    setTypingMessageId(null);
    let latestRagProgress = initialRagProgress(conversation.source_doc_ids.length > 0 || Boolean(userMessage.imageDataUrl)) || [];
    try {
      if (isCurrentWorkspace()) {
        setMessages((current) =>
          current.map((item, index) =>
            index === pendingIndex
              ? {
                  ...item,
                  content: conversation.source_doc_ids.length > 0 ? "RAG 调用链启动中..." : "大模型正在思考...",
                  ragProgress: latestRagProgress,
                }
              : item,
          ),
        );
      }
      const progressUpdater = createRagProgressFrameUpdater(
        (_item, index) => index === pendingIndex,
        isCurrentWorkspace,
      );
      const updateRagProgress = (stage: RagProgressStage) => {
        latestRagProgress = mergeRagProgress(latestRagProgress, stage);
        progressUpdater.schedule(stage, latestRagProgress);
      };
      try {
        const response = await queryRagStream(nextSettings, {
          query: userMessage.content,
          requestId: recoveryRequestId,
          docIds: conversation.source_doc_ids,
          history: recoveryMessages
            .slice(0, userIndex)
            .map((message) => `${message.role}: ${message.content}`)
            .slice(-8),
          imageDataUrl: userMessage.imageDataUrl,
          onEvent: (event) => {
            if (event.type === "stage") {
              updateRagProgress(event);
            }
          },
        });
        progressUpdater.cancel();
        if (!isCurrentSession()) return;
        const nextMessages: ChatMessage[] = recoveryMessages.map((item, index) =>
          index === pendingIndex
            ? {
                ...item,
                content: response.answer,
                requestId: response.request_id,
                citations: response.citations,
                status: "done" as const,
                ragProgress: latestRagProgress,
              }
            : item,
        );
        if (isCurrentWorkspace()) {
          setMessages(nextMessages);
          setTypingMessageId(pending.id);
        }
        await persistConversation(nextMessages, conversation.id, conversation.source_doc_ids, nextSettings, workspaceId);
      } finally {
        progressUpdater.cancel();
      }
    } catch (error) {
      if (!isCurrentSession()) return;
      if (isFetchInterrupted(error)) {
        const interruptedMessages = markPendingAnswerInterrupted(
          recoveryMessages,
          pending.id,
          latestRagProgress,
        );
        if (isCurrentWorkspace()) {
          setMessages(interruptedMessages);
          setTypingMessageId(null);
        }
        try {
          await persistConversation(
            interruptedMessages,
            conversation.id,
            conversation.source_doc_ids,
            nextSettings,
            workspaceId,
          );
        } catch {
          // Keep the server's existing sending state so a later reload can retry recovery.
        }
        return;
      }
      const failedMessages: ChatMessage[] = recoveryMessages.map((item, index) =>
        index === pendingIndex
          ? {
              ...item,
              content: error instanceof Error ? error.message : "回答恢复失败",
              status: "failed" as const,
              ragProgress: latestRagProgress,
            }
          : item,
      );
      if (isCurrentWorkspace()) {
        setMessages(failedMessages);
        setTypingMessageId(null);
      }
      await persistConversation(failedMessages, conversation.id, conversation.source_doc_ids, nextSettings, workspaceId);
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
    workspaceId = activeWorkspaceId,
  ) {
    if (nextMessages.length === 0) return null;
    const title =
      targetConversationId && targetConversationId === conversationId
        ? conversationTitle
        : inferConversationTitle(nextMessages);
    const saved = await saveConversation(targetSettings, {
      id: targetConversationId,
      title,
      messages: nextMessages,
      sourceDocIds,
    });
    if (activeWorkspaceIdRef.current === workspaceId) {
      setConversationId(saved.id);
      setConversationTitle(saved.title);
      setConversations((current) => [
        {
          id: saved.id,
          tenant_id: saved.tenant_id,
          title: saved.title,
          message_count: saved.messages.length,
          source_doc_ids: saved.source_doc_ids,
          created_at: saved.created_at,
          updated_at: saved.updated_at,
        },
        ...current.filter((item) => item.id !== saved.id),
      ]);
    }
    addConversationToWorkspace(workspaceId, saved.id);
    return saved;
  }

  async function handleOpenConversation(targetConversationId: string) {
    const requestWorkspaceId = activeWorkspaceId;
    const loaded = await getConversation(settings, targetConversationId);
    if (activeWorkspaceIdRef.current !== requestWorkspaceId) return;
    suppressAutoConversationLoadRef.current = false;
    setConversationId(loaded.id);
    setConversationTitle(loaded.title);
    setTypingMessageId(null);
    const loadedMessages = loaded.messages.map(normalizeMessage);
    setMessages(loadedMessages);
    const selectedIds = new Set(loaded.source_doc_ids);
    setSources((current) =>
      current.map((source) => ({
        ...source,
        selected:
          source.status === "ready" &&
          (selectedIds.has(source.doc_id) || Boolean(source.child_doc_ids?.some((docId) => selectedIds.has(docId)))),
      })),
    );
    if (hasPendingAssistant(loadedMessages)) {
      void resumePendingAnswer({ ...loaded, messages: loadedMessages }, settings, requestWorkspaceId);
    }
  }

  function handleNewConversation() {
    suppressAutoConversationLoadRef.current = true;
    setConversationId(null);
    setConversationTitle("未命名对话");
    setMessages([]);
    setTypingMessageId(null);
  }

  async function handleRenameConversation(targetConversationId: string, title: string) {
    const renamed = await renameConversation(settings, targetConversationId, title);
    setConversations((current) => {
      const existing = current.find((item) => item.id === targetConversationId);
      if (!existing) return current;
      const next = {
        ...existing,
        title: renamed.title,
        updated_at: renamed.updated_at,
      };
      return [next, ...current.filter((item) => item.id !== targetConversationId)];
    });
    if (conversationId === targetConversationId) {
      setConversationTitle(renamed.title);
    }
  }

  async function handleDeleteConversation(targetConversationId: string) {
    await deleteConversation(settings, targetConversationId);
    removeConversationFromWorkspace(activeWorkspaceId, targetConversationId);
    setConversations((current) => current.filter((item) => item.id !== targetConversationId));
    if (conversationId === targetConversationId) {
      suppressAutoConversationLoadRef.current = true;
      setConversationId(null);
      setConversationTitle("未命名对话");
      setMessages([]);
      setTypingMessageId(null);
    }
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

  async function handleCreateArtifact(kind: ArtifactKind) {
    if (selectedDocIds.length === 0 || !beginArtifactGeneration()) return;
    const requestWorkspaceId = activeWorkspaceId;
    const isCurrentWorkspace = () => activeWorkspaceIdRef.current === requestWorkspaceId;
    const sourceDocIds = [...selectedDocIds];
    const titleSuffix = kind === "table" ? "数据表格" : "思维导图";
    const title = selectedSources.length === 1 ? `${selectedSources[0].title} ${titleSuffix}` : `选中来源${titleSuffix}`;
    const pendingIdPrefix = kind === "table" ? "pending-table" : "pending-mindmap";
    const pendingArtifact: MindMapArtifact = {
      id: `${pendingIdPrefix}-${Date.now()}`,
      title,
      status: "generating",
      tenant_id: settings.tenantId,
      source_doc_ids: sourceDocIds,
      created_at: Date.now(),
      updated_at: Date.now(),
      artifact_type: kind,
      workspace_id: requestWorkspaceId,
      root: kind === "mindmap" ? null : undefined,
      table: kind === "table" ? null : undefined,
    };
    setArtifacts((items) => [pendingArtifact, ...items]);
    let trackedArtifactId = pendingArtifact.id;
    try {
      const artifact =
        kind === "table"
          ? await createDataTable(settings, title, sourceDocIds, requestWorkspaceId)
          : await createMindMap(settings, title, sourceDocIds, requestWorkspaceId);
      trackedArtifactId = artifact.id;
      addArtifactToWorkspace(requestWorkspaceId, artifact.id);
      if (isCurrentWorkspace()) {
        setArtifacts((items) => upsertArtifact(items, artifact, pendingArtifact.id));
      }
      const readyArtifact = await waitForArtifact(settings, artifact.id, requestWorkspaceId, (updatedArtifact) => {
        if (!isCurrentWorkspace()) return;
        setArtifacts((items) => upsertArtifact(items, updatedArtifact));
        setActiveArtifact((current) => (current?.id === updatedArtifact.id ? updatedArtifact : current));
      });
      if (isCurrentWorkspace()) {
        setArtifacts((items) => upsertArtifact(items, readyArtifact));
        openArtifact(readyArtifact);
      }
    } catch (error) {
      if (isCurrentWorkspace()) {
        if (error instanceof ArtifactPendingError) {
          return;
        }
        const failedMessage = error instanceof Error ? error.message : "生成失败";
        setArtifacts((items) =>
          items.map((item) =>
            item.id === pendingArtifact.id || item.id === trackedArtifactId
              ? { ...item, status: "failed", error: failedMessage }
            : item,
          ),
        );
        setActiveArtifact((current) =>
          current?.id === trackedArtifactId ? { ...current, status: "failed", error: failedMessage } : current,
        );
      }
    } finally {
      finishArtifactGeneration();
    }
  }

  async function handleCreateMindMap() {
    await handleCreateArtifact("mindmap");
  }

  async function handleCreateDataTable() {
    await handleCreateArtifact("table");
  }

  async function handleRenameArtifact(artifact: MindMapArtifact, newTitle: string) {
    if (!newTitle.trim() || newTitle === artifact.title) return;
    try {
      await renameArtifact(settings, artifact.id, newTitle, activeWorkspaceId);
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
      if (sourceReferencedByOtherWorkspace(activeWorkspaceId, sourceIdsForWorkspace([source]), workspaces)) {
        saveWorkspaceSourceTitle(activeWorkspaceId, source.doc_id, newTitle);
        if (activeSourceContent?.doc_id === source.doc_id && activeSourceContent.doc_version === source.doc_version) {
          setActiveSourceContent({ ...activeSourceContent, title: newTitle });
        }
        return;
      }
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
      await deleteArtifact(settings, artifact.id, activeWorkspaceId);
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
    const nextActive = loadActiveWorkspaceId(anonWorkspaces, null);
    saveWorkspaces(anonWorkspaces, null);
    activeWorkspaceIdRef.current = nextActive;
    setWorkspaces(anonWorkspaces);
    setActiveWorkspaceId(nextActive);
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
    initializeEmptyWorkspaceData(nextWorkspace.id);
    activeWorkspaceIdRef.current = nextWorkspace.id;
    setWorkspaces((items) => [nextWorkspace, ...items]);
    setActiveWorkspaceId(nextWorkspace.id);
    setSources([]);
    setMessages([]);
    setConversations([]);
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

  function handleRenameWorkspaceById(workspaceId: string, name: string) {
    if (!name.trim() || !isAuthenticated) return;
    const nextName = name.trim();
    setWorkspaces((items) =>
      items.map((workspace) =>
        workspace.id === workspaceId ? { ...workspace, name: nextName, updated_at: Date.now() } : workspace,
      ),
    );
  }

  function handleSelectWorkspace(id: string) {
    if (id === activeWorkspaceId || !workspaces.some((workspace) => workspace.id === id)) return;
    switchWorkspace(id);
  }

  function switchWorkspace(id: string) {
    activeWorkspaceIdRef.current = id;
    setActiveWorkspaceId(id);
    setSources([]);
    setMessages([]);
    setConversations([]);
    setConversationId(null);
    setConversationTitle("未命名对话");
    setArtifacts([]);
    setActiveArtifact(null);
    setActiveSourceContent(null);
    suppressAutoConversationLoadRef.current = false;
    void refresh(settings, id);
  }

  async function handleDeleteWorkspace(id: string) {
    setBusy(true);
    setStatus("正在删除知识库数据...");
    try {
      await deleteWorkspaceRemoteData(id, settings, workspaces);
      const { workspaces: remaining, nextActive } = deleteWorkspace(id, auth.user?.id ?? null);
      setWorkspaces(remaining);
      switchWorkspace(nextActive);
    } catch (error) {
      setStatus(error instanceof Error ? `删除知识库失败：${error.message}` : "删除知识库失败");
    } finally {
      setBusy(false);
    }
  }

  function closeAnnouncement() {
    const announcement = announcements[0];
    if (announcement) {
      const audience = auth.user?.id ?? "guest";
      const key = announcementDismissKey(audience, announcement.id);
      sessionStorage.setItem(`announcement-dismissed:${key}`, "1");
    }
    setAnnouncementOpen(false);
  }

  return (
    <div className={`workspace-shell ${chromeCollapsed ? "chrome-collapsed" : ""}`}>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">
            <DatabaseZap size={22} />
          </span>
          <span>{displayWorkspaceName}</span>
        </div>
        <div className="topbar-actions">
          <a
            className="icon-button"
            href="https://github.com/lizhenisu/project4llm"
            target="_blank"
            rel="noreferrer"
            aria-label="打开 GitHub 仓库"
            title="GitHub"
          >
            <Github size={18} />
          </a>
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
                <img src={auth.user.avatar_url} alt="" decoding="async" />
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
            onRetrySource={handleRetrySource}
            onRenameSource={handleRenameSource}
            onOpenSource={handleOpenSource}
            activeContent={activeSourceContent}
            contentLoading={sourceContentLoading}
            contentError={sourceContentError}
            assetToken={settings.token}
            assetApiBaseUrl={settings.apiBaseUrl}
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
            conversations={conversations}
            activeConversationId={conversationId}
            chromeCollapsed={chromeCollapsed}
            selectedSources={selectedSources}
            authenticated={isAuthenticated}
            busy={busy}
            conversationTitle={conversationTitle}
            typingMessageId={typingMessageId}
            assetToken={settings.token}
            assetApiBaseUrl={settings.apiBaseUrl}
            onTypingComplete={() => setTypingMessageId(null)}
            onAsk={handleAsk}
            onFeedback={handleFeedback}
            onNewConversation={handleNewConversation}
            onOpenConversation={handleOpenConversation}
            onRenameConversation={handleRenameConversation}
            onDeleteConversation={handleDeleteConversation}
            onToggleChrome={() => setChromeCollapsed((collapsed) => !collapsed)}
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
      <footer className="statusbar">
        <span>{status}</span>
        <span className="version-badge" onClick={() => onNavigate("/architecture")} title="查看系统架构">v{APP_VERSION}</span>
      </footer>
      <SettingsDialog
        open={settingsOpen}
        workspaces={workspaces}
        activeWorkspaceId={activeWorkspaceId}
        authenticated={isAuthenticated}
        onClose={() => setSettingsOpen(false)}
        onNewWorkspace={handleNewWorkspace}
        onRenameWorkspace={handleRenameWorkspaceById}
        onDeleteWorkspace={handleDeleteWorkspace}
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
    <img className="account-avatar-image" src={user.avatar_url} alt="" decoding="async" />
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
        {announcement.link_url ? (
          <a className="announcement-link" href={announcement.link_url} target="_blank" rel="noreferrer">
            <ExternalLink size={16} />
            {announcement.link_label || "查看详情"}
          </a>
        ) : null}
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
  const [loginLinkMenuOpen, setLoginLinkMenuOpen] = useState(false);
  const [refreshingLoginToken, setRefreshingLoginToken] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [savingPassword, setSavingPassword] = useState(false);
  const loginLink = useMemo(() => buildLoginLink(auth.token), [auth.token]);
  const canEditProfileName = user.profile_name_edit_allowed !== false;
  const canEditAvatar = user.avatar_edit_allowed !== false;
  const canRefreshLoginToken = user.username !== "test_user";

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

  async function refreshExclusiveLoginToken() {
    if (!canRefreshLoginToken || refreshingLoginToken) return;
    setError("");
    setMessage("");
    setRefreshingLoginToken(true);
    try {
      const response = await refreshLoginToken(settings);
      auth.replaceSession(response);
      setLoginLinkVisible(true);
      setLoginLinkMenuOpen(false);
      setMessage("专属登录链接已刷新");
    } catch (err) {
      setError(err instanceof Error ? err.message : "刷新 token 失败");
    } finally {
      setRefreshingLoginToken(false);
    }
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
              {avatarUrl ? <img src={avatarUrl} alt="" decoding="async" /> : <span>{displayName.slice(0, 1).toUpperCase() || "U"}</span>}
            </div>
            <div>
              <strong>{displayName || user.display_name}</strong>
              <span>{user.role === "admin" ? "管理员" : "普通用户"} · {user.status === "banned" ? "已封禁" : "正常"}</span>
            </div>
          </div>
          <label>
            用户名
            <input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" disabled={!canEditProfileName} />
          </label>
          <label>
            显示名称
            <input value={displayName} onChange={(event) => setDisplayName(event.target.value)} autoComplete="name" disabled={!canEditProfileName} />
          </label>
          <label>
            头像地址
            <input value={avatarUrl} onChange={(event) => setAvatarUrl(event.target.value)} placeholder="https://..." disabled={!canEditAvatar} />
          </label>
          {!canEditProfileName || !canEditAvatar ? (
            <p className="muted-text">部分资料修改权限已由管理员关闭。</p>
          ) : null}
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
          <p className="muted-text">通过专属登录链接可以实现无密码账户登录，请勿将该链接分享给别人。</p>
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
                {loginLinkCopied ? <Check size={16} style={{ color: "var(--green)" }} /> : <Copy size={16} />}
              </button>
              <div className="login-link-more">
                <button
                  type="button"
                  className="icon-button"
                  aria-label="更多专属登录链接选项"
                  aria-expanded={loginLinkMenuOpen}
                  onClick={() => setLoginLinkMenuOpen((open) => !open)}
                >
                  <MoreHorizontal size={16} />
                </button>
                {loginLinkMenuOpen ? (
                  <>
                    <button
                      type="button"
                      className="login-link-menu-backdrop"
                      aria-label="关闭专属登录链接选项"
                      onClick={() => setLoginLinkMenuOpen(false)}
                    />
                    <div className="login-link-menu" role="menu">
                      <button
                        type="button"
                        role="menuitem"
                        disabled={!canRefreshLoginToken || refreshingLoginToken}
                        title={canRefreshLoginToken ? "刷新专属登录链接 token" : "测试账号使用固定 token，不能刷新"}
                        onClick={refreshExclusiveLoginToken}
                      >
                        <RefreshCw size={15} />
                        <span>{refreshingLoginToken ? "刷新中..." : "刷新 token"}</span>
                      </button>
                    </div>
                  </>
                ) : null}
              </div>
            </div>
          </label>
          {!canRefreshLoginToken ? (
            <p className="muted-text">测试账号使用固定专属 token，不能刷新。</p>
          ) : null}
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
  const [activeSection, setActiveSection] = useState<AdminSection>("settings");
  const [navCollapsed, setNavCollapsed] = useState(false);
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [linkUrl, setLinkUrl] = useState("");
  const [linkLabel, setLinkLabel] = useState("");
  const [error, setError] = useState("");
  const [adminSettings, setAdminSettings] = useState<AdminSettings | null>(null);
  const [settingsBusy, setSettingsBusy] = useState(false);
  const [userTotal, setUserTotal] = useState<number | null>(null);

  useEffect(() => {
    Promise.all([getAdminSettings(settings), listAdminUsers(settings, { limit: 1 })])
      .then(([nextSettings, userPage]) => {
        setAdminSettings(nextSettings);
        setUserTotal(userPage.total);
      })
      .catch((err) => setError(err instanceof Error ? err.message : "加载管理员控制台失败"));
  }, [settings]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    try {
      const announcement = await publishAnnouncement(settings, { title, content, linkUrl, linkLabel });
      onAnnouncement(announcement);
      setAdminSettings((current) =>
        current
          ? { ...current, latest_announcement: announcement }
          : { registration_enabled: true, latest_announcement: announcement },
      );
      setTitle("");
      setContent("");
      setLinkUrl("");
      setLinkLabel("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "发布公告失败");
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

  async function removeLatestAnnouncement() {
    const announcement = adminSettings?.latest_announcement;
    if (!announcement) return;
    setError("");
    setSettingsBusy(true);
    try {
      await deleteAnnouncement(settings, announcement.id);
      setAdminSettings((current) => current ? { ...current, latest_announcement: null } : current);
    } catch (err) {
      setError(err instanceof Error ? err.message : "删除公告失败");
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
          <p>发布公告、调整系统开关，用户管理从独立入口进入。</p>
        </div>
      </div>
      <section className={`admin-console-layout ${navCollapsed ? "is-nav-collapsed" : ""}`}>
        <aside className="admin-console-nav" aria-label="管理员功能导航">
          <button
            type="button"
            className="admin-nav-collapse"
            aria-label={navCollapsed ? "展开功能区" : "收起功能区"}
            title={navCollapsed ? "展开功能区" : "收起功能区"}
            onClick={() => setNavCollapsed((collapsed) => !collapsed)}
          >
            {navCollapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
            <span>
              <strong>功能区</strong>
              <small>{navCollapsed ? "已收起" : "可收起"}</small>
            </span>
          </button>
          <button
            type="button"
            className={activeSection === "settings" ? "is-active" : ""}
            title="系统设置"
            onClick={() => setActiveSection("settings")}
          >
            <SettingsIcon size={17} />
            <span>
              <strong>系统设置</strong>
              <small>{adminSettings?.registration_enabled ? "注册开放" : "注册关闭"}</small>
            </span>
          </button>
          <button
            type="button"
            className={activeSection === "announcements" ? "is-active" : ""}
            title="公告管理"
            onClick={() => setActiveSection("announcements")}
          >
            <Megaphone size={17} />
            <span>
              <strong>公告管理</strong>
              <small>{adminSettings?.latest_announcement ? "有当前公告" : "暂无公告"}</small>
            </span>
          </button>
          <button
            type="button"
            className={activeSection === "users" ? "is-active" : ""}
            title="用户管理"
            onClick={() => setActiveSection("users")}
          >
            <Users size={17} />
            <span>
              <strong>用户管理</strong>
              <small>{userTotal === null ? "加载中" : `${userTotal} 个账号`}</small>
            </span>
          </button>
          <button
            type="button"
            className={activeSection === "ingestion" ? "is-active" : ""}
            title="摄取运维"
            onClick={() => setActiveSection("ingestion")}
          >
            <DatabaseZap size={17} />
            <span>
              <strong>摄取运维</strong>
              <small>死信与审计</small>
            </span>
          </button>
        </aside>
        <section className="admin-console-content">
          {activeSection === "settings" ? (
            <section className="admin-detail-panel">
              <div className="admin-section-heading">
                <div>
                  <h2>系统设置</h2>
                  <p>控制新账号注册和进入各项管理功能。</p>
                </div>
              </div>
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
              {error ? <p className="error-text">{error}</p> : null}
            </section>
          ) : null}
          {activeSection === "announcements" ? (
            <section className="admin-detail-panel">
              <div className="admin-section-heading">
                <div>
                  <h2>公告管理</h2>
                  <p>发布公告、附加跳转链接，并删除当前公告。</p>
                </div>
              </div>
              <form className="announcement-form admin-form-plain" onSubmit={submit}>
                <label>
                  公告标题
                  <input value={title} onChange={(event) => setTitle(event.target.value)} />
                </label>
                <label>
                  公告内容
                  <textarea value={content} onChange={(event) => setContent(event.target.value)} rows={4} />
                </label>
                <label>
                  跳转链接
                  <input value={linkUrl} onChange={(event) => setLinkUrl(event.target.value)} placeholder="https://example.com 或 /docs" />
                </label>
                <label>
                  按钮文案
                  <input value={linkLabel} onChange={(event) => setLinkLabel(event.target.value)} placeholder="查看详情" />
                </label>
                <button type="submit" disabled={!title.trim() || !content.trim()}>发布公告</button>
              </form>
              <div className="latest-announcement">
                <span>当前公告</span>
                {adminSettings?.latest_announcement ? (
                  <>
                    <strong>{adminSettings.latest_announcement.title}</strong>
                    <p>{adminSettings.latest_announcement.content}</p>
                    {adminSettings.latest_announcement.link_url ? (
                      <a href={adminSettings.latest_announcement.link_url} target="_blank" rel="noreferrer">
                        <ExternalLink size={14} />
                        {adminSettings.latest_announcement.link_label || "查看详情"}
                      </a>
                    ) : null}
                    <small>
                      {adminSettings.latest_announcement.author_name || "系统公告"} · {formatDateTime(adminSettings.latest_announcement.created_at)}
                    </small>
                    <button type="button" className="inline-action danger" disabled={settingsBusy} onClick={removeLatestAnnouncement}>
                      <Trash2 size={15} />
                      删除公告
                    </button>
                  </>
                ) : (
                  <p>暂无公告</p>
                )}
              </div>
              {error ? <p className="error-text">{error}</p> : null}
            </section>
          ) : null}
          {activeSection === "users" ? (
            <AdminUsersPanel settings={settings} currentUser={currentUser} />
          ) : null}
          {activeSection === "ingestion" ? (
            <AdminIngestionPanel settings={settings} />
          ) : null}
        </section>
      </section>
    </main>
  );
}

function AdminIngestionPanel({ settings }: { settings: Settings }) {
  const pageSize = 20;
  const [tasks, setTasks] = useState<AdminDeadLetterTask[]>([]);
  const [auditEvents, setAuditEvents] = useState<AdminIngestionAuditEvent[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [auditError, setAuditError] = useState("");
  const [notice, setNotice] = useState("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    listAdminDeadLetters(settings, { limit: pageSize, offset })
      .then((deadLetters) => {
        if (cancelled) return;
        setTasks(deadLetters.tasks);
        setTotal(deadLetters.total);
        const maxOffset = deadLetters.total > 0
          ? Math.floor((deadLetters.total - 1) / pageSize) * pageSize
          : 0;
        if (offset > maxOffset) {
          setOffset(maxOffset);
        }
        const visibleKeys = new Set(deadLetters.tasks.map(adminIngestionTaskKey));
        setSelectedKeys((current) => current.filter((key) => visibleKeys.has(key)));
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "加载摄取运维数据失败");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [offset, refreshVersion, settings]);

  useEffect(() => {
    let cancelled = false;
    setAuditError("");
    listAdminIngestionAudit(settings, { limit: 20 })
      .then((audit) => {
        if (!cancelled) setAuditEvents(audit.events);
      })
      .catch((err) => {
        if (!cancelled) {
          setAuditError(err instanceof Error ? err.message : "加载操作审计失败");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [refreshVersion, settings]);

  const selectedTasks = tasks.filter((task) =>
    selectedKeys.includes(adminIngestionTaskKey(task)),
  );
  const allPageSelected = tasks.length > 0 && selectedTasks.length === tasks.length;

  function toggleTask(task: AdminDeadLetterTask) {
    const key = adminIngestionTaskKey(task);
    setSelectedKeys((current) =>
      current.includes(key)
        ? current.filter((item) => item !== key)
        : [...current, key],
    );
  }

  async function redriveSelected() {
    if (!selectedTasks.length) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const response = await redriveAdminDeadLetters(
        settings,
        selectedTasks.map((task) => ({
          tenant_id: task.tenant_id,
          task_id: task.task_id,
        })),
      );
      setNotice(
        response.rejected
          ? `已重新排队 ${response.queued} 个，${response.rejected} 个未处理。`
          : `已重新排队 ${response.queued} 个任务。`,
      );
      setSelectedKeys([]);
      setRefreshVersion((version) => version + 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : "批量重新处理失败");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="admin-ingestion-panel">
      <div className="admin-section-heading">
        <div>
          <h2>摄取运维</h2>
          <p>查看终止重试的摄取任务，批量重新排队，并核对管理员操作记录。</p>
        </div>
        <button
          type="button"
          className="inline-action"
          disabled={loading || busy}
          onClick={() => setRefreshVersion((version) => version + 1)}
        >
          <RefreshCw size={15} />
          刷新
        </button>
      </div>

      <div className="admin-ingestion-toolbar">
        <label>
          <input
            type="checkbox"
            checked={allPageSelected}
            disabled={!tasks.length || busy}
            aria-label="选择当前页全部死信任务"
            onChange={() =>
              setSelectedKeys(
                allPageSelected ? [] : tasks.map(adminIngestionTaskKey),
              )
            }
          />
          当前页 {tasks.length} 个，共 {total} 个死信任务
        </label>
        <button
          type="button"
          className="inline-action"
          disabled={!selectedTasks.length || busy}
          onClick={redriveSelected}
        >
          <RefreshCw size={15} />
          {busy ? "正在重新排队…" : `重新处理所选（${selectedTasks.length}）`}
        </button>
      </div>

      {loading ? <p className="muted-text">正在加载死信任务…</p> : null}
      {!loading && !tasks.length ? (
        <div className="admin-ingestion-empty">
          <CheckCircle2 size={20} />
          <span>当前没有死信任务</span>
        </div>
      ) : null}
      <div className="admin-ingestion-task-list" aria-label="摄取死信任务">
        {tasks.map((task) => {
          const key = adminIngestionTaskKey(task);
          return (
            <article className="admin-ingestion-task" key={key}>
              <label className="admin-ingestion-task-select">
                <input
                  type="checkbox"
                  checked={selectedKeys.includes(key)}
                  disabled={busy}
                  aria-label={`选择死信任务 ${task.title}`}
                  onChange={() => toggleTask(task)}
                />
              </label>
              <div className="admin-ingestion-task-main">
                <strong>{task.title}</strong>
                <span>{task.tenant_id} · {task.source_type.toUpperCase()}</span>
                <small>{task.error || "未记录错误详情"}</small>
              </div>
              <div className="admin-ingestion-task-meta">
                <span>尝试 {task.attempt_count} 次</span>
                <span>{formatDateTime(task.dead_lettered_at)}</span>
              </div>
            </article>
          );
        })}
      </div>

      <div className="admin-pagination">
        <button
          type="button"
          className="inline-action"
          disabled={offset === 0 || loading}
          onClick={() => setOffset((value) => Math.max(0, value - pageSize))}
        >
          上一页
        </button>
        <button
          type="button"
          className="inline-action"
          disabled={offset + pageSize >= total || loading}
          onClick={() => setOffset((value) => value + pageSize)}
        >
          下一页
        </button>
      </div>

      <section className="admin-ingestion-audit">
        <div>
          <h3>最近操作审计</h3>
          <span>保留最近的批量重新处理结果</span>
        </div>
        {!auditEvents.length ? <p className="muted-text">暂无操作记录</p> : null}
        <div className="admin-ingestion-audit-list">
          {auditEvents.map((event) => (
            <article key={event.id}>
              <strong>{adminIngestionOutcomeLabel(event.outcome)}</strong>
              <span>{event.tenant_id} · {event.task_id}</span>
              <small>{formatDateTime(event.created_at)} · 操作人 {event.actor_user_id}</small>
            </article>
          ))}
        </div>
        {auditError ? <p className="error-text" role="alert">{auditError}</p> : null}
      </section>
      {notice ? <p className="success-text" role="status">{notice}</p> : null}
      {error ? <p className="error-text" role="alert">{error}</p> : null}
    </section>
  );
}

function AdminUsersPanel({
  settings,
  currentUser,
}: {
  settings: Settings;
  currentUser: AuthUser;
}) {
  const pageSize = 50;
  const [users, setUsers] = useState<AuthUser[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [query, setQuery] = useState("");
  const [appliedQuery, setAppliedQuery] = useState("");
  const [selectedUserIds, setSelectedUserIds] = useState<string[]>([]);
  const [userDrafts, setUserDrafts] = useState<Record<string, AdminUserDraft>>({});
  const [busyUserId, setBusyUserId] = useState<string | null>(null);
  const [settingsBusy, setSettingsBusy] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    listAdminUsers(settings, { query: appliedQuery, limit: pageSize, offset })
      .then((page) => {
        if (cancelled) return;
        setUsers(page.users);
        setTotal(page.total);
        setUserDrafts(userRowsToDrafts(page.users));
        setSelectedUserIds((ids) => ids.filter((id) => page.users.some((user) => user.id === id)));
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : "加载用户失败");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [settings, appliedQuery, offset]);

  function submitSearch(event: FormEvent) {
    event.preventDefault();
    setOffset(0);
    setAppliedQuery(query.trim());
    setSelectedUserIds([]);
  }

  function updateUserDraft(userId: string, patch: Partial<AdminUserDraft>) {
    setUserDrafts((drafts) => ({
      ...drafts,
      [userId]: { ...(drafts[userId] || emptyUserDraft()), ...patch },
    }));
  }

  async function saveUserPermissions(user: AuthUser) {
    const draft = userDrafts[user.id] || userToDraft(user);
    setError("");
    setBusyUserId(user.id);
    try {
      const response = await updateAdminUsers(settings, [
        {
          user_id: user.id,
          profile_name_edit_allowed: draft.profileNameEditAllowed,
          avatar_edit_allowed: draft.avatarEditAllowed,
        },
      ]);
      const updated = response.users[0];
      setUsers((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      setUserDrafts((drafts) => ({ ...drafts, [updated.id]: userToDraft(updated) }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "保存用户权限失败");
    } finally {
      setBusyUserId(null);
    }
  }

  async function toggleUserStatus(user: AuthUser) {
    setError("");
    setBusyUserId(user.id);
    try {
      const nextStatus = user.status === "banned" ? "active" : "banned";
      const updated = await updateAdminUserStatus(settings, user.id, nextStatus);
      setUsers((items) => items.map((item) => (item.id === updated.id ? updated : item)));
      setUserDrafts((drafts) => ({ ...drafts, [updated.id]: userToDraft(updated) }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "更新用户状态失败");
    } finally {
      setBusyUserId(null);
    }
  }

  async function bulkUpdateSelectedUsers(status: "active" | "banned") {
    const ids = selectableUserIds(users, currentUser).filter((id) => selectedUserIds.includes(id));
    if (ids.length === 0) return;
    setError("");
    setSettingsBusy(true);
    try {
      const response = await updateAdminUsers(
        settings,
        ids.map((userId) => ({ user_id: userId, status })),
      );
      const updatedById = new Map(response.users.map((user) => [user.id, user]));
      setUsers((items) => items.map((item) => updatedById.get(item.id) || item));
      setUserDrafts((drafts) => ({ ...drafts, ...userRowsToDrafts(response.users) }));
      setSelectedUserIds([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "批量更新用户失败");
    } finally {
      setSettingsBusy(false);
    }
  }

  function toggleSelectedUser(userId: string) {
    setSelectedUserIds((ids) => (ids.includes(userId) ? ids.filter((id) => id !== userId) : [...ids, userId]));
  }

  function toggleAllSelectableUsers() {
    const ids = selectableUserIds(users, currentUser);
    setSelectedUserIds((current) => (current.length === ids.length ? [] : ids));
  }

  const selectableIds = selectableUserIds(users, currentUser);
  const allSelected = selectableIds.length > 0 && selectedUserIds.length === selectableIds.length;
  const pageStart = total === 0 ? 0 : offset + 1;
  const pageEnd = Math.min(offset + pageSize, total);

  return (
    <section className="admin-users-panel">
      <div className="admin-section-heading">
        <div>
          <h2>用户管理</h2>
          <p>按页搜索用户，批量封禁账号，并控制用户是否能自行修改名称和头像。</p>
        </div>
      </div>
        <form className="admin-user-search" onSubmit={submitSearch}>
          <div className="search-field">
            <Search size={16} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="搜索用户名、显示名称或 Tenant"
            />
          </div>
          <button className="primary-pill" type="submit">搜索</button>
        </form>
        <div className="admin-bulk-actions">
          <label>
            <input type="checkbox" checked={allSelected} onChange={toggleAllSelectableUsers} disabled={selectableIds.length === 0} />
            选择当前页普通用户
          </label>
          <div>
            <button type="button" className="inline-action danger" disabled={selectedUserIds.length === 0 || settingsBusy} onClick={() => bulkUpdateSelectedUsers("banned")}>
              <Ban size={15} />
              批量封禁
            </button>
            <button type="button" className="inline-action" disabled={selectedUserIds.length === 0 || settingsBusy} onClick={() => bulkUpdateSelectedUsers("active")}>
              <CheckCircle2 size={15} />
              批量解封
            </button>
          </div>
        </div>
        <div className="admin-user-page-meta">
          <span>{loading ? "加载中..." : `显示 ${pageStart}-${pageEnd} / ${total}`}</span>
          {appliedQuery ? <span>搜索：{appliedQuery}</span> : null}
        </div>
        {error ? <p className="error-text">{error}</p> : null}
        <div className="admin-user-list">
          {users.map((user) => {
            const draft = userDrafts[user.id] || userToDraft(user);
            const editable = user.role !== "admin" && user.id !== currentUser.id;
            return (
              <article className="admin-user-row" key={user.id}>
                <label className="admin-user-select">
                  <input
                    type="checkbox"
                    checked={selectedUserIds.includes(user.id)}
                    disabled={!editable}
                    onChange={() => toggleSelectedUser(user.id)}
                  />
                </label>
                <div className="admin-user-main">
                  <div className="admin-user-avatar">
                    {user.avatar_url ? (
                      <img src={user.avatar_url} alt="" loading="lazy" decoding="async" />
                    ) : (
                      <span>{user.display_name.slice(0, 1).toUpperCase()}</span>
                    )}
                  </div>
                  <div>
                    <strong>{user.display_name}</strong>
                    <span>{user.username} · {user.tenant_id}</span>
                    <small>
                      {user.role === "admin" ? "管理员" : "普通用户"} · {user.status === "banned" ? "已封禁" : "正常"}
                    </small>
                  </div>
                </div>
                <div className="admin-user-dates">
                  <span>注册 {formatDateTime(user.created_at)}</span>
                  <span>最近登录 {user.last_login_at ? formatDateTime(user.last_login_at) : "未登录"}</span>
                </div>
                <div className="admin-user-permissions">
                  <label>
                    <input
                      type="checkbox"
                      checked={draft.profileNameEditAllowed}
                      disabled={!editable || busyUserId === user.id}
                      onChange={(event) => updateUserDraft(user.id, { profileNameEditAllowed: event.target.checked })}
                    />
                    允许改名称
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={draft.avatarEditAllowed}
                      disabled={!editable || busyUserId === user.id}
                      onChange={(event) => updateUserDraft(user.id, { avatarEditAllowed: event.target.checked })}
                    />
                    允许改头像
                  </label>
                </div>
                <div className="admin-row-actions">
                  {editable ? (
                    <>
                      <button
                        className="inline-action"
                        type="button"
                        disabled={busyUserId === user.id}
                        onClick={() => saveUserPermissions(user)}
                      >
                        保存权限
                      </button>
                      <button
                        className={user.status === "banned" ? "inline-action" : "inline-action danger"}
                        type="button"
                        disabled={busyUserId === user.id}
                        onClick={() => toggleUserStatus(user)}
                      >
                        {user.status === "banned" ? <CheckCircle2 size={15} /> : <Ban size={15} />}
                        {user.status === "banned" ? "解封" : "封禁"}
                      </button>
                    </>
                  ) : (
                    <span className="muted-text">不可操作</span>
                  )}
                </div>
              </article>
            );
          })}
          {!loading && users.length === 0 ? <p className="muted-text">没有找到用户。</p> : null}
        </div>
        <div className="admin-pagination">
          <button className="inline-action" type="button" disabled={offset === 0 || loading} onClick={() => setOffset(Math.max(0, offset - pageSize))}>
            上一页
          </button>
          <button className="inline-action" type="button" disabled={offset + pageSize >= total || loading} onClick={() => setOffset(offset + pageSize)}>
            下一页
          </button>
        </div>
    </section>
  );
}

function adminIngestionTaskKey(task: Pick<AdminDeadLetterTask, "tenant_id" | "task_id">) {
  return `${task.tenant_id}\u0000${task.task_id}`;
}

function adminIngestionOutcomeLabel(outcome: string) {
  const labels: Record<string, string> = {
    queued: "已重新排队",
    not_found: "任务不存在",
    not_retryable: "状态不可重试",
    admission_rejected_global: "全局队列已满",
    admission_rejected_tenant: "租户队列已满",
    admission_unavailable: "准入服务不可用",
    reservation_lost: "准入预留已失效",
    retry_unavailable: "重试服务不可用",
  };
  return labels[outcome] || "未知结果";
}

function normalizeMessage(message: ChatMessage): ChatMessage {
  return {
    ...message,
    requestId: message.requestId,
    citations: message.citations || [],
    status: message.status || "done",
  };
}

function emptyUserDraft(): AdminUserDraft {
  return { profileNameEditAllowed: true, avatarEditAllowed: true };
}

function userToDraft(user: AuthUser): AdminUserDraft {
  return {
    profileNameEditAllowed: user.profile_name_edit_allowed !== false,
    avatarEditAllowed: user.avatar_edit_allowed !== false,
  };
}

function userRowsToDrafts(users: AuthUser[]): Record<string, AdminUserDraft> {
  return Object.fromEntries(users.map((user) => [user.id, userToDraft(user)]));
}

function selectableUserIds(users: AuthUser[], currentUser: AuthUser): string[] {
  return users
    .filter((user) => user.role !== "admin" && user.id !== currentUser.id)
    .map((user) => user.id);
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

function markPendingAnswerInterrupted(
  messages: ChatMessage[],
  pendingId: string,
  ragProgress: RagProgressStage[],
): ChatMessage[] {
  return messages.map((message) =>
    message.id === pendingId
      ? {
          ...message,
          content: "连接已中断，刷新页面后将自动恢复回答。",
          status: "sending" as const,
          ragProgress,
        }
      : message,
  );
}

function isFetchInterrupted(error: unknown) {
  if (error instanceof DOMException && error.name === "AbortError") return true;
  return error instanceof TypeError && /(failed to fetch|networkerror|load failed|fetch failed)/i.test(error.message);
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

async function waitForArtifact(
  settings: Settings,
  artifactId: string,
  workspaceId: string,
  onUpdate?: (artifact: MindMapArtifact) => void,
): Promise<MindMapArtifact> {
  const deadline = Date.now() + 180_000;
  while (Date.now() < deadline) {
    await new Promise((resolve) => window.setTimeout(resolve, 1200));
    const artifact = await getArtifact(settings, artifactId, workspaceId);
    onUpdate?.(artifact);
    if (artifact.status === "ready") {
      return artifact;
    }
    if (artifact.status === "failed") {
      throw new Error(artifact.error || "生成失败");
    }
  }
  throw new ArtifactPendingError();
}

class ArtifactPendingError extends Error {
  constructor() {
    super("生成仍在后台进行");
  }
}

function upsertArtifact(items: MindMapArtifact[], artifact: MindMapArtifact, removeId?: string): MindMapArtifact[] {
  return [artifact, ...items.filter((item) => item.id !== artifact.id && item.id !== removeId)];
}

function mergePolledArtifacts(current: MindMapArtifact[], polled: MindMapArtifact[]): MindMapArtifact[] {
  const polledIds = new Set(polled.map((artifact) => artifact.id));
  const localPending = current.filter(
    (artifact) => isLocalPendingArtifact(artifact.id) && !polledIds.has(artifact.id),
  );
  return [...localPending, ...polled];
}

function isLocalPendingArtifact(artifactId: string) {
  return artifactId.startsWith("pending-");
}

async function waitForSourcesReady(
  settings: Settings,
  pendingSources: SourceItem[],
  sourceKeysBeforeUpload: Set<string>,
  onProgress?: (rows: SourceItem[]) => void,
): Promise<SourceItem[]> {
  const pendingIds = new Set(pendingSources.map((source) => source.doc_id));
  const pendingUris = new Set(pendingSources.map((source) => source.source_uri));
  const pendingTitles = new Set(pendingSources.map((source) => source.title));
  const deadline = Date.now() + 180_000;
  let attempts = 0;
  while (Date.now() < deadline) {
    await sleep(sourceReadyPollDelayMs(attempts));
    attempts += 1;
    const rows = await listSourcesCoalesced(settings);
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
    onProgress?.(rows);
  }
  return listSourcesCoalesced(settings);
}

function resolveUploadedSources(
  rows: SourceItem[],
  pendingSources: SourceItem[],
  sourceKeysBeforeUpload: Set<string>,
) {
  const pendingIds = new Set(pendingSources.map((source) => source.doc_id));
  const pendingUris = new Set(pendingSources.map((source) => source.source_uri));
  const pendingTitles = new Set(pendingSources.map((source) => source.title));
  return rows.filter(
    (source) =>
      source.status === "ready" &&
      !pendingIds.has(source.doc_id) &&
      !sourceKeysBeforeUpload.has(sourceStateKey(source)) &&
      (pendingUris.has(source.source_uri) || pendingTitles.has(source.title)),
  );
}

function artifactPollDelayMs(failureCount: number) {
  const base = documentIsHidden() ? ARTIFACT_STATUS_POLL_HIDDEN_MS : ARTIFACT_STATUS_POLL_MS;
  const backedOff = Math.min(ARTIFACT_STATUS_POLL_MAX_MS, base * Math.max(1, failureCount + 1));
  return withJitter(backedOff);
}

function sourceReadyPollDelayMs(attempt: number) {
  const visibilityMultiplier = documentIsHidden() ? 3 : 1;
  const progressMultiplier = attempt < 10 ? 1 : attempt < 30 ? 2 : 4;
  return withJitter(Math.min(SOURCE_READY_POLL_MAX_MS, SOURCE_READY_POLL_BASE_MS * visibilityMultiplier * progressMultiplier));
}

function withJitter(valueMs: number) {
  const jitter = 0.85 + Math.random() * 0.3;
  return Math.round(valueMs * jitter);
}

function sleep(delayMs: number) {
  return new Promise((resolve) => window.setTimeout(resolve, delayMs));
}

function documentIsHidden() {
  return typeof document !== "undefined" && document.hidden;
}

function sourceRowsForList(rows: SourceItem[]) {
  const readyByDocId = new Map<string, SourceItem>();
  for (const source of rows) {
    if (source.status !== "ready") continue;
    const previous = readyByDocId.get(source.doc_id);
    if (!previous || preferSourceRow(source, previous)) {
      readyByDocId.set(source.doc_id, source);
    }
  }
  const readyRows = [...readyByDocId.values()];
  const readyTitles = new Set(readyRows.map((source) => source.title));
  const readyUris = new Set(readyRows.map((source) => source.source_uri));
  const taskRows = rows.filter(
    (source) =>
      source.status !== "ready" &&
      !readyTitles.has(source.title) &&
      !readyUris.has(source.source_uri),
  );
  return [...taskRows, ...readyRows];
}

function preferSourceRow(candidate: SourceItem, current: SourceItem) {
  if (candidate.current !== current.current) {
    return candidate.current;
  }
  if (candidate.doc_version !== current.doc_version) {
    return candidate.doc_version > current.doc_version;
  }
  return (candidate.updated_at || 0) > (current.updated_at || 0);
}

function filterWorkspaceSources(rows: SourceItem[], workspaceSourceIds: string[]) {
  if (workspaceSourceIds.length === 0) {
    return rows.filter((source) => source.status !== "ready");
  }
  const workspaceIds = new Set(workspaceSourceIds);
  return rows.filter((source) => source.status !== "ready" || sourceMatchesWorkspace(source, workspaceIds));
}

function applyWorkspaceSourceTitles(rows: SourceItem[], workspaceId: string) {
  const titles = loadWorkspaceSourceTitles(workspaceId);
  return rows.map((source) => {
    const title = titles[source.doc_id];
    return title ? { ...source, title } : source;
  });
}

async function deleteWorkspaceRemoteData(workspaceId: string, settings: Settings, workspaces: WorkspaceRecord[]) {
  const [sourceRows, conversationRows, artifactRows] = await Promise.all([
    listSourcesCoalesced(settings),
    listConversations(settings),
    listArtifacts(settings, workspaceId),
  ]);
  const workspaceSourceIds = loadWorkspaceSources(workspaceId);
  const sourceRowsToDelete = hasWorkspaceSources(workspaceId)
    ? filterWorkspaceSources(sourceRowsForList(sourceRows), workspaceSourceIds)
    : sourceRowsForList(sourceRows);
  const sourceDocIds = dedupeStrings(
    sourceRowsToDelete
      .filter((source) => !sourceReferencedByOtherWorkspace(workspaceId, sourceIdsForWorkspace([source]), workspaces))
      .map((source) => source.doc_id),
  );
  const conversationIds = hasWorkspaceConversations(workspaceId)
    ? loadWorkspaceConversations(workspaceId)
    : conversationRows.map((conversation) => conversation.id);
  const artifactIds = hasWorkspaceArtifacts(workspaceId)
    ? loadWorkspaceArtifacts(workspaceId)
    : artifactRows.map((artifact) => artifact.id);

  await Promise.all([
    ...sourceDocIds.map((docId) => deleteSource(settings, docId)),
    ...dedupeStrings(conversationIds).map((conversationId) => deleteConversation(settings, conversationId)),
    ...dedupeStrings(artifactIds).map((artifactId) => deleteArtifact(settings, artifactId, workspaceId)),
  ]);
}

function sourceReferencedByOtherWorkspace(
  workspaceId: string,
  sourceIds: string[],
  workspaces: WorkspaceRecord[],
) {
  if (sourceIds.length === 0) return false;
  const ids = new Set(sourceIds);
  return workspaces.some((workspace) => {
    if (workspace.id === workspaceId) return false;
    if (!hasWorkspaceSources(workspace.id)) {
      return true;
    }
    return loadWorkspaceSources(workspace.id).some((sourceId) => ids.has(sourceId));
  });
}

function sourceMatchesWorkspace(source: SourceItem, workspaceIds: Set<string>) {
  return (
    workspaceIds.has(source.doc_id) ||
    Boolean(source.child_doc_ids?.some((docId) => workspaceIds.has(docId))) ||
    Boolean(source.workspace_alias_ids?.some((aliasId) => workspaceIds.has(aliasId)))
  );
}

function sourceIdsForWorkspace(sources: SourceItem[]) {
  return sources.flatMap((source) => [
    source.doc_id,
    ...(source.child_doc_ids || []),
    ...(source.workspace_alias_ids || []),
  ]);
}

function dedupeStrings(values: string[]) {
  return [...new Set(values)];
}

function mergeSelectedState(next: SourceItem[], current: SourceItem[], workspaceId: string) {
  // Respect user's explicit selection from localStorage first.
  const cached = loadCachedSelection(workspaceId);
  if (cached !== null) {
    return next.map((item) => ({
      ...item,
      selected: item.status === "ready" ? (cached.get(sourceStateKey(item)) ?? false) : false,
    }));
  }
  // Fallback: merge from in-memory current state (for runtime updates).
  const selected = new Map(current.map((item) => [sourceStateKey(item), item.selected ?? item.current]));
  return next.map((item) => ({
    ...item,
    selected: item.status === "ready" ? (selected.get(sourceStateKey(item)) ?? item.current) : false,
  }));
}

function preservePendingSourceRows(next: SourceItem[], current: SourceItem[]) {
  const nextKeys = new Set(next.map(sourceStateKey));
  const nextTitles = new Set(next.map((source) => source.title));
  const nextUris = new Set(next.map((source) => source.source_uri));
  const pendingRows = current.filter(
    (source) =>
      source.status !== "ready" &&
      !nextKeys.has(sourceStateKey(source)) &&
      !nextTitles.has(source.title) &&
      !nextUris.has(source.source_uri),
  );
  return [...pendingRows, ...next];
}

function sourceStateKey(source: SourceItem) {
  return `${source.doc_id}::${source.doc_version}`;
}

const SELECTION_CACHE_PREFIX = "source-selection:";

function loadCachedSelection(workspaceId: string): Map<string, boolean> | null {
  try {
    const raw = localStorage.getItem(`${SELECTION_CACHE_PREFIX}${workspaceId}`);
    if (raw === null) return null;
    return new Map(JSON.parse(raw));
  } catch {
    return null;
  }
}

function saveCachedSelection(workspaceId: string, sources: SourceItem[]) {
  const entries: [string, boolean][] = sources.map((s) => [sourceStateKey(s), s.selected ?? false]);
  localStorage.setItem(`${SELECTION_CACHE_PREFIX}${workspaceId}`, JSON.stringify(entries));
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
