import type { Settings, WorkspaceRecord } from "./types";

const KEY = "production-rag-settings";
const WORKSPACE_NAME_KEY = "production-rag-workspace-name";
const WORKSPACES_PREFIX = "production-rag-workspaces";
const WORKSPACE_SOURCES_PREFIX = "production-rag-workspace-sources";
const WORKSPACE_SOURCE_TITLES_PREFIX = "production-rag-workspace-source-titles";
const WORKSPACE_CONVERSATIONS_PREFIX = "production-rag-workspace-conversations";
const WORKSPACE_ARTIFACTS_PREFIX = "production-rag-workspace-artifacts";
const ACTIVE_WORKSPACE_PREFIX = "production-rag-active-workspace-id";
export const DEFAULT_WORKSPACE_NAME = "Production RAG 知识库";
export const EMPTY_WORKSPACE_NAME = "未命名知识库";

export const defaultSettings: Settings = {
  apiBaseUrl: "/api",
  token: "",
  tenantId: "team_a",
  aclGroups: ["engineering"],
};

export function loadSettings(): Settings {
  const raw = localStorage.getItem(KEY);
  if (!raw) return defaultSettings;
  try {
    return { ...defaultSettings, ...JSON.parse(raw) };
  } catch {
    return defaultSettings;
  }
}

export function saveSettings(settings: Settings) {
  localStorage.setItem(KEY, JSON.stringify(settings));
}

export function loadWorkspaceName(userId: string | null = null): string {
  const workspaces = loadUserWorkspaces(userId);
  const activeId = loadActiveWorkspaceId(workspaces, userId);
  return workspaces.find((workspace) => workspace.id === activeId)?.name || workspaces[0]?.name || DEFAULT_WORKSPACE_NAME;
}

export function saveWorkspaceName(name: string) {
  localStorage.setItem(WORKSPACE_NAME_KEY, name);
}

function workspacesKey(userId: string | null): string {
  return userId ? `${WORKSPACES_PREFIX}:${userId}` : WORKSPACES_PREFIX;
}

function activeWorkspaceKey(userId: string | null): string {
  return userId ? `${ACTIVE_WORKSPACE_PREFIX}:${userId}` : ACTIVE_WORKSPACE_PREFIX;
}

export function loadUserWorkspaces(userId: string | null): WorkspaceRecord[] {
  return loadWorkspacesInternal(userId);
}

export function loadWorkspaces(): WorkspaceRecord[] {
  return loadWorkspacesInternal(null);
}

function loadWorkspacesInternal(userId: string | null): WorkspaceRecord[] {
  const key = workspacesKey(userId);
  const raw = localStorage.getItem(key);
  if (raw) {
    try {
      const parsed = JSON.parse(raw) as WorkspaceRecord[];
      const normalized = parsed
        .filter((item) => item?.id && item?.name)
        .map((item) => ({
          id: item.id,
          name: item.name,
          user_id: item.user_id ?? null,
          created_at: item.created_at || Date.now(),
          updated_at: item.updated_at || item.created_at || Date.now(),
      }));
      // Reject workspaces that don't belong to the requested user scope
      const scoped = normalized.filter((w) => (w.user_id ?? null) === userId);
      if (scoped.length > 0) {
        return scoped;
      }
      // If all stored workspaces are foreign, discard them and fall through to defaults
      localStorage.removeItem(key);
    } catch {
      localStorage.removeItem(key);
    }
  }
  return createDefaultWorkspaces(userId);
}

function createDefaultWorkspaces(userId: string | null): WorkspaceRecord[] {
  const now = Date.now();
  return [
    {
      id: "default-workspace",
      name: localStorage.getItem(WORKSPACE_NAME_KEY) || DEFAULT_WORKSPACE_NAME,
      user_id: userId,
      created_at: now,
      updated_at: now,
    },
  ];
}

export function saveWorkspaces(workspaces: WorkspaceRecord[], userId: string | null = null) {
  localStorage.setItem(workspacesKey(userId), JSON.stringify(workspaces));
}

export function loadActiveWorkspaceId(workspaces = loadWorkspaces(), userId: string | null = null): string {
  const stored = localStorage.getItem(activeWorkspaceKey(userId));
  if (stored && workspaces.some((workspace) => workspace.id === stored)) {
    return stored;
  }
  return workspaces[0]?.id || createWorkspaceId(userId);
}

export function saveActiveWorkspaceId(id: string, userId: string | null = null) {
  localStorage.setItem(activeWorkspaceKey(userId), id);
}

export function loadWorkspaceSources(workspaceId: string): string[] {
  const key = `${WORKSPACE_SOURCES_PREFIX}:${workspaceId}`;
  const raw = localStorage.getItem(key);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function hasWorkspaceSources(workspaceId: string) {
  return localStorage.getItem(`${WORKSPACE_SOURCES_PREFIX}:${workspaceId}`) !== null;
}

export function saveWorkspaceSources(workspaceId: string, sourceIds: string[]) {
  localStorage.setItem(`${WORKSPACE_SOURCES_PREFIX}:${workspaceId}`, JSON.stringify(sourceIds));
}

export function addSourcesToWorkspace(workspaceId: string, newIds: string[]) {
  const existing = loadWorkspaceSources(workspaceId);
  const merged = dedupeIds([...existing, ...newIds]);
  saveWorkspaceSources(workspaceId, merged);
}

export function removeSourcesFromWorkspace(workspaceId: string, removedIds: string[]) {
  if (removedIds.length === 0) return;
  const removed = new Set(removedIds);
  const remaining = loadWorkspaceSources(workspaceId).filter((id) => !removed.has(id));
  saveWorkspaceSources(workspaceId, remaining);
  const titles = loadWorkspaceSourceTitles(workspaceId);
  let changed = false;
  for (const id of removed) {
    if (titles[id] !== undefined) {
      delete titles[id];
      changed = true;
    }
  }
  if (changed) {
    saveWorkspaceSourceTitles(workspaceId, titles);
  }
}

export function loadWorkspaceSourceTitles(workspaceId: string): Record<string, string> {
  const key = `${WORKSPACE_SOURCE_TITLES_PREFIX}:${workspaceId}`;
  const raw = localStorage.getItem(key);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

export function saveWorkspaceSourceTitle(workspaceId: string, sourceId: string, title: string) {
  const titles = loadWorkspaceSourceTitles(workspaceId);
  titles[sourceId] = title;
  saveWorkspaceSourceTitles(workspaceId, titles);
}

function saveWorkspaceSourceTitles(workspaceId: string, titles: Record<string, string>) {
  localStorage.setItem(`${WORKSPACE_SOURCE_TITLES_PREFIX}:${workspaceId}`, JSON.stringify(titles));
}

export function loadWorkspaceConversations(workspaceId: string): string[] {
  return loadIdArray(`${WORKSPACE_CONVERSATIONS_PREFIX}:${workspaceId}`);
}

export function hasWorkspaceConversations(workspaceId: string) {
  return localStorage.getItem(`${WORKSPACE_CONVERSATIONS_PREFIX}:${workspaceId}`) !== null;
}

export function addConversationToWorkspace(workspaceId: string, conversationId: string) {
  const existing = loadWorkspaceConversations(workspaceId);
  saveWorkspaceConversations(workspaceId, dedupeIds([...existing, conversationId]));
}

function saveWorkspaceConversations(workspaceId: string, ids: string[]) {
  localStorage.setItem(`${WORKSPACE_CONVERSATIONS_PREFIX}:${workspaceId}`, JSON.stringify(ids));
}

export function loadWorkspaceArtifacts(workspaceId: string): string[] {
  return loadIdArray(`${WORKSPACE_ARTIFACTS_PREFIX}:${workspaceId}`);
}

export function hasWorkspaceArtifacts(workspaceId: string) {
  return localStorage.getItem(`${WORKSPACE_ARTIFACTS_PREFIX}:${workspaceId}`) !== null;
}

export function addArtifactToWorkspace(workspaceId: string, artifactId: string) {
  const existing = loadWorkspaceArtifacts(workspaceId);
  saveArtifactIds(workspaceId, dedupeIds([...existing, artifactId]));
}

function saveArtifactIds(workspaceId: string, ids: string[]) {
  localStorage.setItem(`${WORKSPACE_ARTIFACTS_PREFIX}:${workspaceId}`, JSON.stringify(ids));
}

export function deleteWorkspace(workspaceId: string, userId: string | null) {
  let workspaces = loadUserWorkspaces(userId).filter(w => w.id !== workspaceId);
  if (workspaces.length === 0) {
    workspaces = [
      userId ? createUserWorkspaceRecord(EMPTY_WORKSPACE_NAME, userId) : createWorkspaceRecord(EMPTY_WORKSPACE_NAME),
    ];
    initializeEmptyWorkspaceData(workspaces[0].id);
  }
  saveWorkspaces(workspaces, userId);
  const nextActive = loadActiveWorkspaceId(workspaces, userId);
  saveActiveWorkspaceId(nextActive, userId);
  localStorage.removeItem(`${WORKSPACE_SOURCES_PREFIX}:${workspaceId}`);
  localStorage.removeItem(`${WORKSPACE_SOURCE_TITLES_PREFIX}:${workspaceId}`);
  localStorage.removeItem(`${WORKSPACE_CONVERSATIONS_PREFIX}:${workspaceId}`);
  localStorage.removeItem(`${WORKSPACE_ARTIFACTS_PREFIX}:${workspaceId}`);
  return { workspaces, nextActive };
}

export function initializeEmptyWorkspaceData(workspaceId: string) {
  saveWorkspaceSources(workspaceId, []);
  saveWorkspaceConversations(workspaceId, []);
  saveArtifactIds(workspaceId, []);
}

function loadIdArray(key: string): string[] {
  const raw = localStorage.getItem(key);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function dedupeIds(ids: string[]): string[] {
  return [...new Set(ids)];
}

export function createWorkspaceRecord(name: string): WorkspaceRecord {
  const now = Date.now();
  return {
    id: createWorkspaceId(null),
    name: name.trim() || DEFAULT_WORKSPACE_NAME,
    user_id: null,
    created_at: now,
    updated_at: now,
  };
}

export function createUserWorkspaceRecord(name: string, userId: string): WorkspaceRecord {
  const now = Date.now();
  return {
    id: createWorkspaceId(userId),
    name: name.trim() || DEFAULT_WORKSPACE_NAME,
    auto_named: true,
    user_id: userId,
    created_at: now,
    updated_at: now,
  };
}

function createWorkspaceId(_userId: string | null) {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `workspace-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
