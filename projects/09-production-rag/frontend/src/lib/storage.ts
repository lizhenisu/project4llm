import type { Settings, WorkspaceRecord } from "./types";

const KEY = "production-rag-settings";
const WORKSPACE_NAME_KEY = "production-rag-workspace-name";
const WORKSPACES_PREFIX = "production-rag-workspaces";
const ACTIVE_WORKSPACE_PREFIX = "production-rag-active-workspace-id";
export const DEFAULT_WORKSPACE_NAME = "Production RAG 知识库";

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
