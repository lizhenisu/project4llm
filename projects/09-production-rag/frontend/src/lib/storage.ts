import type { Settings, WorkspaceRecord } from "./types";

const KEY = "production-rag-settings";
const WORKSPACE_NAME_KEY = "production-rag-workspace-name";
const WORKSPACES_KEY = "production-rag-workspaces";
const ACTIVE_WORKSPACE_KEY = "production-rag-active-workspace-id";
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

export function loadWorkspaceName(): string {
  const workspaces = loadWorkspaces();
  const activeId = loadActiveWorkspaceId(workspaces);
  return workspaces.find((workspace) => workspace.id === activeId)?.name || workspaces[0]?.name || DEFAULT_WORKSPACE_NAME;
}

export function saveWorkspaceName(name: string) {
  localStorage.setItem(WORKSPACE_NAME_KEY, name);
}

export function loadWorkspaces(): WorkspaceRecord[] {
  const raw = localStorage.getItem(WORKSPACES_KEY);
  if (raw) {
    try {
      const parsed = JSON.parse(raw) as WorkspaceRecord[];
      const workspaces = parsed
        .filter((item) => item?.id && item?.name)
        .map((item) => ({
          id: item.id,
          name: item.name,
          created_at: item.created_at || Date.now(),
          updated_at: item.updated_at || item.created_at || Date.now(),
        }));
      if (workspaces.length > 0) {
        return workspaces;
      }
    } catch {
      localStorage.removeItem(WORKSPACES_KEY);
    }
  }
  const now = Date.now();
  return [
    {
      id: "default-workspace",
      name: localStorage.getItem(WORKSPACE_NAME_KEY) || DEFAULT_WORKSPACE_NAME,
      created_at: now,
      updated_at: now,
    },
  ];
}

export function saveWorkspaces(workspaces: WorkspaceRecord[]) {
  localStorage.setItem(WORKSPACES_KEY, JSON.stringify(workspaces));
}

export function loadActiveWorkspaceId(workspaces = loadWorkspaces()): string {
  const stored = localStorage.getItem(ACTIVE_WORKSPACE_KEY);
  if (stored && workspaces.some((workspace) => workspace.id === stored)) {
    return stored;
  }
  return workspaces[0]?.id || createWorkspaceId();
}

export function saveActiveWorkspaceId(id: string) {
  localStorage.setItem(ACTIVE_WORKSPACE_KEY, id);
}

export function createWorkspaceRecord(name: string): WorkspaceRecord {
  const now = Date.now();
  return {
    id: createWorkspaceId(),
    name: name.trim() || DEFAULT_WORKSPACE_NAME,
    created_at: now,
    updated_at: now,
  };
}

function createWorkspaceId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `workspace-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
