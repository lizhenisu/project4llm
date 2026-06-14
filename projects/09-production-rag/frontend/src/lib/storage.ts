import type { Settings } from "./types";

const KEY = "production-rag-settings";
const WORKSPACE_NAME_KEY = "production-rag-workspace-name";
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
  return localStorage.getItem(WORKSPACE_NAME_KEY) || DEFAULT_WORKSPACE_NAME;
}

export function saveWorkspaceName(name: string) {
  localStorage.setItem(WORKSPACE_NAME_KEY, name);
}
