import type { Settings } from "./types";

const KEY = "production-rag-settings";

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
