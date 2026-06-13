import type {
  ChatMessage,
  Announcement,
  AuthResponse,
  AuthUser,
  Conversation,
  ConversationListItem,
  MindMapArtifact,
  QueryResponse,
  Settings,
  SourceContent,
  SourceItem,
} from "./types";

const RAG_CONTEXT_LIMIT = 5;
let unauthorizedHandler: (() => void) | null = null;

export function setUnauthorizedHandler(handler: (() => void) | null) {
  unauthorizedHandler = handler;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

type RequestOptions = RequestInit & {
  settings: Settings;
  json?: unknown;
};

type ApiConversation = Omit<Conversation, "messages"> & {
  messages: Array<Omit<ChatMessage, "requestId"> & { request_id?: string | null }>;
};

async function request<T>(path: string, options: RequestOptions): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set("X-RAG-Tenant-ID", options.settings.tenantId);
  headers.set("X-RAG-ACL-Groups", options.settings.aclGroups.join(","));
  if (options.settings.token) {
    headers.set("Authorization", `Bearer ${options.settings.token}`);
  }
  let body = options.body;
  if (options.json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(options.json);
  }
  const response = await fetch(`${options.settings.apiBaseUrl}${path}`, {
    ...options,
    headers,
    body,
  });
  if (!response.ok) {
    const detail = await readErrorDetail(response);
    if (response.status === 401) {
      unauthorizedHandler?.();
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

async function readErrorDetail(response: Response): Promise<string> {
  const contentType = response.headers.get("content-type") ?? "";
  const defaultMessage = response.statusText || `HTTP ${response.status}`;
  if (contentType.includes("application/json")) {
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string") {
        return payload.detail;
      }
      return JSON.stringify(payload);
    } catch {
      return defaultMessage;
    }
  }
  const text = await response.text();
  return text || defaultMessage;
}

export function health(settings: Settings) {
  return request<{ status: string }>("/health", { settings });
}

export async function listSources(settings: Settings): Promise<SourceItem[]> {
  const payload = await request<{ sources: SourceItem[] }>(
    `/sources?tenant_id=${encodeURIComponent(settings.tenantId)}`,
    { settings },
  );
  return payload.sources;
}

export function getSource(settings: Settings, docId: string): Promise<SourceItem> {
  return request<SourceItem>(
    `/sources/${encodeURIComponent(docId)}?tenant_id=${encodeURIComponent(settings.tenantId)}`,
    { settings },
  );
}

export function getSourceContent(settings: Settings, docId: string): Promise<SourceContent> {
  return request<SourceContent>(
    `/sources/content/${encodeURIComponent(docId)}?tenant_id=${encodeURIComponent(settings.tenantId)}`,
    { settings },
  );
}

export async function uploadSource(settings: Settings, file: File): Promise<SourceItem[]> {
  const form = new FormData();
  form.append("file", file);
  form.append("tenant_id", settings.tenantId);
  form.append("acl_groups", settings.aclGroups.join(","));
  const payload = await request<{ sources: SourceItem[] }>("/sources/upload", {
    method: "POST",
    settings,
    body: form,
  });
  return payload.sources;
}

export function deleteSource(settings: Settings, docId: string) {
  return request<{ status: string }>(
    `/sources/${encodeURIComponent(docId)}?tenant_id=${encodeURIComponent(settings.tenantId)}`,
    { method: "DELETE", settings },
  );
}

export function queryRag(
  settings: Settings,
  params: {
    query: string;
    docIds: string[];
    history: string[];
  },
) {
  return request<QueryResponse>("/query", {
    method: "POST",
    settings,
    json: {
      query: params.query,
      query_mode: "text",
      history: params.history,
      tenant_id: settings.tenantId,
      acl_groups: settings.aclGroups,
      doc_ids: params.docIds,
      candidate_limit: 20,
      context_limit: RAG_CONTEXT_LIMIT,
    },
  });
}

export function sendFeedback(
  settings: Settings,
  requestId: string,
  rating: 1 | -1,
  selectedDocIds: string[],
  comment = "",
) {
  return request<{ status: string }>("/feedback", {
    method: "POST",
    settings,
    json: {
      request_id: requestId,
      rating,
      comment,
      selected_doc_ids: selectedDocIds,
    },
  });
}

export async function listConversations(settings: Settings): Promise<ConversationListItem[]> {
  const payload = await request<{ conversations: ConversationListItem[] }>(
    `/conversations?tenant_id=${encodeURIComponent(settings.tenantId)}`,
    { settings },
  );
  return payload.conversations;
}

export function getConversation(settings: Settings, conversationId: string): Promise<Conversation> {
  return request<ApiConversation>(
    `/conversations/${encodeURIComponent(conversationId)}?tenant_id=${encodeURIComponent(settings.tenantId)}`,
    { settings },
  ).then(normalizeConversation);
}

export function saveConversation(
  settings: Settings,
  params: {
    id?: string | null;
    title: string;
    messages: ChatMessage[];
    sourceDocIds: string[];
  },
): Promise<Conversation> {
  return request<ApiConversation>("/conversations", {
    method: "POST",
    settings,
    json: {
      id: params.id || null,
      tenant_id: settings.tenantId,
      title: params.title,
      messages: params.messages.map((message) => ({
        id: message.id,
        role: message.role,
        content: message.content,
        status: message.status || "done",
        request_id: message.requestId || null,
        citations: message.citations || [],
        created_at: message.created_at || null,
      })),
      source_doc_ids: params.sourceDocIds,
    },
  }).then(normalizeConversation);
}

export function deleteConversation(settings: Settings, conversationId: string) {
  return request<{ status: string; conversation_id: string }>(
    `/conversations/${encodeURIComponent(conversationId)}?tenant_id=${encodeURIComponent(settings.tenantId)}`,
    { method: "DELETE", settings },
  );
}

function normalizeConversation(conversation: ApiConversation): Conversation {
  return {
    ...conversation,
    messages: conversation.messages.map((message) => ({
      ...message,
      requestId: message.request_id || undefined,
    })),
  };
}

export async function deleteArtifact(
  settings: Settings,
  artifactId: string,
): Promise<{ status: string; artifact_id: string }> {
  return request(`/artifacts/${artifactId}?tenant_id=${encodeURIComponent(settings.tenantId)}`, {
    method: "DELETE",
    settings,
  });
}

export async function renameArtifact(
  settings: Settings,
  artifactId: string,
  title: string,
): Promise<{ status: string; artifact_id: string; title: string }> {
  return request(`/artifacts/${artifactId}?tenant_id=${encodeURIComponent(settings.tenantId)}`, {
    method: "PATCH",
    settings,
    json: { title },
  });
}

export async function listArtifacts(settings: Settings): Promise<MindMapArtifact[]> {
  const payload = await request<{ artifacts: MindMapArtifact[] }>(
    `/artifacts?tenant_id=${encodeURIComponent(settings.tenantId)}`,
    { settings },
  );
  return payload.artifacts;
}

export function getArtifact(settings: Settings, artifactId: string): Promise<MindMapArtifact> {
  return request<MindMapArtifact>(
    `/artifacts/${encodeURIComponent(artifactId)}?tenant_id=${encodeURIComponent(settings.tenantId)}`,
    { settings },
  );
}

export function createMindMap(
  settings: Settings,
  title: string,
  sourceDocIds: string[],
): Promise<MindMapArtifact> {
  return request<MindMapArtifact>("/artifacts/mindmap", {
    method: "POST",
    settings,
    json: {
      title,
      tenant_id: settings.tenantId,
      acl_groups: settings.aclGroups,
      source_doc_ids: sourceDocIds,
      context_limit: RAG_CONTEXT_LIMIT,
    },
  });
}

export function createDataTable(
  settings: Settings,
  title: string,
  sourceDocIds: string[],
): Promise<MindMapArtifact> {
  return request<MindMapArtifact>("/artifacts/table", {
    method: "POST",
    settings,
    json: {
      title,
      tenant_id: settings.tenantId,
      acl_groups: settings.aclGroups,
      source_doc_ids: sourceDocIds,
      context_limit: RAG_CONTEXT_LIMIT,
    },
  });
}

export function registerAccount(
  settings: Settings,
  params: { username: string; password: string; displayName?: string },
): Promise<AuthResponse> {
  return request<AuthResponse>("/auth/register", {
    method: "POST",
    settings: { ...settings, token: "" },
    json: {
      username: params.username,
      password: params.password,
      display_name: params.displayName || null,
    },
  });
}

export function loginAccount(
  settings: Settings,
  params: { username: string; password: string },
): Promise<AuthResponse> {
  return request<AuthResponse>("/auth/login", {
    method: "POST",
    settings: { ...settings, token: "" },
    json: {
      username: params.username,
      password: params.password,
    },
  });
}

export function logoutAccount(settings: Settings): Promise<{ status: string }> {
  return request<{ status: string }>("/auth/logout", {
    method: "POST",
    settings,
  });
}

export function getCurrentUser(settings: Settings): Promise<AuthUser> {
  return request<AuthUser>("/auth/me", { settings });
}

export async function listAnnouncements(settings: Settings): Promise<Announcement[]> {
  const payload = await request<{ announcements: Announcement[] }>("/announcements?limit=5", {
    settings,
  });
  return payload.announcements;
}

export async function listAdminUsers(settings: Settings): Promise<AuthUser[]> {
  const payload = await request<{ users: AuthUser[] }>("/admin/users", {
    settings,
  });
  return payload.users;
}

export function publishAnnouncement(
  settings: Settings,
  params: { title: string; content: string },
): Promise<Announcement> {
  return request<Announcement>("/admin/announcements", {
    method: "POST",
    settings,
    json: params,
  });
}
