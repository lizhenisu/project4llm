import type {
  ChatMessage,
  AdminUserList,
  AdminSettings,
  Announcement,
  AuthResponse,
  AuthUser,
  Conversation,
  ConversationListItem,
  MindMapArtifact,
  QueryResponse,
  QueryStreamEvent,
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
  messages: Array<
    Omit<ChatMessage, "requestId" | "feedbackRating" | "imageDataUrl" | "ragProgress"> & {
      request_id?: string | null;
      feedback_rating?: 1 | -1 | null;
      image_data_url?: string | null;
      rag_progress?: ChatMessage["ragProgress"] | null;
    }
  >;
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
  if (response.status === 503) {
    return "当前服务繁忙，请稍后重试。";
  }
  if (response.status === 429) {
    return "请求过于频繁，请稍后重试。";
  }
  if (response.status === 413) {
    return "文件过大，请压缩文件或拆分后再上传。";
  }
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

function normalizeStreamError(detail: string): string {
  if (
    detail.includes("Query service is busy") ||
    detail.includes("Model API concurrency limit reached")
  ) {
    return "当前服务繁忙，请稍后重试。";
  }
  return detail;
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

export function getSourceContent(settings: Settings, docId: string, docVersion?: number): Promise<SourceContent> {
  const versionParam = docVersion ? `&doc_version=${encodeURIComponent(docVersion)}` : "";
  return request<SourceContent>(
    `/sources/content/${encodeURIComponent(docId)}?tenant_id=${encodeURIComponent(settings.tenantId)}${versionParam}`,
    { settings },
  ).then((content) => normalizeSourceContentAssets(settings, content));
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

export function deleteSource(settings: Settings, docId: string, docVersion?: number) {
  const versionParam = docVersion ? `&doc_version=${encodeURIComponent(docVersion)}` : "";
  return request<{ status: string }>(
    `/sources/${encodeURIComponent(docId)}?tenant_id=${encodeURIComponent(settings.tenantId)}${versionParam}`,
    { method: "DELETE", settings },
  );
}

export function renameSource(
  settings: Settings,
  docId: string,
  title: string,
  docVersion?: number,
): Promise<{ status: string; doc_id: string; title: string }> {
  const versionParam = docVersion ? `&doc_version=${encodeURIComponent(docVersion)}` : "";
  return request<{ status: string; doc_id: string; title: string }>(
    `/sources/${encodeURIComponent(docId)}?tenant_id=${encodeURIComponent(settings.tenantId)}${versionParam}`,
    {
      method: "PATCH",
      settings,
      json: { title },
    },
  );
}

export function retrySource(
  settings: Settings,
  docId: string,
): Promise<{ status: string; source: SourceItem }> {
  return request<{ status: string; source: SourceItem }>(
    `/sources/${encodeURIComponent(docId)}/retry?tenant_id=${encodeURIComponent(settings.tenantId)}`,
    { method: "POST", settings },
  );
}

export function queryRag(
  settings: Settings,
  params: {
    query: string;
    requestId: string;
    docIds: string[];
    history: string[];
    imageDataUrl?: string | null;
  },
): Promise<QueryResponse> {
  return request<QueryResponse>("/query", {
    method: "POST",
    settings,
    json: {
      query: params.query,
      request_id: params.requestId,
      query_mode: params.imageDataUrl ? "multimodal" : "text",
      image_data_url: params.imageDataUrl || null,
      history: params.history,
      tenant_id: settings.tenantId,
      acl_groups: settings.aclGroups,
      doc_ids: params.docIds,
      candidate_limit: 20,
      context_limit: RAG_CONTEXT_LIMIT,
    },
  }).then((response) => normalizeQueryResponseAssets(settings, response));
}

export async function queryRagStream(
  settings: Settings,
  params: {
    query: string;
    requestId: string;
    docIds: string[];
    history: string[];
    imageDataUrl?: string | null;
    onEvent: (event: QueryStreamEvent) => void;
  },
): Promise<QueryResponse> {
  const headers = new Headers();
  headers.set("Content-Type", "application/json");
  headers.set("X-RAG-Tenant-ID", settings.tenantId);
  headers.set("X-RAG-ACL-Groups", settings.aclGroups.join(","));
  if (settings.token) {
    headers.set("Authorization", `Bearer ${settings.token}`);
  }
  const response = await fetch(`${settings.apiBaseUrl}/query/stream`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      query: params.query,
      request_id: params.requestId,
      query_mode: params.imageDataUrl ? "multimodal" : "text",
      image_data_url: params.imageDataUrl || null,
      history: params.history,
      tenant_id: settings.tenantId,
      acl_groups: settings.aclGroups,
      doc_ids: params.docIds,
      candidate_limit: 20,
      context_limit: RAG_CONTEXT_LIMIT,
    }),
  });
  if (!response.ok) {
    const detail = await readErrorDetail(response);
    if (response.status === 401) {
      unauthorizedHandler?.();
    }
    throw new ApiError(detail, response.status);
  }
  if (!response.body) {
    throw new ApiError("流式响应不可用", response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let result: QueryResponse | null = null;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      const event = parseStreamEvent(line);
      if (!event) continue;
      if (event.type === "result") {
        result = normalizeQueryResponseAssets(settings, event);
      }
      params.onEvent(event);
      if (event.type === "error") {
        throw new ApiError(normalizeStreamError(event.detail), response.status);
      }
    }
    if (done) break;
  }
  if (buffer.trim()) {
    const event = parseStreamEvent(buffer);
    if (event) {
      if (event.type === "result") {
        result = normalizeQueryResponseAssets(settings, event);
      }
      params.onEvent(event);
      if (event.type === "error") {
        throw new ApiError(normalizeStreamError(event.detail), response.status);
      }
    }
  }
  if (!result) {
    throw new ApiError("回答流结束但没有返回最终结果", response.status);
  }
  return result;
}

function parseStreamEvent(line: string): QueryStreamEvent | null {
  const trimmed = line.trim();
  if (!trimmed) return null;
  try {
    return JSON.parse(trimmed) as QueryStreamEvent;
  } catch {
    return { type: "error", detail: "回答流解析失败" };
  }
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
      tenant_id: settings.tenantId,
      acl_groups: settings.aclGroups,
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
  ).then((conversation) => normalizeConversation(settings, conversation));
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
        image_data_url: message.imageDataUrl || null,
        created_at: message.created_at || null,
        feedback_rating: message.feedbackRating ?? null,
        rag_progress: message.ragProgress || [],
      })),
      source_doc_ids: params.sourceDocIds,
    },
  }).then((conversation) => normalizeConversation(settings, conversation));
}

export function deleteConversation(settings: Settings, conversationId: string) {
  return request<{ status: string; conversation_id: string }>(
    `/conversations/${encodeURIComponent(conversationId)}?tenant_id=${encodeURIComponent(settings.tenantId)}`,
    { method: "DELETE", settings },
  );
}

function normalizeConversation(settings: Settings, conversation: ApiConversation): Conversation {
  return {
    ...conversation,
    messages: conversation.messages.map((message) => ({
      ...message,
      requestId: message.request_id || undefined,
      citations: message.citations?.map((citation) => normalizeCitationAssets(settings, citation)),
      imageDataUrl: message.image_data_url ?? null,
      feedbackRating: message.feedback_rating ?? null,
      ragProgress: message.rag_progress ?? undefined,
    })),
  };
}

function normalizeSourceContentAssets(settings: Settings, content: SourceContent): SourceContent {
  return {
    ...content,
    blocks: content.blocks?.map((block) => ({
      ...block,
      url: normalizeAssetUrl(settings, block.url),
    })),
  };
}

function normalizeQueryResponseAssets(settings: Settings, response: QueryResponse): QueryResponse {
  return {
    ...response,
    citations: response.citations.map((citation) => normalizeCitationAssets(settings, citation)),
  };
}

function normalizeCitationAssets<T extends { metadata: Record<string, unknown> }>(settings: Settings, citation: T): T {
  return {
    ...citation,
    metadata: normalizeMetadataAssets(settings, citation.metadata),
  };
}

function normalizeMetadataAssets(settings: Settings, metadata: Record<string, unknown>): Record<string, unknown> {
  const blocks = metadata.display_blocks;
  if (!Array.isArray(blocks)) {
    return metadata;
  }
  return {
    ...metadata,
    display_blocks: blocks.map((block) => {
      if (!isRecord(block)) return block;
      return {
        ...block,
        url: normalizeAssetUrl(settings, typeof block.url === "string" ? block.url : undefined),
      };
    }),
  };
}

function normalizeAssetUrl(settings: Settings, url: string | undefined): string | undefined {
  if (!url) {
    return url;
  }
  const parsedUrl = new URL(url, window.location.origin);
  if (
    !parsedUrl.pathname.startsWith("/source-assets/")
    && !parsedUrl.pathname.startsWith("/api/source-assets/")
  ) {
    return url;
  }
  const apiBaseUrl = settings.apiBaseUrl.replace(/\/+$/, "");
  const normalizedUrl = url.startsWith("/source-assets/")
    ? new URL(`${apiBaseUrl}${url}`, window.location.origin)
    : parsedUrl;
  normalizedUrl.searchParams.set("tenant_id", settings.tenantId);
  normalizedUrl.searchParams.delete("token");
  return normalizedUrl.toString();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export async function deleteArtifact(
  settings: Settings,
  artifactId: string,
  workspaceId: string,
): Promise<{ status: string; artifact_id: string }> {
  return request(`/artifacts/${artifactId}?tenant_id=${encodeURIComponent(settings.tenantId)}&workspace_id=${encodeURIComponent(workspaceId)}`, {
    method: "DELETE",
    settings,
  });
}

export async function renameArtifact(
  settings: Settings,
  artifactId: string,
  title: string,
  workspaceId: string,
): Promise<{ status: string; artifact_id: string; title: string }> {
  return request(`/artifacts/${artifactId}?tenant_id=${encodeURIComponent(settings.tenantId)}&workspace_id=${encodeURIComponent(workspaceId)}`, {
    method: "PATCH",
    settings,
    json: { title },
  });
}

export async function listArtifacts(settings: Settings, workspaceId: string): Promise<MindMapArtifact[]> {
  const payload = await request<{ artifacts: MindMapArtifact[] }>(
    `/artifacts?tenant_id=${encodeURIComponent(settings.tenantId)}&workspace_id=${encodeURIComponent(workspaceId)}`,
    { settings },
  );
  return payload.artifacts;
}

export function getArtifact(settings: Settings, artifactId: string, workspaceId: string): Promise<MindMapArtifact> {
  return request<MindMapArtifact>(
    `/artifacts/${encodeURIComponent(artifactId)}?tenant_id=${encodeURIComponent(settings.tenantId)}&workspace_id=${encodeURIComponent(workspaceId)}`,
    { settings },
  );
}

export function createMindMap(
  settings: Settings,
  title: string,
  sourceDocIds: string[],
  workspaceId: string,
): Promise<MindMapArtifact> {
  return request<MindMapArtifact>("/artifacts/mindmap", {
    method: "POST",
    settings,
    json: {
      title,
      tenant_id: settings.tenantId,
      workspace_id: workspaceId,
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
  workspaceId: string,
): Promise<MindMapArtifact> {
  return request<MindMapArtifact>("/artifacts/table", {
    method: "POST",
    settings,
    json: {
      title,
      tenant_id: settings.tenantId,
      workspace_id: workspaceId,
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

export function refreshLoginToken(settings: Settings): Promise<AuthResponse> {
  return request<AuthResponse>("/auth/token/refresh", {
    method: "POST",
    settings,
  });
}

export function getCurrentUser(settings: Settings): Promise<AuthUser> {
  return request<AuthUser>("/auth/me", { settings });
}

export function updateCurrentUser(
  settings: Settings,
  params: { username: string; displayName: string; avatarUrl: string },
): Promise<AuthUser> {
  return request<AuthUser>("/auth/me", {
    method: "PATCH",
    settings,
    json: {
      username: params.username,
      display_name: params.displayName,
      avatar_url: params.avatarUrl,
    },
  });
}

export function changeCurrentPassword(
  settings: Settings,
  params: { currentPassword: string; newPassword: string },
): Promise<{ status: string }> {
  return request<{ status: string }>("/auth/password", {
    method: "PATCH",
    settings,
    json: {
      current_password: params.currentPassword,
      new_password: params.newPassword,
    },
  });
}

export async function listAnnouncements(settings: Settings): Promise<Announcement[]> {
  const payload = await request<{ announcements: Announcement[] }>("/announcements?limit=5", {
    settings,
  });
  return payload.announcements;
}

export function listAdminUsers(
  settings: Settings,
  params: { query?: string; limit?: number; offset?: number } = {},
): Promise<AdminUserList> {
  const query = new URLSearchParams();
  if (params.query) query.set("q", params.query);
  if (params.limit) query.set("limit", String(params.limit));
  if (params.offset) query.set("offset", String(params.offset));
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return request<AdminUserList>(`/admin/users${suffix}`, {
    settings,
  });
}

export function getAdminSettings(settings: Settings): Promise<AdminSettings> {
  return request<AdminSettings>("/admin/settings", {
    settings,
  });
}

export function updateRegistrationEnabled(settings: Settings, registrationEnabled: boolean): Promise<AdminSettings> {
  return request<AdminSettings>("/admin/settings/registration", {
    method: "PATCH",
    settings,
    json: { registration_enabled: registrationEnabled },
  });
}

export function updateAdminUserStatus(
  settings: Settings,
  userId: string,
  status: "active" | "banned",
): Promise<AuthUser> {
  return request<AuthUser>(`/admin/users/${encodeURIComponent(userId)}/status`, {
    method: "PATCH",
    settings,
    json: { status },
  });
}

export function updateAdminUsers(
  settings: Settings,
  users: Array<{
    user_id: string;
    status?: "active" | "banned";
    profile_name_edit_allowed?: boolean;
    avatar_edit_allowed?: boolean;
  }>,
): Promise<AdminUserList> {
  return request<AdminUserList>("/admin/users/bulk", {
    method: "PATCH",
    settings,
    json: { users },
  });
}

export function publishAnnouncement(
  settings: Settings,
  params: { title: string; content: string; linkUrl?: string; linkLabel?: string },
): Promise<Announcement> {
  return request<Announcement>("/admin/announcements", {
    method: "POST",
    settings,
    json: {
      title: params.title,
      content: params.content,
      link_url: params.linkUrl || "",
      link_label: params.linkLabel || "",
    },
  });
}

export function deleteAnnouncement(
  settings: Settings,
  announcementId: string,
): Promise<{ status: string; announcement_id: string }> {
  return request<{ status: string; announcement_id: string }>(
    `/admin/announcements/${encodeURIComponent(announcementId)}`,
    {
      method: "DELETE",
      settings,
    },
  );
}
