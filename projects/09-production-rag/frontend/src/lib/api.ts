import type { MindMapArtifact, QueryResponse, Settings, SourceItem } from "./types";

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
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload);
    } catch {
      detail = await response.text();
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
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
      context_limit: 5,
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

export async function listArtifacts(settings: Settings): Promise<MindMapArtifact[]> {
  const payload = await request<{ artifacts: MindMapArtifact[] }>(
    `/artifacts?tenant_id=${encodeURIComponent(settings.tenantId)}`,
    { settings },
  );
  return payload.artifacts;
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
    },
  });
}
