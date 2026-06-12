export type SourceStatus = "uploading" | "processing" | "ready" | "failed";

export type SourceItem = {
  doc_id: string;
  title: string;
  source_type: string;
  source_uri: string;
  doc_version: number;
  chunk_count: number;
  acl_groups: string[];
  status: SourceStatus;
  current: boolean;
  created_at?: number | null;
  updated_at?: number | null;
  selected?: boolean;
  error?: string;
};

export type Citation = {
  doc_id: string;
  title: string;
  source_uri: string;
  source_type: string;
  chunk_index: number;
  score: number;
  rerank_score?: number | null;
  acl_groups: string[];
  metadata: Record<string, unknown>;
  text_preview?: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  requestId?: string;
  citations?: Citation[];
  status?: "sending" | "done" | "failed";
};

export type QueryResponse = {
  request_id: string;
  answer: string;
  citations: Citation[];
  trace?: Record<string, unknown>;
};

export type MindMapNode = {
  id: string;
  label: string;
  children?: MindMapNode[];
  citationIds?: string[];
};

export type MindMapArtifact = {
  id: string;
  title: string;
  status: "generating" | "ready" | "failed";
  tenant_id: string;
  source_doc_ids: string[];
  created_at: number;
  updated_at: number;
  root?: MindMapNode | null;
  error?: string;
};

export type Settings = {
  apiBaseUrl: string;
  token: string;
  tenantId: string;
  aclGroups: string[];
};
