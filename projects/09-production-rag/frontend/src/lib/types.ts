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
  child_doc_ids?: string[];
  selected?: boolean;
  error?: string;
};

export type SourceContent = {
  doc_id: string;
  title: string;
  source_type: string;
  source_uri: string;
  doc_version: number;
  child_doc_ids: string[];
  guide: string;
  tags: string[];
  text: string;
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
  created_at?: number | null;
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

export type DataTableArtifact = {
  title: string;
  columns: string[];
  rows: string[][];
  summary?: string;
};

export type MindMapArtifact = {
  id: string;
  title: string;
  status: "generating" | "ready" | "failed";
  tenant_id: string;
  source_doc_ids: string[];
  created_at: number;
  updated_at: number;
  artifact_type?: "mindmap" | "table";
  root?: MindMapNode | null;
  table?: DataTableArtifact | null;
  error?: string;
};

export type ConversationListItem = {
  id: string;
  tenant_id: string;
  title: string;
  message_count: number;
  source_doc_ids: string[];
  created_at: number;
  updated_at: number;
};

export type Conversation = {
  id: string;
  tenant_id: string;
  title: string;
  messages: ChatMessage[];
  source_doc_ids: string[];
  created_at: number;
  updated_at: number;
};

export type Settings = {
  apiBaseUrl: string;
  token: string;
  tenantId: string;
  aclGroups: string[];
};

export type AuthUser = {
  id: string;
  username: string;
  display_name: string;
  role: "admin" | "user";
  tenant_id: string;
  created_at: number;
  last_login_at?: number | null;
};

export type AuthResponse = {
  user: AuthUser;
  token: string;
  expires_at: number;
};

export type Announcement = {
  id: string;
  title: string;
  content: string;
  author_id: string;
  author_name?: string | null;
  created_at: number;
};
