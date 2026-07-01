export type SourceStatus = "uploading" | "queued" | "processing" | "ready" | "failed";

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
  workspace_alias_ids?: string[];
  selected?: boolean;
  error?: string;
  retryable?: boolean;
  attempt_count?: number;
  next_attempt_at?: number;
  dead_lettered?: boolean;
  ingestion_stage?: string;
  progress_percent?: number;
  progress_detail?: string;
  eta_seconds?: number | null;
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
  blocks?: Array<{
    type: "text" | "image";
    text?: string;
    title?: string;
    url?: string;
    page?: string;
  }>;
  suggested_title?: string;
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
  text?: string;
  text_preview?: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  requestId?: string;
  citations?: Citation[];
  imageDataUrl?: string | null;
  status?: "sending" | "done" | "failed";
  created_at?: number | null;
  feedbackRating?: 1 | -1 | null;
  ragProgress?: RagProgressStage[];
};

export type QueryResponse = {
  request_id: string;
  answer: string;
  citations: Citation[];
  trace?: Record<string, unknown>;
};

export type RagProgressStatus = "pending" | "active" | "done" | "failed";

export type RagProgressStage = {
  stage: string;
  label: string;
  detail: string;
  status: RagProgressStatus;
  latency_ms?: number;
  candidate_count?: number;
  reranked_count?: number;
  context_count?: number;
};

export type QueryStreamEvent =
  | ({ type: "stage" } & RagProgressStage & Record<string, unknown>)
  | { type: "result"; request_id: string; answer: string; citations: Citation[]; trace?: Record<string, unknown> }
  | { type: "error"; detail: string };

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
  workspace_id?: string;
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

export type WorkspaceRecord = {
  id: string;
  name: string;
  auto_named?: boolean;
  user_id: string | null;
  created_at: number;
  updated_at: number;
};

export type AuthUser = {
  id: string;
  username: string;
  display_name: string;
  role: "admin" | "user";
  tenant_id: string;
  created_at: number;
  avatar_url?: string;
  status?: "active" | "banned";
  profile_name_edit_allowed?: boolean;
  avatar_edit_allowed?: boolean;
  last_login_at?: number | null;
};

export type AdminUserList = {
  users: AuthUser[];
  total: number;
  limit: number;
  offset: number;
  query: string;
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
  link_url?: string;
  link_label?: string;
  author_id: string;
  author_name?: string | null;
  created_at: number;
};

export type AdminSettings = {
  registration_enabled: boolean;
  latest_announcement?: Announcement | null;
};

export type AdminDeadLetterTask = {
  tenant_id: string;
  task_id: string;
  title: string;
  source_type: string;
  error: string;
  attempt_count: number;
  dead_lettered_at: number;
  updated_at: number;
};

export type AdminDeadLetterList = {
  tasks: AdminDeadLetterTask[];
  total: number;
  limit: number;
  offset: number;
};

export type AdminIngestionAuditEvent = {
  id: string;
  actor_user_id: string;
  tenant_id: string;
  task_id: string;
  operation: string;
  outcome: string;
  detail: string;
  created_at: number;
};

export type AdminIngestionAuditList = {
  events: AdminIngestionAuditEvent[];
  total: number;
  limit: number;
  offset: number;
};

export type AdminIngestionRedriveResponse = {
  results: Array<{
    tenant_id: string;
    task_id: string;
    outcome: string;
  }>;
  queued: number;
  rejected: number;
};
