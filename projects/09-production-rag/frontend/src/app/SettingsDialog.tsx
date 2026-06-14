import { useEffect, useState } from "react";
import { MoreHorizontal, PencilLine, X } from "lucide-react";
import type { WorkspaceRecord } from "../lib/types";

type Props = {
  open: boolean;
  workspaceName: string;
  workspaces: WorkspaceRecord[];
  activeWorkspaceId: string;
  authenticated: boolean;
  onClose: () => void;
  onNewWorkspace: () => void;
  onRenameWorkspace: (id: string, name: string) => void;
  onSelectWorkspace: (id: string) => void;
};

export function SettingsDialog({
  open,
  workspaceName,
  workspaces,
  activeWorkspaceId,
  authenticated,
  onClose,
  onNewWorkspace,
  onRenameWorkspace,
  onSelectWorkspace,
}: Props) {
  const [nameDraft, setNameDraft] = useState(workspaceName);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);

  useEffect(() => {
    setNameDraft(workspaceName);
  }, [workspaceName]);

  if (!open) return null;

  return (
    <>
      <div className="modal-backdrop" role="presentation" onMouseDown={onClose} />
      <aside
        className="settings-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-dialog-title"
        aria-label="数据库设置"
      >
        <div className="settings-panel-header">
        <h2 id="settings-dialog-title">数据库设置</h2>
          <button type="button" className="icon-button" aria-label="关闭设置" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
        <section className="settings-section">
          <div>
            <strong>知识库工作区</strong>
            <p>选择当前使用的数据库，或创建新的数据库入口。</p>
          </div>
          <div className="database-list" aria-label="数据库列表" onMouseDown={(event) => event.stopPropagation()}>
            {workspaces.length === 0 ? (
              <p className="muted-text">暂无数据库，登录后可新建。</p>
            ) : (
              workspaces.map((workspace) => {
              const active = workspace.id === activeWorkspaceId;
                const isRenaming = renamingId === workspace.id;
              return (
                  <div
                  key={workspace.id}
                    className={`database-list-item ${active ? "active" : ""}`}
                  >
                    <button
                  type="button"
                      className="database-item-trigger"
                  aria-pressed={active}
                  onClick={() => onSelectWorkspace(workspace.id)}
                >
                      <span>
                        {isRenaming ? (
                          <input
                            className="inline-title-input"
                            value={renameDraft}
                            autoFocus
                            onKeyDown={(e) => {
                              if (e.key === "Enter") {
                                onRenameWorkspace(workspace.id, renameDraft.trim());
                                setRenamingId(null);
                              }
                              if (e.key === "Escape") {
                                setRenamingId(null);
                              }
                            }}
                            onBlur={() => {
                              setTimeout(() => {
                                if (document.activeElement?.tagName === "INPUT") return;
                                onRenameWorkspace(workspace.id, renameDraft.trim());
                                setRenamingId(null);
                              }, 120);
                            }}
                            onChange={(e) => setRenameDraft(e.target.value)}
                            onClick={(e) => e.stopPropagation()}
                          />
                        ) : (
                          workspace.name
                      )}
                      </span>
                      <span className="current-label">
                        {active ? "当前数据库" : "点击切换"}
                      </span>
                </button>
                    {authenticated && !isRenaming ? (
                      <div className="db-item-menu">
                        <button
                          type="button"
                          className="icon-button"
                          aria-label="更多操作"
                          onClick={(e) => {
                            e.stopPropagation();
                            setOpenMenuId(openMenuId === workspace.id ? null : workspace.id);
                          }}
                        >
                          <MoreHorizontal size={14} />
                        </button>
                        {openMenuId === workspace.id ? (
                          <div className="dropdown-menu" role="menu">
                            <button
                              type="button"
                              role="menuitem"
                              onClick={(e) => {
                                e.stopPropagation();
                                setRenamingId(workspace.id);
                                setRenameDraft(workspace.name);
                                setOpenMenuId(null);
                              }}
                            >
                              <PencilLine size={14} />
                              重命名知识库
                            </button>
                          </div>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
              );
              })
            )}
            {openMenuId && (
              <div className="dropdown-backdrop" onMouseDown={() => setOpenMenuId(null)} />
            )}
          </div>
          {authenticated ? (
            <>
              <label>
                当前数据库名称
                <input value={nameDraft} onChange={(event) => setNameDraft(event.target.value)} />
              </label>
              <div className="dialog-actions">
                <button type="button" onClick={onNewWorkspace}>
                  新建数据库
                </button>
                <button
                  type="button"
                  className="primary-pill"
                  onClick={() => onRenameWorkspace(activeWorkspaceId, nameDraft.trim())}
                  disabled={!nameDraft.trim()}
                >
                  重命名数据库
                </button>
              </div>
            </>
          ) : (
            <p className="muted-text">登录后可创建与重命名数据库。</p>
          )}
        </section>
      </aside>
    </>
  );
}
