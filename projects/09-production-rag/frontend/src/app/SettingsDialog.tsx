import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { Database, MoreHorizontal, PencilLine, Plus, Trash2, X } from "lucide-react";
import type { WorkspaceRecord } from "../lib/types";

type Props = {
  open: boolean;
  workspaces: WorkspaceRecord[];
  activeWorkspaceId: string;
  authenticated: boolean;
  onClose: () => void;
  onNewWorkspace: () => void;
  onRenameWorkspace: (id: string, name: string) => void;
  onDeleteWorkspace: (id: string) => void | Promise<void>;
  onSelectWorkspace: (id: string) => void;
};

export function SettingsDialog({
  open,
  workspaces,
  activeWorkspaceId,
  authenticated,
  onClose,
  onNewWorkspace,
  onRenameWorkspace,
  onDeleteWorkspace,
  onSelectWorkspace,
}: Props) {
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const [menuPosition, setMenuPosition] = useState<{ top: number; right: number } | null>(null);

  useEffect(() => {
    if (!openMenuId) return;
    function handleOutsideMenuClick(event: MouseEvent) {
      const target = event.target as HTMLElement;
      if (target.closest(".settings-db-menu") || target.closest(".settings-db-menu-trigger")) {
        return;
      }
      setOpenMenuId(null);
      setMenuPosition(null);
    }
    document.addEventListener("mousedown", handleOutsideMenuClick);
    return () => document.removeEventListener("mousedown", handleOutsideMenuClick);
  }, [openMenuId]);

  if (!open) return null;

  return (
    <>
      <div className="settings-backdrop" role="presentation" onMouseDown={onClose} />
      <aside
        className="settings-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-dialog-title"
        aria-label="知识库设置"
      >
        <div className="settings-panel-header">
          <div>
            <h2 id="settings-dialog-title">知识库设置</h2>
            <p>管理当前账号可用的知识库工作区。</p>
          </div>
          <button type="button" className="icon-button" aria-label="关闭设置" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
        <section className="settings-section">
          <div className="settings-section-heading">
            <div className="settings-section-icon">
              <Database size={18} />
            </div>
            <div>
              <strong>知识库工作区</strong>
              <p>选择当前使用的数据库，或通过行内菜单重命名与删除。</p>
            </div>
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
                    className={`database-list-item ${active ? "active" : ""}${isRenaming ? " is-editing" : ""}`}
                  >
                    <button
                      type="button"
                      className="database-item-trigger"
                      aria-pressed={active}
                      onClick={() => onSelectWorkspace(workspace.id)}
                    >
                      <span className={`database-name${isRenaming ? " is-editing" : ""}`}>
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
                          className="icon-button settings-db-menu-trigger"
                          aria-label="更多操作"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (openMenuId === workspace.id) {
                              setOpenMenuId(null);
                              setMenuPosition(null);
                              return;
                            }
                            const rect = e.currentTarget.getBoundingClientRect();
                            setMenuPosition({
                              top: rect.bottom + 4,
                              right: document.documentElement.clientWidth - rect.right,
                            });
                            setOpenMenuId(workspace.id);
                          }}
                        >
                          <MoreHorizontal size={14} />
                        </button>
                        {openMenuId === workspace.id && menuPosition ? createPortal(
                          <>
                          <div className="dropdown-backdrop" onMouseDown={() => setOpenMenuId(null)} />
                          <div
                            className="dropdown-menu settings-db-menu"
                            role="menu"
                            style={{ top: menuPosition.top, right: menuPosition.right }}
                            onMouseDown={(e) => e.stopPropagation()}
                          >
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
                            <button
                              type="button"
                              role="menuitem"
                              className="danger"
                              onClick={(e) => {
                                e.stopPropagation();
                                setOpenMenuId(null);
                                void onDeleteWorkspace(workspace.id);
                              }}
                            >
                              <Trash2 size={14} />
                              删除知识库
                            </button>
                          </div>
                          </>,
                          document.body,
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                );
              })
            )}
          </div>
          {authenticated ? (
            <div className="settings-panel-actions">
              <button type="button" className="primary-pill" onClick={onNewWorkspace}>
                <Plus size={16} />
                新建数据库
              </button>
            </div>
          ) : (
            <p className="muted-text">登录后可创建、切换与管理数据库。</p>
          )}
        </section>
      </aside>
    </>
  );
}
