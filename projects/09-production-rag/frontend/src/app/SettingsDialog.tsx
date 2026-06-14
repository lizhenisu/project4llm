import { useEffect, useState } from "react";
import type { WorkspaceRecord } from "../lib/types";

type Props = {
  open: boolean;
  workspaceName: string;
  workspaces: WorkspaceRecord[];
  activeWorkspaceId: string;
  onClose: () => void;
  onNewWorkspace: () => void;
  onRenameWorkspace: (name: string) => void;
  onSelectWorkspace: (id: string) => void;
};

export function SettingsDialog({
  open,
  workspaceName,
  workspaces,
  activeWorkspaceId,
  onClose,
  onNewWorkspace,
  onRenameWorkspace,
  onSelectWorkspace,
}: Props) {
  const [nameDraft, setNameDraft] = useState(workspaceName);

  useEffect(() => {
    setNameDraft(workspaceName);
  }, [workspaceName]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <div
        className="settings-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="settings-dialog-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <h2 id="settings-dialog-title">数据库设置</h2>
        <section className="settings-section">
          <div>
            <strong>知识库工作区</strong>
            <p>选择当前使用的数据库，或创建新的数据库入口。</p>
          </div>
          <div className="database-list" aria-label="数据库列表">
            {workspaces.map((workspace) => {
              const active = workspace.id === activeWorkspaceId;
              return (
                <button
                  key={workspace.id}
                  type="button"
                  className={active ? "active" : ""}
                  aria-pressed={active}
                  onClick={() => onSelectWorkspace(workspace.id)}
                >
                  <span>{workspace.name}</span>
                  {active ? <small>当前数据库</small> : <small>点击切换</small>}
                </button>
              );
            })}
          </div>
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
              onClick={() => onRenameWorkspace(nameDraft.trim())}
              disabled={!nameDraft.trim()}
            >
              重命名数据库
            </button>
          </div>
        </section>
        <div className="dialog-actions">
          <button type="button" onClick={onClose}>
            关闭
          </button>
        </div>
      </div>
    </div>
  );
}
