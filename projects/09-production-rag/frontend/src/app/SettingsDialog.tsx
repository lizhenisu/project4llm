import { useEffect, useState } from "react";
import type { Settings } from "../lib/types";

type Props = {
  open: boolean;
  settings: Settings;
  workspaceName: string;
  onClose: () => void;
  onSave: (settings: Settings) => void;
  onNewWorkspace: () => void;
  onRenameWorkspace: (name: string) => void;
};

export function SettingsDialog({
  open,
  settings,
  workspaceName,
  onClose,
  onSave,
  onNewWorkspace,
  onRenameWorkspace,
}: Props) {
  const [draft, setDraft] = useState(settings);
  const [nameDraft, setNameDraft] = useState(workspaceName);

  useEffect(() => {
    setDraft(settings);
  }, [settings]);

  useEffect(() => {
    setNameDraft(workspaceName);
  }, [workspaceName]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <div className="settings-dialog" role="dialog" aria-modal="true" onMouseDown={(event) => event.stopPropagation()}>
        <h2>连接设置</h2>
        <section className="settings-section">
          <div>
            <strong>知识库工作区</strong>
            <p>当前前端工作区名称会显示在顶部栏。</p>
          </div>
          <label>
            名称
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
        <label>
          API Base URL
          <input value={draft.apiBaseUrl} onChange={(event) => setDraft({ ...draft, apiBaseUrl: event.target.value })} />
        </label>
        <label>
          Token
          <input value={draft.token} onChange={(event) => setDraft({ ...draft, token: event.target.value })} />
        </label>
        <label>
          Tenant
          <input value={draft.tenantId} onChange={(event) => setDraft({ ...draft, tenantId: event.target.value })} />
        </label>
        <label>
          ACL Groups
          <input
            value={draft.aclGroups.join(",")}
            onChange={(event) =>
              setDraft({
                ...draft,
                aclGroups: event.target.value.split(",").map((item) => item.trim()).filter(Boolean),
              })
            }
          />
        </label>
        <div className="dialog-actions">
          <button type="button" onClick={onClose}>
            取消
          </button>
          <button
            type="button"
            className="primary-pill"
            onClick={() => {
              onSave(draft);
              onClose();
            }}
          >
            保存
          </button>
        </div>
      </div>
    </div>
  );
}
