import { useEffect, useState } from "react";
import type { Settings } from "../lib/types";

type Props = {
  open: boolean;
  settings: Settings;
  onClose: () => void;
  onSave: (settings: Settings) => void;
};

export function SettingsDialog({ open, settings, onClose, onSave }: Props) {
  const [draft, setDraft] = useState(settings);

  useEffect(() => {
    setDraft(settings);
  }, [settings]);

  if (!open) return null;

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <div className="settings-dialog" role="dialog" aria-modal="true" onMouseDown={(event) => event.stopPropagation()}>
        <h2>连接设置</h2>
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
