import type { ReactNode } from "react";

type Props = {
  label: string;
  children: ReactNode;
  disabled?: boolean;
  onClick?: () => void;
};

export function IconButton({ label, children, disabled, onClick }: Props) {
  return (
    <button className="icon-button" type="button" aria-label={label} title={label} disabled={disabled} onClick={onClick}>
      {children}
    </button>
  );
}
