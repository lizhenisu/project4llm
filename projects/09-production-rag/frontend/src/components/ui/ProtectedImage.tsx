import { useEffect, useState } from "react";
import type { ImgHTMLAttributes } from "react";

type Props = Omit<ImgHTMLAttributes<HTMLImageElement>, "src"> & {
  src: string;
  token?: string;
  apiBaseUrl?: string;
};

export function ProtectedImage({ src, token, apiBaseUrl, ...props }: Props) {
  const [resolvedSrc, setResolvedSrc] = useState(() =>
    token && isProtectedAssetUrl(src, apiBaseUrl) ? "" : src,
  );

  useEffect(() => {
    if (!token || !isProtectedAssetUrl(src, apiBaseUrl)) {
      setResolvedSrc(src);
      return;
    }
    const controller = new AbortController();
    let objectUrl = "";
    setResolvedSrc("");
    void fetch(src, {
      headers: { Authorization: `Bearer ${token}` },
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Asset request failed with HTTP ${response.status}`);
        }
        return response.blob();
      })
      .then((blob) => {
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        setResolvedSrc(objectUrl);
      })
      .catch(() => {
        if (!controller.signal.aborted) setResolvedSrc("");
      });
    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [apiBaseUrl, src, token]);

  return resolvedSrc ? <img {...props} src={resolvedSrc} /> : null;
}

function isProtectedAssetUrl(value: string, apiBaseUrl?: string) {
  try {
    const url = new URL(value, window.location.origin);
    const allowedOrigin = new URL(apiBaseUrl || window.location.origin, window.location.origin).origin;
    if (url.origin !== allowedOrigin) return false;
    const path = url.pathname;
    return path.startsWith("/source-assets/") || path.startsWith("/api/source-assets/");
  } catch {
    return false;
  }
}
