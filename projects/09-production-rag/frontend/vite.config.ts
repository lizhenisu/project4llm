import react from "@vitejs/plugin-react";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";

function appVersion() {
  const candidates = [resolve(__dirname, "../VERSION"), resolve(__dirname, "VERSION")];
  const versionPath = candidates.find((path) => existsSync(path));
  if (!versionPath) return "0.0.0";
  return readFileSync(versionPath, "utf-8").trim() || "0.0.0";
}

export default defineConfig({
  plugins: [react()],
  define: {
    "import.meta.env.VITE_APP_VERSION": JSON.stringify(appVersion()),
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.indexOf("@xyflow/react") >= 0 || id.indexOf("@xyflow/system") >= 0) {
            return "mindmap-flow";
          }
          return undefined;
        },
      },
    },
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8008",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
