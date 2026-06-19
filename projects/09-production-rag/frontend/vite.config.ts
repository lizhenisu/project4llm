import react from "@vitejs/plugin-react";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { defineConfig } from "vite";
import type { Plugin } from "vite";

function appVersion() {
  const candidates = [resolve(__dirname, "../VERSION"), resolve(__dirname, "VERSION")];
  const versionPath = candidates.find((path) => existsSync(path));
  if (!versionPath) return "0.0.0";
  return readFileSync(versionPath, "utf-8").trim() || "0.0.0";
}

function markdownUtf8Plugin(): Plugin {
  return {
    name: "markdown-utf8",
    configureServer(server) {
      const publicDir = resolve(__dirname, "public");
      server.middlewares.use(async (req, res, next) => {
        const url = new URL(req.url || "/", "http://localhost");
        const requestPath = decodeURIComponent(url.pathname);
        if (requestPath === "/PROJECT_EVALUATION.md" && !url.searchParams.has("raw")) {
          const htmlPath = resolve(__dirname, "index.html");
          const html = readFileSync(htmlPath, "utf-8");
          res.statusCode = 200;
          res.setHeader("Content-Type", "text/html; charset=utf-8");
          res.end(await server.transformIndexHtml(requestPath, html));
          return;
        }
        if (!requestPath.endsWith(".md")) {
          next();
          return;
        }
        if (!url.searchParams.has("raw")) {
          next();
          return;
        }
        const relativePath = requestPath.replace(/^\/+/, "");
        const filePath = resolve(publicDir, relativePath);
        if (!filePath.startsWith(publicDir) || !existsSync(filePath)) {
          next();
          return;
        }
        res.statusCode = 200;
        res.setHeader("Content-Type", "text/markdown; charset=utf-8");
        res.end(readFileSync(filePath, "utf-8"));
      });
    },
  };
}

export default defineConfig({
  plugins: [markdownUtf8Plugin(), react()],
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
