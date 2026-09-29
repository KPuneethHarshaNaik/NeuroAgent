import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { resolve } from "node:path";

// The FastAPI app serves this directory statically:
//     app.mount("/", StaticFiles(directory="frontend", html=True))
// so the build must land in frontend/ and must never empty it -- frontend/legacy/ holds the
// pre-rewrite vanilla tool until the React tool page is verified in its place.
const frontendDir = resolve(import.meta.dirname, "frontend");

export default defineConfig({
  root: resolve(import.meta.dirname, "web"),
  base: "/",
  appType: "mpa",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": resolve(import.meta.dirname, "web/src") },
  },
  build: {
    outDir: frontendDir,
    emptyOutDir: false,
    rollupOptions: {
      input: {
        // landing -> frontend/index.html (served at /) and tool -> frontend/app/index.html (/app/)
        index: resolve(import.meta.dirname, "web/index.html"),
        app: resolve(import.meta.dirname, "web/app/index.html"),
      },
    },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/health": "http://127.0.0.1:8000",
      "/live-jobs": "http://127.0.0.1:8000",
      "/jobs": "http://127.0.0.1:8000",
      "/internal": "http://127.0.0.1:8000",
    },
  },
});
