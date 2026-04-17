import { defineConfig } from "vite";
import { resolve } from "path";

export default defineConfig({
  root: ".",
  publicDir: "public",
  build: {
    outDir: "dist",
    target: "esnext",
    rollupOptions: {
      input: {
        main: resolve(__dirname, "index.html"),
      },
    },
  },
  server: {
    // Allow BrowserStack Local's real-device hostname (iOS/Android
    // tunnel resolves localhost → bs-local.com, which then proxies
    // back to the dev machine's loopback). Vite 6+ rejects unknown
    // Host headers by default.
    allowedHosts: ["bs-local.com"],
    headers: {
      // Required for SharedArrayBuffer (needed by some WebGPU impls)
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  optimizeDeps: {
    exclude: ["zkgpu_web"],
  },
  worker: {
    format: "es",
  },
});
