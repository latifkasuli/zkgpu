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
