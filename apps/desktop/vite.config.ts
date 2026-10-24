import path from "node:path";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const host = process.env.TAURI_DEV_HOST;
const root = path.dirname(fileURLToPath(import.meta.url));

// Vite options tailored for Tauri. The fixed port and the src-tauri watch
// ignore are required by `tauri dev` / `tauri build`.
export default defineConfig(() => ({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(root, "./src"),
    },
  },
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // The glue and the binary share function-table indices. Ignore only the
      // binary so a regen reloads the glue; ignoring the folder serves the old
      // glue against the new module (`__wasm_bindgen_func_elem_* is not a function`).
      ignored: ["**/src-tauri/**", "**/src/wasm/**/*.wasm"],
    },
  },
}));
