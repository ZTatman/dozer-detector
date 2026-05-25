import { defineConfig } from "vite";

export default defineConfig({
  publicDir: "public",
  build: {
    outDir: "dist-modern",
    rollupOptions: {
      input: "modern/index.html",
    },
  },
});
