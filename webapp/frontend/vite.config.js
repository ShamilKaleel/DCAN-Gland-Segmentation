import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies /api to the FastAPI backend so there's no CORS/port pain.
// `base: ""` makes the production build use relative asset paths so FastAPI can
// serve dist/ from any mount point.
export default defineConfig({
  plugins: [react()],
  base: "",
  server: {
    port: 5173,
    // Allow access through ngrok/other tunnels. A leading dot whitelists every
    // subdomain, so the random ngrok-free host you get each session just works.
    allowedHosts: [".ngrok-free.dev", ".ngrok-free.app", ".ngrok.io"],
    proxy: {
      "/api": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
  },
});
