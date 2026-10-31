/**
 * Vite + Vitest configuration.
 *
 * The dashboard is served by the FastAPI app at `/` in production, so every
 * request is same-origin: `/api/*` for REST and `/ws/live` for the live feed.
 * In development the same paths are proxied to the backend (or to the mock
 * server, `npm run mock`) on port 8000, so the code never needs a base URL.
 */
import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

const BACKEND = process.env.PUFFER_BACKEND ?? 'http://localhost:8000';

const proxy = {
  '/api': { target: BACKEND, changeOrigin: true },
  '/ws': { target: BACKEND.replace(/^http/, 'ws'), ws: true, changeOrigin: true },
};

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
    target: 'es2022',
  },
  test: {
    environment: 'node',
    include: ['tests/**/*.test.ts'],
  },
});
