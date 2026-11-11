import { defineConfig } from 'vite';

export default defineConfig({
  // WebGPU only exists in recent browsers; esnext also allows the
  // top-level await in main.ts.
  build: { target: 'esnext' },
  // localhost is a secure context, so navigator.gpu works without HTTPS.
  server: { port: 5173 },
});
