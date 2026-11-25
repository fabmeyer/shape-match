import { execSync } from 'node:child_process';
import { defineConfig } from 'vite';

// the git commit the app was built from, for the debug report ("unknown" outside a git checkout)
function appVersion(): string {
  try {
    const commit = execSync('git rev-parse --short HEAD', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
    const dirty = execSync('git status --porcelain', { stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim() !== '';
    return dirty ? `${commit}+changes` : commit;
  } catch {
    return 'unknown';
  }
}

export default defineConfig({
  // WebGPU only exists in recent browsers; esnext also allows the
  // top-level await in main.ts.
  build: { target: 'esnext' },
  // localhost is a secure context, so navigator.gpu works without HTTPS.
  server: { port: 5173 },
  define: { __APP_VERSION__: JSON.stringify(appVersion()) },
});
