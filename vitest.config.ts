import { defineConfig, mergeConfig } from 'vitest/config';
import { playwright } from '@vitest/browser-playwright';
import viteConfig from './vite.config.ts';

// two test projects:
//   unit: *.test.ts in Node, no GPU (npm test)
//   gpu:  *.gpu.test.ts in headless Chromium with WebGPU, GPU results against the CPU reference (npm run test:gpu)
export default mergeConfig(viteConfig, defineConfig({
  test: {
    projects: [
      {
        extends: true,
        test: { name: 'unit', include: ['src/**/*.test.ts'], exclude: ['src/**/*.gpu.test.ts'], environment: 'node' },
      },
      {
        extends: true,
        test: {
          name: 'gpu',
          include: ['src/**/*.gpu.test.ts'],
          browser: {
            enabled: true,
            headless: true,
            provider: playwright({ launchOptions: { args: ['--enable-unsafe-webgpu'] } }),
            instances: [{ browser: 'chromium' }],
          },
        },
      },
    ],
  },
}));
