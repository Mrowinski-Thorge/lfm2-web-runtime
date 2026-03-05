import { defineConfig } from 'vite';

export default defineConfig({
  // GitHub Pages subpath: https://<user>.github.io/lfm2-web-runtime/
  base: '/lfm2-web-runtime/',
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
  server: {
    headers: {
      // Dev server headers only - not needed for production (single-threaded WASM)
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  build: {
    target: 'esnext',
  },
});
