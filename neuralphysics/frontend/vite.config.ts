import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

// Default build: dist/, served by FastAPI (in development the API runs separately on port 8000).
// Demo build (`--mode demo`): one self-contained dist-demo/index.html that replays recorded runs.
export default defineConfig(({ mode }) => {
  const demo = mode === 'demo';
  return {
    base: demo ? './' : '/',
    publicDir: false, // public/demo holds the recordings; the demo build embeds them instead of copying
    plugins: [react(), tailwindcss(), ...(demo ? [viteSingleFile()] : [])],
    server: {
      proxy: {
        '/api': 'http://127.0.0.1:8000',
        '/ws': { target: 'ws://127.0.0.1:8000', ws: true },
      },
    },
    build: {
      outDir: demo ? 'dist-demo' : 'dist',
      chunkSizeWarningLimit: 20000,
      ...(demo ? { assetsInlineLimit: 100_000_000 } : {}),
    },
  };
});
