import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    allowedHosts: [
      '.ngrok-free.dev',
      '.ngrok.free.dev',
      '.ngrok-free.app',
      '.ngrok.app',
      '.ngrok.io',
      'kent-persevering-willingly.ngrok-free.dev',
      '.trycloudflare.com', // Cloudflare quick tunnels (Kaggle demo notebook)
    ],
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/generated': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/uploads': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/static': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  preview: {
    allowedHosts: [
      '.ngrok-free.dev',
      '.ngrok.free.dev',
      '.ngrok-free.app',
      '.ngrok.app',
      '.ngrok.io',
      '.trycloudflare.com', // Cloudflare quick tunnels (Kaggle demo notebook)
    ],
  },
});
