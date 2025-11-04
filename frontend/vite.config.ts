import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true, // Allow external access
    allowedHosts: [
      'knotty-bee-unlibidinously.ngrok-free.dev',
      '.ngrok-free.dev', // Allow all ngrok subdomains
      '.ngrok.io', // Allow ngrok.io domains too
      '.serveo.net', // Allow serveo.net domains
    ],
    headers: {
      // Required for ONNX Runtime WASM
      // Using require-corp instead of credentialless for better compatibility
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Resource-Policy': 'cross-origin',
    },
    proxy: {
      // Proxy API requests to backend (use /api prefix to avoid conflict with frontend route)
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
})
