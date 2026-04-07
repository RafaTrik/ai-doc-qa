import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/upload':          'http://localhost:8000',
      '/ask':             'http://localhost:8000',
      '/suggest':         'http://localhost:8000',
      '/summary':         'http://localhost:8000',
      '/compare':         'http://localhost:8000',
      '/contradictions':  'http://localhost:8000',
      '/health':          'http://localhost:8000',
    },
  },
})
