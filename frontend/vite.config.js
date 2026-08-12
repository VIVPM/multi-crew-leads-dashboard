import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ command, mode }) => {
  // Fail the build itself, not just the bundle at runtime in a browser — a
  // silently-missing VITE_BACKEND_URL should never make it to a deploy.
  if (command === 'build') {
    const env = loadEnv(mode, process.cwd(), '')
    if (!env.VITE_BACKEND_URL) {
      throw new Error(
        'VITE_BACKEND_URL is not set. It is required for a production build ' +
        '(no hardcoded fallback — see frontend/src/api.js) — set it in ' +
        'frontend/.env locally, or in the build environment for a real deploy.'
      )
    }
  }
  return {
    plugins: [react()],
  }
})
