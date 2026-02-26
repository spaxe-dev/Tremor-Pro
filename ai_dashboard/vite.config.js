import { defineConfig } from 'vite'

export default defineConfig({
    // Allow index.html at the root to be the entry point
    root: '.',
    server: {
        port: 5173,
        open: true
    },
    build: {
        outDir: 'dist'
    }
})
