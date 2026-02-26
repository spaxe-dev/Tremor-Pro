/** @type {import('tailwindcss').Config} */
export default {
    content: [
        './index.html',
        './src/**/*.{js,ts}',
        './dashboard.js'
    ],
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', 'system-ui', 'sans-serif'],
                grotesk: ['"Space Grotesk"', 'sans-serif'],
                mono: ['"JetBrains Mono"', 'monospace'],
            },
            colors: {
                base: '#060610',
                teal: {
                    DEFAULT: '#2dd4bf',
                    dim: 'rgba(45,212,191,0.12)',
                    glow: 'rgba(45,212,191,0.3)',
                },
                violet: {
                    DEFAULT: '#a78bfa',
                    deep: '#8b5cf6',
                    dim: 'rgba(167,139,250,0.12)',
                    glow: 'rgba(167,139,250,0.3)',
                },
                emerald: {
                    DEFAULT: '#34d399',
                },
                rose: {
                    DEFAULT: '#fb7185',
                    dim: 'rgba(251,113,133,0.12)',
                },
                amber: {
                    DEFAULT: '#fbbf24',
                    dim: 'rgba(251,191,36,0.12)',
                },
            },
            boxShadow: {
                teal: '0 0 24px rgba(45,212,191,0.25)',
                violet: '0 0 24px rgba(139,92,246,0.3)',
                rose: '0 0 20px rgba(251,113,133,0.25)',
                amber: '0 0 20px rgba(251,191,36,0.25)',
                green: '0 0 20px rgba(52,211,153,0.25)',
            },
            animation: {
                'spin-slow': 'spin 3s linear infinite',
                'pulse-slow': 'pulse 3s ease-in-out infinite',
            }
        }
    },
    plugins: []
}
