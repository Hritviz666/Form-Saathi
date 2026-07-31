/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        saffron: { 50: '#fff8f0', 100: '#fff0d9', 200: '#ffd99e', 300: '#ffbc5c', 400: '#ff9f1c', 500: '#f07c00', 600: '#c45f00', 700: '#9a4800', 800: '#7a3800', 900: '#5c2900' },
        indigo: { 50: '#f0f4ff', 100: '#dde6ff', 200: '#b8c9ff', 300: '#85a0ff', 400: '#4d6fff', 500: '#1a3fff', 600: '#0028e6', 700: '#001db8', 800: '#001590', 900: '#000e6b' },
        forest: { 50: '#f0faf4', 100: '#d8f5e4', 200: '#a8e8c4', 300: '#6dd4a0', 400: '#32bb7c', 500: '#139e60', 600: '#0b7d4a', 700: '#095e37', 800: '#074529', 900: '#04301c' },
      },
      fontFamily: {
        display: ['"Baloo 2"', 'sans-serif'],
        body: ['"Noto Sans"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      animation: {
        'slide-up': 'slideUp 0.3s ease-out',
        'fade-in': 'fadeIn 0.4s ease-out',
        'pulse-ring': 'pulseRing 1.5s ease-out infinite',
        'dot-bounce': 'dotBounce 1.2s ease-in-out infinite',
      },
      keyframes: {
        slideUp: { '0%': { transform: 'translateY(12px)', opacity: 0 }, '100%': { transform: 'translateY(0)', opacity: 1 } },
        fadeIn: { '0%': { opacity: 0 }, '100%': { opacity: 1 } },
        pulseRing: { '0%': { transform: 'scale(1)', opacity: 0.5 }, '100%': { transform: 'scale(1.6)', opacity: 0 } },
        dotBounce: { '0%,80%,100%': { transform: 'translateY(0)' }, '40%': { transform: 'translateY(-6px)' } },
      },
    },
  },
  plugins: [],
}
