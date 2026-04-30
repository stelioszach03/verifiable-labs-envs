import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx,mdx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Brand: minimal, ink + accent.
        ink: {
          DEFAULT: "#0b0d12",
          soft: "#1a1d24",
          muted: "#5a6172",
        },
        paper: "#fafbfc",
        accent: {
          DEFAULT: "#5d5fef",
          hover: "#4a4cd6",
        },
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
