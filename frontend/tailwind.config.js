/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        // "Case ledger" identity — deep ink navy body, warm paper panels,
        // brass/amber for evidence & highlighting, muted stamp inks for outcomes.
        ink: {
          950: "#0b1220",
          900: "#101a2c",
          800: "#182338",
          700: "#212f49",
          600: "#334463",
          faint: "#7385a6",
        },
        paper: {
          DEFAULT: "#f3ecda",
          dim: "#e8dfc7",
          line: "#d8cba7",
        },
        amber: {
          DEFAULT: "#b8863b",
          bright: "#d6a355",
          dim: "#8a6529",
        },
        stamp: {
          green: "#3f6b4a",
          red: "#8b3a3a",
          navy: "#26405f",
        },
      },
      fontFamily: {
        display: ["'IBM Plex Serif'", "Georgia", "serif"],
        body: ["'IBM Plex Sans'", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        ledger: "0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 24px rgba(5,9,18,0.35)",
      },
      backgroundImage: {
        grain: "radial-gradient(rgba(255,255,255,0.025) 1px, transparent 1px)",
      },
      backgroundSize: {
        grain: "3px 3px",
      },
    },
  },
  plugins: [],
};
