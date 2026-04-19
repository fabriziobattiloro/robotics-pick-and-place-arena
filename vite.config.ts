import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// ANSI color codes for terminal output
const colors: Record<string, string> = {
  BOT: '\x1b[38;5;99m',    // indigo
  SIM: '\x1b[38;5;39m',    // sky blue
  ANIMATOR: '\x1b[38;5;205m', // pink
  OK: '\x1b[38;5;42m',     // green
  WARN: '\x1b[38;5;214m',  // orange
  ERR: '\x1b[38;5;196m',   // red
  RESET: '\x1b[0m',
  DIM: '\x1b[2m',
  BOLD: '\x1b[1m',
};

function terminalLogPlugin() {
  return {
    name: 'terminal-log',
    configureServer(server: { middlewares: { use: Function } }) {
      server.middlewares.use('/__terminal_log', (req: any, res: any) => {
        if (req.method !== 'POST') { res.writeHead(404); res.end(); return; }
        let body = '';
        req.on('data', (chunk: string) => { body += chunk; });
        req.on('end', () => {
          try {
            const { tag, message, data } = JSON.parse(body);
            const color = colors[tag] || colors.BOT;
            const time = new Date().toLocaleTimeString();
            const dataStr = data ? ` ${colors.DIM}${JSON.stringify(data)}${colors.RESET}` : '';
            console.log(`${colors.DIM}${time}${colors.RESET} ${color}${colors.BOLD}[${tag}]${colors.RESET} ${message}${dataStr}`);
          } catch { /* ignore malformed */ }
          res.writeHead(200); res.end('ok');
        });
      });
    },
  };
}

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
        proxy: {
          '/anthropic-api': {
            target: 'https://api.anthropic.com',
            changeOrigin: true,
            rewrite: (path: string) => path.replace(/^\/anthropic-api/, ''),
          },
        },
      },
      plugins: [
        terminalLogPlugin(),
        {
          name: 'externalize-mujoco',
          resolveId(id) {
            if (id === 'mujoco_wasm') return { id: 'https://unpkg.com/mujoco-js@0.0.7/dist/mujoco_wasm.js', external: true };
          },
        },
        react(),
      ],
      define: {
        'process.env.GOOGLE_API_KEY': JSON.stringify(env.GOOGLE_API_KEY || env.GEMINI_API_KEY),
        'process.env.ANTHROPIC_API_KEY': JSON.stringify(env.ANTHROPIC_API_KEY || ''),
        'process.env.OPENAI_API_KEY': JSON.stringify(env.OPENAI_API_KEY || ''),
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      },
      optimizeDeps: {
        exclude: ['mujoco_wasm']
      },
      build: {
        rollupOptions: {
          external: ['mujoco_wasm']
        }
      }
    };
});
