/**
 * Sends a log message to the Vite dev server, which prints it in the terminal.
 * Falls back silently in production builds.
 */
export function tlog(tag: string, message: string, data?: Record<string, unknown>) {
  if (import.meta.env.PROD) return;
  const body = JSON.stringify({ tag, message, data });
  navigator.sendBeacon('/__terminal_log', body);
}
