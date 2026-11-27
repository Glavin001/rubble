// One tool list (shared/tools.ts), three routes to the same functions:
// - window.studio: studio.tools, studio.call(name, input), studio.<name>(input) (JSON in, JSON out);
// - WebMCP: each tool registered with document.modelContext.registerTool (WebMCP spec,
//   "4.2 ModelContext Interface", "4.2.1 ModelContextTool Dictionary"); the W3C polyfill
//   (@w3c/webmcp-polyfill, pinned and bundled, no network) installs document.modelContext
//   where the browser has none;
// - HTTP: the server tools are POST /api/tool/<name> (server/api.ts).

import { installWebMCP } from '@w3c/webmcp-polyfill';
import { TOOLS, type ToolDef } from '../shared/tools.ts';
import { tool } from './api.ts';

export type PageTools = Record<string, (input: Record<string, unknown>) => unknown | Promise<unknown>>;

export interface Studio {
  tools: ToolDef[];
  call: (name: string, input?: Record<string, unknown>) => Promise<unknown>;
  /** How the tools reach WebMCP: the browser's own document.modelContext, the polyfill's, or none. */
  webmcp: 'native' | 'polyfill' | 'none';
  [name: string]: unknown;
}

interface ModelContextLike {
  registerTool: (tool: Record<string, unknown>, options?: { signal?: AbortSignal }) => Promise<void> | void;
}

declare global {
  interface Window {
    studio: Studio;
  }
}

// The page's current handlers: React re-renders replace them; the tools are registered once.
let handlers: { page: PageTools; after: (name: string, result: unknown) => void } | null = null;

export function installStudio(page: PageTools, afterServerWrite: (name: string, result: unknown) => void): Studio {
  handlers = { page, after: afterServerWrite };
  if (window.studio) return window.studio;
  const call = async (name: string, input: Record<string, unknown> = {}) => {
    const t = TOOLS.find((x) => x.name === name);
    if (!t) throw new Error(`no tool ${name}; studio.tools lists them`);
    if (t.side === 'page') return handlers!.page[name](input ?? {});
    const r = await tool(name, input ?? {});
    if (!t.readOnly) handlers!.after(name, r);
    return r;
  };
  const studio: Studio = { tools: TOOLS, call, webmcp: 'none' };
  for (const t of TOOLS) studio[t.name] = (input: Record<string, unknown> = {}) => call(t.name, input);
  window.studio = studio;

  // WebMCP: the spec's document.modelContext; a browser without it gets the W3C polyfill.
  const hadNative = 'modelContext' in document;
  try {
    if (!hadNative) installWebMCP();
  } catch (e) {
    console.warn('WebMCP polyfill not installed:', e);
  }
  const mc = ((document as unknown as { modelContext?: ModelContextLike }).modelContext ??
    (navigator as unknown as { modelContext?: ModelContextLike }).modelContext) as ModelContextLike | undefined;
  if (mc?.registerTool) {
    studio.webmcp = hadNative ? 'native' : 'polyfill';
    const controller = new AbortController();
    for (const t of TOOLS) {
      Promise.resolve(
        mc.registerTool(
          {
            name: t.name,
            title: t.title,
            description: t.description,
            inputSchema: t.inputSchema,
            annotations: { readOnlyHint: t.readOnly },
            execute: (input: Record<string, unknown>) => call(t.name, input ?? {}),
          },
          { signal: controller.signal },
        ),
      ).catch((e) => console.warn(`WebMCP: ${t.name} not registered:`, e));
    }
    addEventListener('pagehide', () => controller.abort());
  }
  return studio;
}
