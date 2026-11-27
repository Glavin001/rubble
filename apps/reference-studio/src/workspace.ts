// Choosing the workspace folder: a folder the user picks (File System Access API,
// showDirectoryPicker: Chromium browsers), its handle remembered in IndexedDB and its
// read-write permission asked again on return; or this browser's origin private file system
// (OPFS) when no folder is chosen or the API is absent (the data then lives only in this
// browser). A new workspace gets the shipped structure schemas.

import type { StructureSchema } from '../shared/model.ts';
import { HandleWorkspace, type Workspace } from '../shared/workspace.ts';

// The structure schemas shipped with the studio (the repo's workspace/structures/).
const SHIPPED = import.meta.glob('../workspace/structures/*.json', { eager: true, import: 'default' }) as Record<string, StructureSchema>;

const DB = 'reference-studio';
const KEY = 'workspace';

interface PermHandle {
  name: string;
  queryPermission?(o: { mode: 'readwrite' }): Promise<PermissionState>;
  requestPermission?(o: { mode: 'readwrite' }): Promise<PermissionState>;
}

function idb<T>(fn: (s: IDBObjectStore) => IDBRequest<T>, mode: IDBTransactionMode = 'readonly'): Promise<T | undefined> {
  return new Promise((resolve) => {
    let open: IDBOpenDBRequest;
    try {
      open = indexedDB.open(DB, 1);
    } catch {
      return resolve(undefined);
    }
    open.onupgradeneeded = () => open.result.createObjectStore('handles');
    open.onerror = () => resolve(undefined);
    open.onsuccess = () => {
      try {
        const r = fn(open.result.transaction('handles', mode).objectStore('handles'));
        r.onsuccess = () => resolve(r.result);
        r.onerror = () => resolve(undefined);
      } catch {
        resolve(undefined);
      }
    };
  });
}

export const canPickFolder = (): boolean => 'showDirectoryPicker' in window;
export const hasOpfs = (): boolean => !!navigator.storage?.getDirectory;

/** The folder chosen last time, if this browser remembers it: granted (open it at once) or
 * needing a click to grant read-write again. */
export async function remembered(): Promise<{ handle: PermHandle; granted: boolean } | null> {
  const h = (await idb<PermHandle>((s) => s.get(KEY))) ?? null;
  if (!h) return null;
  const p = (await h.queryPermission?.({ mode: 'readwrite' }).catch(() => 'prompt' as PermissionState)) ?? 'prompt';
  return { handle: h, granted: p === 'granted' };
}

/** Ask again for a remembered folder (needs a user gesture). */
export async function regrant(h: PermHandle): Promise<Workspace | null> {
  const p = (await h.requestPermission?.({ mode: 'readwrite' })) ?? 'denied';
  return p === 'granted' ? seeded(new HandleWorkspace(h, 'folder')) : null;
}

/** Choose a folder (needs a user gesture), remembered for next time. */
export async function pickFolder(): Promise<Workspace> {
  const h = await (window as unknown as { showDirectoryPicker(o: unknown): Promise<PermHandle> }).showDirectoryPicker({ id: 'reference-studio', mode: 'readwrite' });
  await idb((s) => s.put(h, KEY), 'readwrite');
  return seeded(new HandleWorkspace(h, 'folder'));
}

/** This browser's private storage. */
export async function opfs(): Promise<Workspace> {
  const root = await navigator.storage.getDirectory();
  return seeded(new HandleWorkspace(root, 'opfs'));
}

export async function forget(): Promise<void> {
  await idb((s) => s.delete(KEY), 'readwrite');
}

/** A workspace with no structure schemas gets the shipped ones. */
async function seeded(ws: Workspace): Promise<Workspace> {
  const have = await ws.list('structures');
  for (const s of Object.values(SHIPPED)) if (!have.includes(`${s.id}.json`)) await ws.write(`structures/${s.id}.json`, JSON.stringify(s, null, 1) + '\n');
  return ws;
}
