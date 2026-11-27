// The studio's storage: one workspace folder holding everything (README "Data model"):
//   shots/, frames/, structures/, scenarios/   points, text and numbers (committable)
//   library/                                   the images (never committed)
//   recordings/                                cached runs, one file per scenario hash (never committed)
//
// Two implementations in the page over the same File System API handle type: a folder the
// user picks (showDirectoryPicker, Chromium) and the origin private file system (OPFS,
// navigator.storage.getDirectory: this browser only). Node scripts and tests use a folder on
// disk (scripts/node-workspace.ts) or memory.

export interface Workspace {
  /** 'folder': a folder the user chose; 'opfs': this browser's private storage; 'disk' or 'memory' outside a page. */
  kind: 'folder' | 'opfs' | 'disk' | 'memory';
  /** What to show for it (the folder's name). */
  name: string;
  /** File names directly in `dir` ('' for the root); none when it does not exist. */
  list(dir: string): Promise<string[]>;
  /** Folder names directly in `dir`. */
  dirs(dir: string): Promise<string[]>;
  read(path: string): Promise<Uint8Array | null>;
  write(path: string, data: Uint8Array | string): Promise<void>;
  remove(path: string): Promise<void>;
}

/** A path inside the workspace: relative, no '..', no empty parts, no backslashes or NUL. */
export function checkPath(path: string): string[] {
  const parts = path.split('/');
  if (!path || path.startsWith('/') || path.includes('\\') || path.includes('\0') || parts.some((p) => p === '' || p === '.' || p === '..')) {
    throw new Error(`path not allowed: ${JSON.stringify(path)}`);
  }
  return parts;
}

const enc = new TextEncoder();

export class MemoryWorkspace implements Workspace {
  kind = 'memory' as const;
  name = 'memory';
  files = new Map<string, Uint8Array>();
  async list(dir: string) {
    const pre = dir ? `${dir}/` : '';
    return [...this.files.keys()].filter((k) => k.startsWith(pre) && !k.slice(pre.length).includes('/')).map((k) => k.slice(pre.length)).sort();
  }
  async dirs(dir: string) {
    const pre = dir ? `${dir}/` : '';
    return [...new Set([...this.files.keys()].filter((k) => k.startsWith(pre) && k.slice(pre.length).includes('/')).map((k) => k.slice(pre.length).split('/')[0]))].sort();
  }
  async read(path: string) {
    checkPath(path);
    return this.files.get(path) ?? null;
  }
  async write(path: string, data: Uint8Array | string) {
    checkPath(path);
    this.files.set(path, typeof data === 'string' ? enc.encode(data) : data);
  }
  async remove(path: string) {
    this.files.delete(path);
  }
}

// The File System API's handle types, as far as the studio uses them (lib.dom has most of
// them; entries() and the permission calls are not in every TypeScript lib yet).
interface DirHandle {
  kind: 'directory';
  name: string;
  getDirectoryHandle(name: string, o?: { create?: boolean }): Promise<DirHandle>;
  getFileHandle(name: string, o?: { create?: boolean }): Promise<FileHandleLike>;
  removeEntry(name: string): Promise<void>;
  entries(): AsyncIterable<[string, { kind: 'file' | 'directory' }]>;
}
interface FileHandleLike {
  getFile(): Promise<Blob>;
  createWritable(): Promise<{ write(d: BufferSource | Blob | string): Promise<void>; close(): Promise<void> }>;
}

/** A workspace over a directory handle: a folder the user picked, or the OPFS root. */
export class HandleWorkspace implements Workspace {
  kind: 'folder' | 'opfs';
  name: string;
  root: DirHandle;
  constructor(root: unknown, kind: 'folder' | 'opfs') {
    this.root = root as DirHandle;
    this.kind = kind;
    this.name = kind === 'opfs' ? 'this browser (private storage)' : this.root.name;
  }
  private async dir(parts: string[], create: boolean): Promise<DirHandle | null> {
    let d = this.root;
    for (const p of parts) {
      try {
        d = await d.getDirectoryHandle(p, { create });
      } catch {
        return null;
      }
    }
    return d;
  }
  private async entries(dir: string, kind: 'file' | 'directory') {
    const d = await this.dir(dir ? checkPath(dir) : [], false);
    if (!d) return [];
    const out: string[] = [];
    for await (const [name, h] of d.entries()) if (h.kind === kind && !name.startsWith('.')) out.push(name);
    return out.sort();
  }
  list(dir: string) {
    return this.entries(dir, 'file');
  }
  dirs(dir: string) {
    return this.entries(dir, 'directory');
  }
  async read(path: string) {
    const parts = checkPath(path);
    const d = await this.dir(parts.slice(0, -1), false);
    if (!d) return null;
    try {
      const f = await (await d.getFileHandle(parts[parts.length - 1])).getFile();
      return new Uint8Array(await f.arrayBuffer());
    } catch {
      return null;
    }
  }
  async write(path: string, data: Uint8Array | string) {
    const parts = checkPath(path);
    const d = await this.dir(parts.slice(0, -1), true);
    if (!d) throw new Error(`cannot create the folder of ${path}`);
    const w = await (await d.getFileHandle(parts[parts.length - 1], { create: true })).createWritable();
    await w.write(typeof data === 'string' ? data : (data as Uint8Array<ArrayBuffer>));
    await w.close();
  }
  async remove(path: string) {
    const parts = checkPath(path);
    const d = await this.dir(parts.slice(0, -1), false);
    if (d) await d.removeEntry(parts[parts.length - 1]).catch(() => undefined);
  }
}
