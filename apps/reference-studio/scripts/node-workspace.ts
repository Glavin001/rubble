// A workspace folder on disk for node scripts and tests (shared/workspace.ts's interface; the
// page uses the File System API instead).

import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { checkPath, type Workspace } from '../shared/workspace.ts';

export const PACKAGE = resolve(dirname(fileURLToPath(import.meta.url)), '..');
export const REPO = resolve(PACKAGE, '..', '..');
export const ENGINE = join(REPO, 'engine');
/** The repo's workspace folder (committed data, gitignored library and recordings). */
export const WORKSPACE = join(PACKAGE, 'workspace');

export class DiskWorkspace implements Workspace {
  kind = 'disk' as const;
  name: string;
  root: string;
  constructor(root: string) {
    this.root = resolve(root);
    this.name = root;
  }
  private p(path: string) {
    return join(this.root, ...checkPath(path));
  }
  private entries(dir: string, dirs: boolean) {
    const d = dir ? this.p(dir) : this.root;
    if (!existsSync(d)) return [];
    return readdirSync(d).filter((f) => !f.startsWith('.') && statSync(join(d, f)).isDirectory() === dirs).sort();
  }
  async list(dir: string) {
    return this.entries(dir, false);
  }
  async dirs(dir: string) {
    return this.entries(dir, true);
  }
  async read(path: string) {
    const p = this.p(path);
    return existsSync(p) ? new Uint8Array(readFileSync(p)) : null;
  }
  async write(path: string, data: Uint8Array | string) {
    const p = this.p(path);
    mkdirSync(dirname(p), { recursive: true });
    const tmp = `${p}.tmp-${process.pid}`;
    writeFileSync(tmp, data);
    renameSync(tmp, p);
  }
  async remove(path: string) {
    rmSync(this.p(path), { force: true });
  }
}
