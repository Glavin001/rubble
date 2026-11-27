// The first-run screen: choose the workspace folder (or use this browser's private storage).

import { useEffect, useState } from 'react';
import type { Workspace } from '../shared/workspace.ts';
import { canPickFolder, hasOpfs, opfs, pickFolder, regrant, remembered } from './workspace.ts';

export function WorkspaceGate({ onOpen }: { onOpen: (ws: Workspace) => void }) {
  const [last, setLast] = useState<Awaited<ReturnType<typeof remembered>>>(null);
  const [err, setErr] = useState<string | null>(null);
  const [checked, setChecked] = useState(false);
  useEffect(() => {
    const q = new URLSearchParams(location.search);
    // ?workspace=opfs: straight to this browser's storage (the automated smoke test).
    if (q.get('workspace') === 'opfs' && hasOpfs()) {
      opfs().then(onOpen, (e) => setErr(String(e?.message ?? e)));
      return;
    }
    remembered().then((r) => {
      if (r?.granted) regrant(r.handle).then((ws) => (ws ? onOpen(ws) : setLast(r)));
      else setLast(r);
      setChecked(true);
    });
  }, [onOpen]);
  const run = (f: () => Promise<Workspace | null>) => f().then((ws) => ws && onOpen(ws), (e) => e?.name !== 'AbortError' && setErr(String(e?.message ?? e)));
  if (!checked && !err) return <div className="page"><p className="muted">Opening the workspace…</p></div>;
  return (
    <div className="page gate">
      <h2>Choose your workspace folder</h2>
      <p>
        The studio keeps everything in one folder: <code>shots/</code>, <code>frames/</code>, <code>structures/</code> and <code>scenarios/</code> (points, text and
        numbers you can commit), <code>library/</code> (the images) and <code>recordings/</code> (cached runs). In the Rubble repo that folder is{' '}
        <code>apps/reference-studio/workspace/</code>.
      </p>
      <div className="row wrap">
        {last && (
          <button className="btn primary" onClick={() => run(() => regrant(last.handle))}>
            Open “{last.handle.name}” again
          </button>
        )}
        {canPickFolder() ? (
          <button className={`btn${last ? '' : ' primary'}`} onClick={() => run(pickFolder)}>Choose a folder…</button>
        ) : (
          <span className="notice">This browser cannot open a folder (the File System Access API's directory picker is in Chromium browsers: Chrome, Edge).</span>
        )}
        {hasOpfs() && <button className="btn" onClick={() => run(opfs)}>Use this browser's storage</button>}
      </div>
      <p className="muted small">“This browser's storage” is the origin private file system: the data then lives only in this browser, not in a folder you can commit.</p>
      {err && <div className="notice bad">{err}</div>}
    </div>
  );
}
