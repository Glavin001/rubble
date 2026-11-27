import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Frame, Region, StructureSchema } from '../shared/model.ts';
import { getState, tool, type FrameDoc, type State } from './api.ts';
import { Annotate, nextElement, type ViewBox } from './Annotate.tsx';
import { Compare } from './Compare.tsx';
import { Help } from './Help.tsx';
import { Library } from './Library.tsx';
import { Link } from './Link.tsx';
import { installStudio } from './studio.ts';

export type ViewName = 'library' | 'annotate' | 'link' | 'compare';
const VIEWS: { id: ViewName; label: string; key: string }[] = [
  { id: 'library', label: 'Library', key: '1' },
  { id: 'annotate', label: 'Annotate', key: '2' },
  { id: 'link', label: 'Link', key: '3' },
  { id: 'compare', label: 'Compare', key: '4' },
];

export interface Ctx {
  state: State;
  frame: FrameDoc | null;
  region: Region | null;
  schema: StructureSchema | null;
  element: string | null;
  setRegion: (id: string | null) => void;
  setElement: (name: string | null) => void;
  /** Run a server tool that changes the open frame: undo snapshot, then reload. */
  edit: (name: string, input: Record<string, unknown>, opts?: { quiet?: boolean; noUndo?: boolean }) => Promise<unknown>;
  /** Run any server tool, then reload the state. */
  act: (name: string, input: Record<string, unknown>) => Promise<unknown>;
  open: (view: ViewName, frame?: string | null) => void;
  /** The open frame changed locally (a drag in progress), not yet saved. */
  draft: (f: FrameDoc) => void;
  snapshot: () => void;
  toast: (msg: string, bad?: boolean) => void;
  revision: number;
}

function parseHash(): { view: ViewName; frame: string | null } {
  const [, v, f] = location.hash.split('/');
  const view = (VIEWS.find((x) => x.id === v)?.id ?? 'library') as ViewName;
  return { view, frame: f ? decodeURIComponent(f) : null };
}

function schemaOfRegion(state: State | null, r: Region | null): StructureSchema | null {
  const id = r?.tracing.instances[0]?.schema;
  return (id && state?.schemas.find((s) => s.id === id)) || null;
}

export function App() {
  const [state, setState] = useState<State | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [route, setRoute] = useState(parseHash);
  const [frame, setFrame] = useState<FrameDoc | null>(null);
  const [regionId, setRegionId] = useState<string | null>(null);
  const [element, setElement] = useState<string | null>(null);
  const [viewBox, setViewBox] = useState<ViewBox | null>(null);
  const [toastMsg, setToast] = useState<{ msg: string; bad: boolean } | null>(null);
  const [help, setHelp] = useState(false);
  const [theme, setTheme] = useState<string>(() => {
    try {
      return localStorage.getItem('studio-theme') ?? 'auto';
    } catch {
      return 'auto';
    }
  });
  const [revision, setRevision] = useState(0);
  const undo = useRef<Frame[]>([]);
  const redo = useRef<Frame[]>([]);

  const toast = useCallback((msg: string, bad = false) => {
    setToast({ msg, bad });
    window.setTimeout(() => setToast((t) => (t?.msg === msg ? null : t)), bad ? 6000 : 2500);
  }, []);

  useEffect(() => {
    if (theme === 'auto') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('studio-theme', theme);
    } catch {
      /* private window */
    }
  }, [theme]);

  const reloadState = useCallback(async () => {
    try {
      setState(await getState());
      setLoadError(null);
    } catch (e) {
      setLoadError((e as Error).message);
    }
  }, []);

  const reloadFrame = useCallback(async (id: string | null) => {
    if (!id) {
      setFrame(null);
      return null;
    }
    try {
      const f = await tool<FrameDoc>('get_frame', { frame: id });
      setFrame(f);
      setRevision((r) => r + 1);
      return f;
    } catch (e) {
      toast((e as Error).message, true);
      setFrame(null);
      return null;
    }
  }, [toast]);

  useEffect(() => {
    reloadState();
    const on = () => setRoute(parseHash());
    addEventListener('hashchange', on);
    return () => removeEventListener('hashchange', on);
  }, [reloadState]);

  useEffect(() => {
    undo.current = [];
    redo.current = [];
    reloadFrame(route.frame).then((f) => {
      setViewBox(null);
      const r = f?.regions[0] ?? null;
      setRegionId(r?.id ?? null);
      const sc = schemaOfRegion(stateRef.current, r);
      setElement(sc && r ? nextElement(sc, r.tracing.instances[0], null) : null);
    });
  }, [route.frame, reloadFrame]);

  const open = useCallback((view: ViewName, f?: string | null) => {
    const id = f === undefined ? route.frame : f;
    location.hash = `#/${view}${id ? `/${encodeURIComponent(id)}` : ''}`;
  }, [route.frame]);

  const region = frame?.regions.find((r) => r.id === regionId) ?? null;
  const schema = schemaOfRegion(state, region);

  const snapshot = useCallback(() => {
    if (!frameRef.current) return;
    undo.current.push(structuredClone(frameRef.current));
    if (undo.current.length > 200) undo.current.shift();
    redo.current = [];
  }, []);

  const edit = useCallback(async (name: string, input: Record<string, unknown>, opts: { quiet?: boolean; noUndo?: boolean } = {}) => {
    if (!opts.noUndo) snapshot();
    try {
      const r = await tool(name, input);
      await reloadFrame(input.frame as string);
      if (!opts.quiet) reloadState();
      return r;
    } catch (e) {
      if (!opts.noUndo) undo.current.pop();
      toast((e as Error).message, true);
      await reloadFrame(input.frame as string);
      throw e;
    }
  }, [snapshot, reloadFrame, reloadState, toast]);

  const act = useCallback(async (name: string, input: Record<string, unknown>) => {
    try {
      const r = await tool(name, input);
      await reloadState();
      if (frameRef.current) await reloadFrame(frameRef.current.id);
      return r;
    } catch (e) {
      toast((e as Error).message, true);
      throw e;
    }
  }, [reloadState, reloadFrame, toast]);

  const undoRedo = useCallback(async (dir: 'undo' | 'redo') => {
    const from = dir === 'undo' ? undo.current : redo.current;
    const to = dir === 'undo' ? redo.current : undo.current;
    const doc = from.pop();
    if (!doc || !frameRef.current) return toast(`nothing to ${dir}`);
    to.push(structuredClone(frameRef.current));
    await tool('put_frame', { frame: doc });
    await reloadFrame(doc.id);
    reloadState();
    toast(dir === 'undo' ? 'undone' : 'redone');
  }, [reloadFrame, reloadState, toast]);

  // Latest values for the page tools (window.studio / WebMCP) and keyboard handlers.
  const stateRef = useRef(state);
  const frameRef = useRef(frame);
  const live = useRef({ route, regionId, element, viewBox, schema, region });
  stateRef.current = state;
  frameRef.current = frame;
  live.current = { route, regionId, element, viewBox, schema, region };

  useEffect(() => {
    installStudio(
      {
        open_view: (i) => {
          const v = String(i.view) as ViewName;
          if (!VIEWS.some((x) => x.id === v)) throw new Error('view is library, annotate, link or compare');
          open(v, typeof i.frame === 'string' ? i.frame : undefined);
          return { view: v, frame: i.frame ?? live.current.route.frame };
        },
        open_frame: async (i) => {
          open('annotate', String(i.frame));
          if (typeof i.region === 'string') setRegionId(i.region);
          return { view: 'annotate', frame: i.frame };
        },
        select_element: (i) => {
          if (typeof i.region === 'string') setRegionId(i.region);
          setElement(String(i.element));
          return { element: i.element };
        },
        set_zoom: (i) => {
          window.dispatchEvent(new CustomEvent('studio-zoom', { detail: i }));
          return { zoom: i.zoom };
        },
        describe_view: () => {
          const l = live.current;
          const f = frameRef.current;
          const inst = l.region?.tracing.instances[0];
          return {
            view: l.route.view,
            frame: f ? { id: f.id, shot: f.shot, moment: f.moment.label, view: f.view, image: f.image?.file ?? null, regions: f.regions.map((r) => ({ id: r.id, source: r.source, schema: r.tracing.instances[0]?.schema ?? null })) } : null,
            region: l.regionId,
            schema: l.schema?.id ?? null,
            element: l.element,
            next_element: l.schema && inst ? nextElement(l.schema, inst, l.element) : null,
            zoom: l.viewBox ? { scale: l.viewBox.s, fit_scale: l.viewBox.fit, pan: [l.viewBox.x, l.viewBox.y] } : null,
            shots: stateRef.current?.shots.length ?? 0,
            frames: stateRef.current?.frames.length ?? 0,
          };
        },
      },
      (_name, _r) => {
        reloadState();
        if (frameRef.current) reloadFrame(frameRef.current.id);
      },
    );
  }, [open, reloadState, reloadFrame]);

  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest('input, textarea, select, [contenteditable]')) return;
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        undoRedo(e.shiftKey ? 'redo' : 'undo');
      } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        undoRedo('redo');
      } else if (e.altKey && /^[1-4]$/.test(e.key)) {
        e.preventDefault();
        const v = VIEWS[Number(e.key) - 1];
        if (v.id === 'library' || live.current.route.frame) open(v.id);
      } else if (e.key === '?') setHelp((h) => !h);
    };
    addEventListener('keydown', on);
    return () => removeEventListener('keydown', on);
  }, [undoRedo, open]);

  const ctx: Ctx | null = useMemo(
    () =>
      state && {
        state,
        frame,
        region,
        schema,
        element,
        setRegion: (id) => {
          setRegionId(id);
          const r = frameRef.current?.regions.find((x) => x.id === id) ?? null;
          const sc = schemaOfRegion(stateRef.current, r);
          setElement(sc && r ? nextElement(sc, r.tracing.instances[0], null) : null);
        },
        setElement,
        edit,
        act,
        open,
        draft: setFrame,
        snapshot,
        toast,
        revision,
      },
    [state, frame, region, schema, element, edit, act, open, snapshot, toast, revision],
  );

  const shot = frame?.shot ? state?.shots.find((s) => s.id === frame.shot) : null;
  return (
    <div className="app">
      <header className="topbar">
        <h1>Reference Studio</h1>
        <nav className="tabs" aria-label="Views">
          {VIEWS.map((v) => (
            <button
              key={v.id}
              className="tab"
              aria-current={route.view === v.id ? 'page' : undefined}
              disabled={v.id !== 'library' && !route.frame}
              title={`${v.label} (Alt+${v.key})${v.id !== 'library' && !route.frame ? ': open a frame from the library first' : ''}`}
              onClick={() => open(v.id)}
            >
              {v.label}
            </button>
          ))}
        </nav>
        {frame && (
          <span className="crumb" title={frame.id}>
            {shot ? `${shot.title} · ` : ''}
            {frame.id}
          </span>
        )}
        <span className="spacer" />
        <button className="btn" onClick={() => setHelp(true)} aria-haspopup="dialog">
          How this works <kbd>?</kbd>
        </button>
        <label className="row small">
          <span className="sr-only">Theme</span>
          <select value={theme} onChange={(e) => setTheme(e.target.value)} aria-label="Theme">
            <option value="auto">Theme: system</option>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </label>
      </header>
      <main className="main">
        {loadError && (
          <div className="page">
            <div className="notice bad">
              The studio's server did not answer ({loadError}). Start it with <code>npm run dev</code> in <code>apps/reference-studio</code>.
            </div>
          </div>
        )}
        {ctx && route.view === 'library' && <Library ctx={ctx} />}
        {ctx && route.view !== 'library' && !frame && (
          <div className="page">
            <div className="notice">Loading the frame, or it does not exist. <button className="btn small" onClick={() => open('library', null)}>Back to the library</button></div>
          </div>
        )}
        {ctx && frame && route.view === 'annotate' && <Annotate ctx={ctx} frame={frame} viewBox={viewBox} setViewBox={setViewBox} />}
        {ctx && frame && route.view === 'link' && <Link ctx={ctx} frame={frame} />}
        {ctx && frame && route.view === 'compare' && <Compare ctx={ctx} frame={frame} />}
      </main>
      {help && <Help onClose={() => setHelp(false)} />}
      {toastMsg && (
        <div className={`toast${toastMsg.bad ? ' bad' : ''}`} role="status" aria-live="polite">
          {toastMsg.msg}
        </div>
      )}
    </div>
  );
}
