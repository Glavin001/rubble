import { useEffect, useRef } from 'react';

export function Help({ onClose }: { onClose: () => void }) {
  const d = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    d.current?.showModal();
  }, []);
  return (
    <dialog ref={d} onClose={onClose} aria-labelledby="help-title">
      <div className="help">
        <div className="row">
          <h2 id="help-title" style={{ flex: 1 }}>How this works</h2>
          <button className="btn" onClick={() => d.current?.close()} autoFocus>Close</button>
        </div>
        <ol>
          <li><b>Library.</b> A <i>shot</i> is one crash run (a video, or a set of screenshots). Add images (drop them, or <i>Add images…</i>); each becomes a <i>frame</i>. Every frame and its status (untraced, traced, paired, linked) is listed here, grouped by shot: click one to reopen it.</li>
          <li><b>Annotate.</b> Mark <i>regions</i>: the part of the image one source shows (real test, BeamNG, Rubble). Choose the structure (vehicle side, top, three-quarter, a truss…). The element list walks you through its keypoints, lines and wheels in order; click to place, drag to adjust, flag what is blurred or guessed. Everything saves as you go; ⌘Z undoes.</li>
          <li><b>Pair.</b> Frames of the same shot and view are compared against the earliest traced one (the “before”), or the one you choose. The regions of one image (real | BeamNG) are compared with each other.</li>
          <li><b>Link.</b> Link the shot to a Rubble scenario; each frame to a moment (and a named camera, or its own camera set by hand). Rubble's outline is generated automatically in the same structure by Rust (crash_outlines).</li>
          <li><b>Compare.</b> Every source in one normalised frame, each against its own before, and the shape numbers (hood fold, chord, creases, wheel setback, angles), computed in Rust from the points.</li>
        </ol>
        <p className="muted small">
          Images stay on this machine (apps/reference-studio/library/, gitignored). Shots, frames and tracings are points and text, committed in
          engine/docs/crash-reference/studio/. Agents use the same actions: window.studio, WebMCP (document.modelContext), or POST /api/tool/&lt;name&gt;.
        </p>
      </div>
    </dialog>
  );
}
