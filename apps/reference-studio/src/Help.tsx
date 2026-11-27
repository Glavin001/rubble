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
          <li><b>Link.</b> Link the shot to a Rubble scenario (built-in, or one you recreated); each frame to a moment (and a named camera, or its own camera set by hand).</li>
          <li><b>Scene.</b> Recreate the crash: in <i>Setup</i> pick the vehicles' designs, poses and speeds and place the boxes they hit (drag them, or type numbers); <i>Simulate</i> runs it on this page's GPU and records every 4 ms. In <i>Review</i> scrub the recording and move the camera: for each reference image, find its moment and its view with its outline over the 3D view, then <i>Pin this image here</i>. Changing the setup keeps every pin and flags it for a check.</li>
          <li><b>Compare.</b> Every source in one normalised frame, each against its own before, and the shape numbers (hood fold, chord, creases, wheel setback, angles). Rubble's shape comes from the recording at the image's pin. All of it computed by Rust (in this page, as wasm).</li>
        </ol>
        <p className="muted small">
          Everything lives in the workspace folder you chose (or this browser's storage): images in library/ (never committed), recordings in recordings/,
          and shots, frames, structures and scenarios as points and text (in the repo: apps/reference-studio/workspace/). Agents use the same actions:
          window.studio or WebMCP (document.modelContext).
        </p>
      </div>
    </dialog>
  );
}
