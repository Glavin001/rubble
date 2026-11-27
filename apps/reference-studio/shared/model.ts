// The Reference Studio's data model (README "Data model"; JSON Schema files in schema/).
// Types only. Every type Rust also reads (tracings, cameras, scenario descriptions, the
// report, the registry) is generated from the Rust (src/generated/crash-reference.ts) and
// re-exported here; the rest is the studio's own bookkeeping (shots, frames, regions). The
// maths (normalising, measuring, Rubble's outlines, the comparison, design ranking) is Rust
// compiled to wasm, called from src/engine.ts.

import type { ElementValue, FrameCamera, Instance, Vis } from '../src/generated/crash-reference.ts';

export type {
  BodyDesc,
  BoxDesc,
  Camera,
  CompareSource,
  Contact,
  CrashReport,
  ElementValue,
  FrameCamera,
  Instance,
  Moment,
  RecordingStatus,
  Registry,
  RubbleSpec,
  ScenarioDesc,
  ScenarioInfo,
  SceneInfo,
  SceneSource,
  Vis,
} from '../src/generated/crash-reference.ts';

export type Xy = [number, number];

/** How well an element can be seen: a measure reports the worst of its inputs. */
export const VIS: Vis[] = ['clear', 'blurred', 'hidden'];

/** Where a region's picture comes from. */
export type SourceTag = 'real' | 'beamng' | 'rubble' | 'other';
export const SOURCES: SourceTag[] = ['real', 'beamng', 'rubble', 'other'];

/** One crash run seen in footage (or in a set of screenshots). */
export interface Shot {
  id: string;
  title: string;
  source:
    | { kind: 'video'; url: string; note?: string }
    | { kind: 'screenshots'; note?: string };
  /** The Rubble scenario setup this run is recreated by: a built-in scenario (S01...) or a
   * data scenario of the workspace (scenarios/<id>.json). */
  rubble: { scenario: string; contact: string; run: number } | null;
  /** The frame each source's shapes are compared against ("before"); frames may override. */
  before_frame: string | null;
  notes: string;
  updated?: string;
}

export interface Region {
  id: string;
  label: string;
  source: SourceTag;
  /** Normalised image coordinates: [x0, y0, x1, y1] from the top-left corner. */
  crop: [number, number, number, number];
  tracing: Tracing;
}

/** What was traced in one region: instances of structures. */
export interface Tracing {
  instances: Instance[];
  traced_by: string;
  approximate: boolean;
  note: string;
  updated?: string;
}

/** Where an image sits in the recorded run: a time chosen by scrubbing in the Scene view's
 * Review mode (ms after first contact, any value, not only a named moment), with its own
 * camera (`Frame.camera`). `recording`: the hash of the recording it was pinned on; when the
 * scene is re-simulated it differs, and the pin is flagged "check after re-simulation". */
export interface FramePin {
  at_ms: number;
  recording: string;
  /** The named moment it is nearest, as a label only. */
  label?: string | null;
  pinned: string;
}

/** An image in the library and what is known about it. */
export interface Frame {
  id: string;
  shot: string | null;
  /** The image file in the workspace's library/ (never committed); null when it is not there yet. */
  image: { file: string; width: number; height: number; sha256: string } | null;
  moment: { label: string; after_impact_ms: number | null };
  /** Seconds into the shot's video, when it came from one. */
  time: { video_s: number; approximate: boolean } | null;
  /** A view label: side, top, three-quarter, ... */
  view: string;
  notes: string;
  /** The Rubble moment and named scenario camera this frame is compared with when it has no
   * pin (null: picked from the moment label and the view). */
  rubble: { moment: string | null; camera: string | null };
  /** This image's own camera, set by hand; used instead of the named camera when present. */
  camera: FrameCamera | null;
  /** Its time in the recorded run (set with its camera in Review); used instead of the named moment. */
  pin?: FramePin | null;
  /** Overrides the shot's before frame. */
  before_frame?: string | null;
  regions: Region[];
  updated?: string;
}

export type ElementKind = 'point' | 'polyline' | 'circle';

export interface ElementDef {
  name: string;
  type: ElementKind;
  label: string;
  colour: string;
  hint: string;
  rubble?: unknown;
}

/** A structure kind as data: structures/<id>.json (the Rust reads its own fields of it). */
export interface StructureSchema {
  id: string;
  title: string;
  kind: string;
  view: string;
  description: string;
  orientation: { front: ('left' | 'right')[]; default: 'left' | 'right'; note: string };
  rubble_view: 'car_frame' | 'camera';
  elements: ElementDef[];
  frames: { id: string; label: string; origin: unknown; unit: unknown }[];
  measures: { id: string; label: string; kind: string }[];
}

export type Status = 'no image' | 'untraced' | 'traced' | 'paired' | 'linked' | 'pinned';

export interface FrameSummary {
  id: string;
  shot: string | null;
  image: Frame['image'];
  thumb: string | null;
  moment: string;
  view: string;
  regions: number;
  traced: number;
  status: Status[];
}

/** A traced element has data. */
export function hasData(v: ElementValue | undefined): boolean {
  return !!v && (!!v.p || (!!v.pts && v.pts.length > 0) || !!v.c);
}
