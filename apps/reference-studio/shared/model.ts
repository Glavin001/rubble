// The Reference Studio's data model (README "Data model"; JSON Schema files in schema/).
// Types only: the server reads and writes these as JSON files, the page shows them. Shape
// maths (normalising, measuring, reading Rubble's outlines) is Rust
// (engine-softbody crash_reference::shape), reached through the server.

export type Xy = [number, number];

/** How well an element can be seen: a measure reports the worst of its inputs. */
export type Vis = 'clear' | 'blurred' | 'hidden';
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
  /** The Rubble scenario setup this run is recreated by. */
  rubble: { scenario: string; contact: string; run: number } | null;
  /** The frame each source's shapes are compared against ("before"); frames may override. */
  before_frame: string | null;
  notes: string;
  updated?: string;
}

/** A camera set by hand for one image, in the scenario's world frame (metres, +Y up). */
export interface FrameCamera {
  eye: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  fov_y_deg: number;
  /** The region it applies to (its crop's aspect), or null for the whole image. */
  region: string | null;
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

export interface ElementValue {
  /** keypoint */
  p?: Xy;
  /** polyline, in order */
  pts?: Xy[];
  /** circle: centre and a point on the rim */
  c?: Xy;
  rim?: Xy;
  vis: Vis;
}

/** One structure traced in a region; points in the region's normalised coordinates. */
export interface Instance {
  id: string;
  /** A StructureSchema id (structures/<id>.json). */
  schema: string;
  /** Where the structure's front is in the image. */
  front: 'left' | 'right';
  elements: Record<string, ElementValue>;
}

/** An image in the library and what is known about it. */
export interface Frame {
  id: string;
  shot: string | null;
  /** The image file in the library (gitignored); null when it is not on disk yet. */
  image: { file: string; width: number; height: number; sha256: string } | null;
  moment: { label: string; after_impact_ms: number | null };
  /** Seconds into the shot's video, when it came from one. */
  time: { video_s: number; approximate: boolean } | null;
  /** A view label: side, top, three-quarter, ... */
  view: string;
  notes: string;
  /** The Rubble moment and named scenario camera this frame is compared with (null: picked
   * from the moment label and the view). */
  rubble: { moment: string | null; camera: string | null };
  /** This image's own camera, set by hand; used instead of the named camera when present. */
  camera: FrameCamera | null;
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

/** A structure kind as data: structures/<id>.json. */
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

export interface Scenario {
  id: string;
  title: string;
  designs: string[];
  cameras: { name: string; eye: number[]; target: number[] }[];
  moments: { name: string; after_impact_ms: number }[];
  reference: { video: string; at_s: number[]; real_test: string };
}

export type Status = 'no image' | 'untraced' | 'traced' | 'paired' | 'linked';

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
