// The studio's actions, defined once. The same list is
// - registered with WebMCP (document.modelContext.registerTool) when the page has it,
// - exposed as window.studio (studio.tools, studio.call(name, input), studio.<name>(input)),
// - served over HTTP: POST /api/tool/<name> with the input as JSON (server tools only).
// "server" tools run in the studio's server (they read and write the files); "page" tools
// change what the open page shows and exist only in the page.

export interface ToolDef {
  name: string;
  title: string;
  description: string;
  side: 'server' | 'page';
  readOnly: boolean;
  inputSchema: { type: 'object'; properties: Record<string, unknown>; required?: string[]; additionalProperties?: boolean };
}

const str = (description: string, extra: Record<string, unknown> = {}) => ({ type: 'string', description, ...extra });
const num = (description: string, extra: Record<string, unknown> = {}) => ({ type: 'number', description, ...extra });
const xy = (what: string) => num(`${what}, normalised to the region (0 at its left/top edge, 1 at its right/bottom)`);
const FRAME = str('Frame id (list_frames)');
const REGION = str('Region id within the frame (get_frame lists them), e.g. "real" or "beamng"');
const ELEMENT = str('Element name from the region\'s structure schema (list_schemas), e.g. "hood" or "front_wheel"');
const INSTANCE = str('Instance id when a region holds more than one structure (default: the first)');
const obj = (properties: Record<string, unknown>, required: string[] = []) => ({ type: 'object' as const, properties, required, additionalProperties: false });

export const TOOLS: ToolDef[] = [
  { name: 'list_shots', title: 'List shots', side: 'server', readOnly: true, inputSchema: obj({}),
    description: 'List every shot (one crash run seen in footage or screenshots) with its linked Rubble scenario and how many of its frames are traced, paired and linked.' },
  { name: 'list_frames', title: 'List frames', side: 'server', readOnly: true,
    inputSchema: obj({ shot: str('Only frames of this shot id; "" for frames in no shot'), status: str('Only frames with this status', { enum: ['no image', 'untraced', 'traced', 'paired', 'linked'] }) }),
    description: 'List frames (images) with their shot, moment, view, regions and status badges.' },
  { name: 'list_untraced', title: 'List untraced frames', side: 'server', readOnly: true, inputSchema: obj({}),
    description: 'List frames that have an image but no traced element yet: the work still to do.' },
  { name: 'get_frame', title: 'Get a frame', side: 'server', readOnly: true, inputSchema: obj({ frame: FRAME }, ['frame']),
    description: 'Everything about one frame: image size, shot, moment, view, Rubble link, camera, regions and their tracings.' },
  { name: 'get_tracing', title: 'Get a tracing', side: 'server', readOnly: true, inputSchema: obj({ frame: FRAME, region: REGION }, ['frame']),
    description: 'The tracing of one region (or every region) of a frame: instances, each element\'s points (normalised region coordinates) and visibility.' },
  { name: 'list_schemas', title: 'List structure schemas', side: 'server', readOnly: true, inputSchema: obj({}),
    description: 'The structure schemas (vehicle-side, vehicle-top, vehicle-three-quarter, truss-side, ...): their elements in tracing order with type (point, polyline, circle), label and a hint on where to place them, and their measures.' },
  { name: 'list_scenarios', title: 'List Rubble scenarios', side: 'server', readOnly: true, inputSchema: obj({}),
    description: 'Rubble\'s crash reference scenarios (from the Rust registry): id, title, designs, named cameras and moments.' },
  { name: 'create_shot', title: 'Create a shot', side: 'server', readOnly: false,
    inputSchema: obj({ id: str('Short id (letters, digits, - _ .); made from the title when omitted'), title: str('What the run is'), video_url: str('Source video, if any (else the shot is "screenshots")'), notes: str('Free notes') }, ['title']),
    description: 'Create a shot: one crash run seen in footage (a video URL) or in a set of screenshots.' },
  { name: 'update_shot', title: 'Update a shot', side: 'server', readOnly: false,
    inputSchema: obj({ shot: str('Shot id'), title: str('New title'), video_url: str('Video URL ("" for screenshots)'), notes: str('Notes'), before_frame: str('Frame each source is compared against ("" for automatic: the earliest traced frame of the same view)') }, ['shot']),
    description: 'Rename a shot or change its source, notes or before frame.' },
  { name: 'add_image', title: 'Add an image', side: 'server', readOnly: false,
    inputSchema: obj({ name: str('File name with extension (.png, .jpg, .webp)'), data_base64: str('The image bytes, base64'), shot: str('Shot to put the new frame in'), view: str('View label (side, top, ...)'), moment: str('Moment label (before, first contact, peak crush, ...)') }, ['name', 'data_base64']),
    description: 'Save an image into the library (gitignored) and create a frame for it. The page\'s drag-and-drop does the same.' },
  { name: 'assign_frame', title: 'Assign a frame to a shot', side: 'server', readOnly: false,
    inputSchema: obj({ frame: FRAME, shot: str('Shot id, or "" for none') }, ['frame', 'shot']),
    description: 'Put a frame in a shot (or take it out).' },
  { name: 'set_frame_info', title: 'Set frame information', side: 'server', readOnly: false,
    inputSchema: obj({ frame: FRAME, moment: str('Moment label: before, first contact, early crush, mid crush, peak crush, rebound, aftermath, ...'), after_impact_ms: num('Time after first contact if known (ms)'), view: str('View label: side, top, three-quarter, ...'), video_s: num('Seconds into the shot\'s video'), notes: str('Notes'), before_frame: str('Frame to compare against as "before" ("" for the shot\'s rule)') }, ['frame']),
    description: 'Set a frame\'s moment, time, view label, notes or before frame.' },
  { name: 'mark_region', title: 'Mark a region', side: 'server', readOnly: false,
    inputSchema: obj({ frame: FRAME, region: str('Region id to create or change (made from the source when omitted)'), label: str('Label shown, e.g. "real test"'), source: str('Where this part of the image comes from', { enum: ['real', 'beamng', 'rubble', 'other'] }), x0: num('Left edge (0..1 of the image width)'), y0: num('Top edge (0..1 of the image height)'), x1: num('Right edge'), y1: num('Bottom edge') }, ['frame', 'source']),
    description: 'Create or change a region: a crop rectangle of the image (normalised image coordinates) with a source tag. An image with a real test half and a BeamNG half gets two regions. Without a rectangle the region is the whole image.' },
  { name: 'remove_region', title: 'Remove a region', side: 'server', readOnly: false, inputSchema: obj({ frame: FRAME, region: REGION }, ['frame', 'region']),
    description: 'Remove a region and its tracing from a frame.' },
  { name: 'set_region_schema', title: 'Choose a structure', side: 'server', readOnly: false,
    inputSchema: obj({ frame: FRAME, region: REGION, schema: str('Structure schema id (list_schemas)'), front: str('Which side of the image the structure\'s front is on', { enum: ['left', 'right'] }), instance: INSTANCE }, ['frame', 'region', 'schema']),
    description: 'Choose the structure traced in a region (creates its instance; elements whose names the schema keeps are kept).' },
  { name: 'set_keypoint', title: 'Place a keypoint or circle', side: 'server', readOnly: false,
    inputSchema: obj({ frame: FRAME, region: REGION, element: ELEMENT, x: xy('x'), y: xy('y'), part: str('For a circle: "c" (centre) or "rim" (a point on its edge)', { enum: ['c', 'rim'] }), instance: INSTANCE }, ['frame', 'region', 'element', 'x', 'y']),
    description: 'Place or move a keypoint, or a circle\'s centre or rim point.' },
  { name: 'add_polyline_point', title: 'Add a polyline point', side: 'server', readOnly: false,
    inputSchema: obj({ frame: FRAME, region: REGION, element: ELEMENT, x: xy('x'), y: xy('y'), index: num('Insert before this index (default: append at the end)'), instance: INSTANCE }, ['frame', 'region', 'element', 'x', 'y']),
    description: 'Add a point to a polyline (appended, or inserted at an index).' },
  { name: 'move_polyline_point', title: 'Move a polyline point', side: 'server', readOnly: false,
    inputSchema: obj({ frame: FRAME, region: REGION, element: ELEMENT, index: num('Point index (0 is the first)'), x: xy('x'), y: xy('y'), instance: INSTANCE }, ['frame', 'region', 'element', 'index', 'x', 'y']),
    description: 'Move one point of a polyline.' },
  { name: 'remove_point', title: 'Remove a point', side: 'server', readOnly: false,
    inputSchema: obj({ frame: FRAME, region: REGION, element: ELEMENT, index: num('Polyline point index; omitted: clear the whole element'), instance: INSTANCE }, ['frame', 'region', 'element']),
    description: 'Remove one polyline point, or clear an element (keypoint, circle or polyline) entirely.' },
  { name: 'set_visibility', title: 'Set visibility', side: 'server', readOnly: false,
    inputSchema: obj({ frame: FRAME, region: REGION, element: ELEMENT, vis: str('How well it can be seen', { enum: ['clear', 'blurred', 'hidden'] }), instance: INSTANCE }, ['frame', 'region', 'element', 'vis']),
    description: 'Flag an element clear, blurred, or hidden/guessed. Measures report the worst flag of what they use.' },
  { name: 'set_tracing_info', title: 'Set tracing information', side: 'server', readOnly: false,
    inputSchema: obj({ frame: FRAME, region: REGION, traced_by: str('Who traced it'), approximate: { type: 'boolean', description: 'Still a first pass' }, note: str('Note') }, ['frame', 'region']),
    description: 'Record who traced a region, whether it is approximate, and a note.' },
  { name: 'copy_tracing_from_previous', title: 'Copy the previous frame\'s tracing', side: 'server', readOnly: false,
    inputSchema: obj({ frame: FRAME, region: REGION }, ['frame', 'region']),
    description: 'Start a region\'s tracing from the same region of the previous frame of the same shot (by video time), to adjust instead of placing every point again.' },
  { name: 'link_scenario', title: 'Link a shot to a scenario', side: 'server', readOnly: false,
    inputSchema: obj({ shot: str('Shot id'), scenario: str('Rubble scenario id (list_scenarios), or "" to unlink'), contact: str('Contact setting', { enum: ['discrete', 'edge', 'continuous'] }), run: num('Run (0: nominal)') }, ['shot', 'scenario']),
    description: 'Say which Rubble scenario setup recreates a shot.' },
  { name: 'link_frame', title: 'Link a frame to a moment and camera', side: 'server', readOnly: false,
    inputSchema: obj({ frame: FRAME, moment: str('Scenario moment name, or "" for automatic (from the moment label)'), camera: str('Named scenario camera, or "" for automatic (from the view label); unused when the frame has its own camera') }, ['frame']),
    description: 'Say which of the scenario\'s moments (and named cameras) a frame is compared with.' },
  { name: 'set_frame_camera', title: 'Set a frame\'s camera', side: 'server', readOnly: false,
    inputSchema: obj({ frame: FRAME, eye: { type: 'array', items: { type: 'number' }, description: 'Camera position [x, y, z] (m, scenario world, +Y up)' }, target: { type: 'array', items: { type: 'number' }, description: 'Point looked at [x, y, z]' }, up: { type: 'array', items: { type: 'number' }, description: 'Up direction (default [0, 1, 0])' }, fov_y_deg: num('Vertical field of view (degrees, default 60)'), region: str('Region the camera\'s image is (its crop), or "" for the whole image'), clear: { type: 'boolean', description: 'Remove the frame\'s camera (use the named scenario camera again)' } }, ['frame']),
    description: 'Set this image\'s own camera by hand (eye, target, up, field of view). Rubble\'s outline for the frame is then projected through it instead of the scenario\'s named camera.' },
  { name: 'get_rubble_outline', title: 'Get Rubble\'s outline', side: 'server', readOnly: true, inputSchema: obj({ frame: FRAME, schema: str('Structure schema (default: the frame\'s first traced one)') }, ['frame']),
    description: 'Rubble\'s automatically generated outline for the frame\'s scenario, moment and camera, in the same structure schema, over Rubble\'s own still (runs the Rust crash_outlines tool on demand and caches it; first call takes about 10 s).' },
  { name: 'get_comparison', title: 'Get the comparison table', side: 'server', readOnly: true, inputSchema: obj({ frame: FRAME }, ['frame']),
    description: 'For a frame: every source (each traced region and Rubble) normalised by its own before frame, and the shape-number table (fold height, chord, creases, wheel setback, angles, ...), computed in Rust.' },
  { name: 'put_frame', title: 'Replace a frame', side: 'server', readOnly: false, inputSchema: obj({ frame: { type: 'object', description: 'A whole Frame document (get_frame)' } }, ['frame']),
    description: 'Replace a frame document whole (the page\'s undo and redo use it).' },
  { name: 'open_view', title: 'Open a view', side: 'page', readOnly: false,
    inputSchema: obj({ view: str('Which view', { enum: ['library', 'annotate', 'link', 'compare'] }), frame: str('Frame to open in it') }, ['view']),
    description: 'Show the library, or open a frame in the annotate, link or compare view.' },
  { name: 'open_frame', title: 'Open a frame', side: 'page', readOnly: false, inputSchema: obj({ frame: FRAME, region: REGION }, ['frame']),
    description: 'Open a frame in the annotate view.' },
  { name: 'select_element', title: 'Select an element', side: 'page', readOnly: false, inputSchema: obj({ element: ELEMENT, region: REGION }, ['element']),
    description: 'Make an element the active one in the annotate view (new clicks place it).' },
  { name: 'set_zoom', title: 'Zoom', side: 'page', readOnly: false, inputSchema: obj({ zoom: num('Zoom factor (1: fit)'), x: num('Centre x (0..1 of the image)'), y: num('Centre y (0..1 of the image)') }, ['zoom']),
    description: 'Zoom the annotate view about a point of the image.' },
  { name: 'describe_view', title: 'Describe the current view', side: 'page', readOnly: true, inputSchema: obj({}),
    description: 'What the page shows: the view, the open frame and region, the active element, zoom and pan, and the next element the guided mode asks for.' },
];

export const TOOL_NAMES = new Set(TOOLS.map((t) => t.name));
