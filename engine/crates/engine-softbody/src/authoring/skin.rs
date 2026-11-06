//! The visual skin: the constructed car of [`super::parts`] (a dense body
//! surface cut into separate panels with real shut lines, and modelled lamps,
//! grille, glass, mirrors, handles, wheels and brakes), bound piece by piece to
//! the skeleton so it deforms with the crash, flexbody style.
//!
//! Binding: every vertex rides one triangle of nodes (n₀, n₁, n₂):
//!
//! ```text
//! e₁ = P₁ − P₀, e₂ = P₂ − P₀, N = normalize(e₁ × e₂)
//! position = P₀ + α·e₁ + β·e₂ + h·N
//! T = normalize(e₁), B = N × T, normal = nₜ·T + n_b·B + n_n·N
//! ```
//!
//! with (α, β, h) and (nₜ, n_b, n_n) fixed at rest. A piece binds to one
//! physics part: the part its vertices lie on (an area-weighted vote of the
//! skin under them). A vertex within the contact band of the skin rides the
//! skin triangle under it (the physics surface, a shut line's triangles
//! included: [`SKIN_BAND`]); off it, a frame of three of its part's nodes,
//! BeamNG's flexbody binding ([`BIND_MAX`] bounds its distance, or the build
//! fails). So a door's skin, its rolled edges and its handle ride the door's
//! nodes and leave with it. Nothing here depends on the skeleton's node
//! spacing: the same mesh rides a coarse or a dense skeleton.
//! Wheels ride their tyre's fan triangles (they turn); calipers the axle and a
//! knuckle node (they do not); seats and the dashboard their frames.
//! `showcase-car-render.mjs` evaluates this on the GPU from the explicit
//! solver's node buffer ([`deform`] is the same on the CPU).

use super::parts::design::{assemble, sedan_styling};
use super::parts::{Bind, Mesh, Piece};
use super::vehicle::Built;
use crate::explicit_beams::{ExplicitBeams, ExplicitBeamsSettings};
use glam::Vec3;
use std::collections::HashMap;

/// A binding triangle stretched past this × a rest edge has torn: its vertices
/// are dropped (their triangles vanish) instead of stretching across the tear.
pub const TEAR_STRETCH: f32 = 2.5;

/// Material ids of the visual mesh.
pub mod mat {
    pub const PAINT: u8 = 0;
    pub const GLASS: u8 = 1;
    pub const CHROME: u8 = 2;
    pub const TRIM: u8 = 3;
    pub const HEADLIGHT: u8 = 4;
    pub const TAILLIGHT: u8 = 5;
    pub const TYRE: u8 = 6;
    pub const RIM: u8 = 7;
    pub const GAP: u8 = 8;
    pub const UNDERBODY: u8 = 9;
    pub const PLATE: u8 = 10;
    pub const GRILLE: u8 = 11;
    pub const SEAT: u8 = 12;
    pub const DASH: u8 = 13;
    pub const DOORCARD: u8 = 14;
    pub const HEADLINER: u8 = 15;
    pub const CARPET: u8 = 16;
    /// Clear lamp lens (drawn with the glass).
    pub const LENS: u8 = 17;
    /// Daytime running light guide (emissive white).
    pub const DRL: u8 = 18;
    /// Lamp reflector (bright, slightly rough metal).
    pub const REFLECTOR: u8 = 19;
    /// Red tail-lamp lens (drawn with the glass).
    pub const TAIL_LENS: u8 = 20;
    /// Tail-lamp light guides (emissive red).
    pub const TAIL_LED: u8 = 21;
    /// Brake disc and wheel barrel (cast iron / dark metal).
    pub const BRAKE: u8 = 22;
    /// Brake caliper (painted).
    pub const CALIPER: u8 = 23;
    /// Tail-lamp housing back (dark red).
    pub const TAIL_BACK: u8 = 24;
    /// Laminated windscreen (cracks instead of shattering; drawn with the glass).
    pub const SCREEN: u8 = 25;
    /// A lit display (instrument cluster, centre screen): its content drawn
    /// from the texture coordinates.
    pub const DISPLAY: u8 = 26;
    /// Privacy glass (rear doors, quarter lights, backlight): tempered, dark
    /// tint; drawn with the glass.
    pub const PRIVACY: u8 = 27;
    /// A rim's diamond-cut spoke faces (bright machined aluminium).
    pub const MACHINED: u8 = 28;
    /// The B-pillar applique: satin black (no mirror-bright sky at grazing
    /// angles once the doors move).
    pub const PILLAR: u8 = 29;
    /// A panel's inner stamping (door inner skin and shut faces, bonnet and
    /// boot-lid inner frames, aprons): grey e-coat, seen through a tear or
    /// once the outer skin is gone.
    pub const INNER: u8 = 30;
    /// The engine and its bay's hardware (block, cam cover, radiator, strut
    /// towers): cast and painted metal.
    pub const ENGINE: u8 = 31;
    pub const COUNT: usize = 32;
}

/// Material names, by id.
pub const MATERIAL_NAMES: [&str; mat::COUNT] = [
    "paint",
    "glass",
    "chrome",
    "trim",
    "headlight",
    "taillight",
    "tyre",
    "rim",
    "gap",
    "underbody",
    "plate",
    "grille",
    "seat",
    "dash",
    "doorcard",
    "headliner",
    "carpet",
    "lens",
    "drl",
    "reflector",
    "tail_lens",
    "tail_led",
    "brake",
    "caliper",
    "tail_back",
    "screen",
    "display",
    "privacy",
    "machined",
    "pillar",
    "inner",
    "engine",
];

/// Whether a material is drawn blended (after the opaque ones).
pub fn transparent(m: u8) -> bool {
    matches!(
        m,
        mat::GLASS | mat::LENS | mat::TAIL_LENS | mat::SCREEN | mat::PRIVACY
    )
}

#[derive(Clone, Debug, Default)]
pub struct ShowcaseVisual {
    /// Rest positions in the `.truck`'s coordinates.
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub colors: Vec<[f32; 3]>,
    pub material: Vec<u8>,
    pub indices: Vec<u32>,
    /// Per vertex: the three solver nodes (indices into the explicit solver's
    /// node list, tyre nodes included) and (α, β, h).
    pub bind_nodes: Vec<[u32; 3]>,
    pub bind_coords: Vec<[f32; 3]>,
    /// The rest normal in the triangle's frame (T, B, N).
    pub bind_normal: Vec<[f32; 3]>,
    /// The binding triangle's rest edge lengths |e₁|, |e₂|: a vertex whose
    /// triangle has stretched past [`TEAR_STRETCH`] × either is torn (dropped).
    pub bind_rest: Vec<[f32; 2]>,
    /// Baked ambient occlusion (1 open; shut lines, recesses and lamp throats
    /// darker).
    pub ao: Vec<f32>,
    /// 1 when the vertex's normal follows the skin's node normals (smooth
    /// shading across the binding triangles): n = normalize(n_rest + Σ wᵢ (Nᵢ − Rᵢ))
    /// with wᵢ the barycentric weights (1 − α − β, α, β) of nodes n₀ n₁ n₂, Nᵢ a
    /// node's normal now (the normalised sum of its skin triangles' cross
    /// products) and Rᵢ at rest.
    pub smooth: Vec<u8>,
    /// The skin triangles node normals come from, and per node (CSR over the
    /// solver's nodes) the triangles it is in, and its rest normal.
    pub normal_tris: Vec<[u32; 3]>,
    pub node_tri_start: Vec<u32>,
    pub node_tri_list: Vec<u32>,
    pub node_rest_normal: Vec<[f32; 3]>,
    /// The solver's node count for this vehicle (tyre nodes included).
    pub node_count: u32,
    /// 1 for a rigid part's vertex (mirror, handle, lamp): (α, β, h) are its
    /// rest offset in the triangle's orthonormal frame (T, B, N), so the part
    /// moves rigidly with the triangle and never stretches.
    pub rigid: Vec<u8>,
    /// Every solver node's rest position (the `.truck`'s coordinates): the
    /// renderer's tear field compares skin triangles against it.
    pub node_rest: Vec<[f32; 3]>,
    /// The pieces (name, first vertex, vertex count), in order.
    pub pieces: Vec<(String, u32, u32)>,
    /// The damage units (glass panes, lamps, mirrors): each a set of the
    /// solver's beams whose state decides the unit's ([`unit_states`]).
    pub units: Vec<DamageUnit>,
    /// Per vertex its damage unit (an index into `units`, `u32::MAX` none).
    pub unit: Vec<u32>,
    /// Per vertex 1 where it throws a glass or lens shard as its unit breaks
    /// (area × [`SHARD_DENSITY`] per unit).
    pub shard: Vec<u8>,
    /// Every solver beam's spawn rest length, in the solver's order (this
    /// vehicle's beams: its first is the vehicle's `beamOffset`).
    pub beam_rest0: Vec<f32>,
    /// Per solver node (CSR) the solver beams on it, tyre beams excluded: the
    /// renderer's per-node plastic strain (paint gloss, windscreen cracks).
    pub node_beam_start: Vec<u32>,
    pub node_beam_list: Vec<u32>,
    /// Per solver node its surface material ([`node_mat`]): which nodes throw
    /// sparks when they scrape (metal), chips (plastic), nothing (rubber).
    pub node_material: Vec<u8>,
    /// Texture files (an imported vehicle's maps, by file name in its
    /// folder; `flextex` decodes them): colour maps (sRGB) and data maps
    /// (linear: normal, roughness, AO). Per vertex its texture set (an index
    /// into `tex_sets`; `u16::MAX` or absent: untextured), each set its
    /// [colour, normal, roughness, AO] files (indices into `textures`, then
    /// `data_textures` for the other three; `u16::MAX` none). The colour
    /// multiplies the vertex's (sampled at its `uvs`, v up); the normal map
    /// (DirectX, green down) tilts its normal in the frame of its `uvs`;
    /// roughness (red) replaces its material's; AO (red) multiplies its own.
    pub textures: Vec<String>,
    pub data_textures: Vec<String>,
    pub tex: Vec<u16>,
    pub tex_sets: Vec<[u16; 4]>,
    /// Per vertex riding a shut line's skin triangle (nodes of two parts):
    /// its index + 1 into the `alt_*` bindings (0 or absent: none), its own
    /// part's frame, computed at build time. It switches to it, for good, the
    /// moment that triangle dies (a beam in [`Self::skin_death`] breaks) or
    /// tears: a detached door or bonnet takes its edge with it instead of
    /// drawing a ribbon of the body's skin.
    pub alt: Vec<u32>,
    pub alt_nodes: Vec<[u32; 3]>,
    pub alt_coords: Vec<[f32; 3]>,
    pub alt_normal: Vec<[f32; 3]>,
    pub alt_rest: Vec<[f32; 2]>,
    /// Per alt binding the frame it replaces (an index into `skin_death`).
    pub alt_tri: Vec<u32>,
    /// Per frame across a part border (the skeleton's skin triangles, in
    /// order, then the node frames with a node across one) the solver beams
    /// whose break kills it (`u32::MAX` none): its collision triangle's
    /// break list (spec §18: its sides' and its break group's beams), else
    /// its sides' beams across the part border.
    pub skin_death: Vec<[u32; 4]>,
    /// Per vertex the skeleton part it rides (`u16::MAX`: none, a wheel's
    /// or a structure group's).
    pub part: Vec<u16>,
    /// Per vertex its wheel: `tyre << 1` for a wheel's rotating piece (tyre,
    /// rim, disc), `tyre << 1 | 1` for its non-rotating hardware (caliper),
    /// `u32::MAX` none; `tyre` indexes `tyre_axles`.
    pub wheel: Vec<u32>,
    /// Per tyre (the solver's order) its axle nodes.
    pub tyre_axles: Vec<[u32; 2]>,
}

/// `ShowcaseVisual::rigid` values.
pub mod rigid_kind {
    /// Rides its frame affinely (the body).
    pub const NONE: u8 = 0;
    /// Rigid on its frame's orthonormal basis (mirrors, handles, lamps).
    pub const RIGID: u8 = 1;
    /// On a raycast wheel ([`super::bind_raycast_wheels`]): `bind_nodes[0]`
    /// the drive wheel, `bind_coords` and `bind_normal` its rest offset and
    /// normal in the wheel's frame (axle, up, forward); it turns with the
    /// wheel's spin.
    pub const WHEEL: u8 = 2;
    /// The same, steered and sprung but not spinning (a caliper).
    pub const WHEEL_HUB: u8 = 3;
}

/// A raycast wheel as drawn (the drive state's outputs: `WheelState`'s
/// centre, axle and up, and its spin angle).
#[derive(Clone, Copy, Debug)]
pub struct WheelPose {
    pub centre: Vec3,
    /// Unit, steered, pointing left.
    pub axle: Vec3,
    pub up: Vec3,
    pub spin: f32,
}

/// A wheel vertex's position and normal from its rest offset `l` and normal
/// `n` in its wheel's frame (axle A, up U, forward F = A × U): it turns about
/// the axle by the spin angle (a positive spin rolls the top forward) unless
/// `hub`. The renderer's `wheelVertex()` is the same.
pub fn wheel_vertex(l: [f32; 3], n: [f32; 3], w: &WheelPose, hub: bool) -> (Vec3, Vec3) {
    let a = w.axle.normalize_or_zero();
    let u = (w.up - a * a.dot(w.up)).normalize_or_zero();
    let f = a.cross(u);
    let (sn, c) = if hub { (0.0, 1.0) } else { w.spin.sin_cos() };
    let (ur, fr) = (u * c + f * sn, f * c - u * sn);
    (
        w.centre + a * l[0] + ur * l[1] + fr * l[2],
        a * n[0] + ur * n[1] + fr * n[2],
    )
}

/// Rebinds the wheels' pieces onto raycast wheels (a spawn without tyres,
/// `ExplicitBeams::from_truck_raycast`): each tyre's vertices ride the drive
/// wheel on the same axle nodes ([`rigid_kind::WHEEL`], its caliper
/// [`rigid_kind::WHEEL_HUB`]), at their rest offset from the tyre's centre
/// in the frame the drive uses at rest (the body's left and up from its
/// reference nodes; `DriveConfig::hub_axles` aside). Drawn from the drive
/// state, as a BeamNG flexbody on a wheel rides the wheel's own nodes.
pub fn bind_raycast_wheels(
    v: &mut ShowcaseVisual,
    config: &crate::explicit_drive::DriveConfig,
) -> Result<(), String> {
    let rest = |i: u32| {
        v.node_rest
            .get(i as usize)
            .map(|q| Vec3::from_array(*q))
            .ok_or_else(|| format!("raycast wheels: node {i} past the visual's nodes"))
    };
    let [c, b, l] = config.refs;
    let fw = (rest(c)? - rest(b)?).normalize_or_zero();
    let up = fw.cross(rest(l)? - rest(c)?).normalize_or_zero();
    let left = up.cross(fw);
    // Per tyre its drive wheel and rest centre.
    let mut of: Vec<Option<(u32, Vec3)>> = Vec::with_capacity(v.tyre_axles.len());
    for ax in &v.tyre_axles {
        let k = config.wheels.iter().position(|w| {
            (w.node1 == ax[0] && w.node2 == ax[1]) || (w.node1 == ax[1] && w.node2 == ax[0])
        });
        of.push(match k {
            Some(k) => Some((k as u32, 0.5 * (rest(ax[0])? + rest(ax[1])?))),
            None => None,
        });
    }
    let pose = |centre: Vec3| WheelPose {
        centre,
        axle: left,
        up,
        spin: 0.0,
    };
    for i in 0..v.positions.len() {
        let w = v.wheel.get(i).copied().unwrap_or(u32::MAX);
        if w == u32::MAX {
            continue;
        }
        let Some(Some((k, centre))) = of.get((w >> 1) as usize).copied() else {
            return Err(format!(
                "raycast wheels: no drive wheel on tyre {}'s axle",
                w >> 1
            ));
        };
        let hub = w & 1 == 1;
        let fr = pose(centre);
        let (a, u) = (fr.axle, fr.up);
        let f = a.cross(u);
        let d = Vec3::from_array(v.positions[i]) - centre;
        let n = Vec3::from_array(v.normals[i]);
        v.bind_nodes[i] = [k, k, k];
        v.bind_coords[i] = [d.dot(a), d.dot(u), d.dot(f)];
        v.bind_normal[i] = [n.dot(a), n.dot(u), n.dot(f)];
        v.bind_rest[i] = [1.0, 1.0];
        v.rigid[i] = if hub {
            rigid_kind::WHEEL_HUB
        } else {
            rigid_kind::WHEEL
        };
        v.smooth[i] = 0;
        if let Some(a) = v.alt.get_mut(i) {
            *a = 0;
        }
    }
    Ok(())
}

/// Surface material of a solver node (`ShowcaseVisual::node_material`).
pub mod node_mat {
    pub const METAL: u8 = 0;
    pub const PLASTIC: u8 = 1;
    pub const GLASS: u8 = 2;
    pub const RUBBER: u8 = 3;
}

/// Damage unit kinds (`DamageUnit::kind`).
pub mod unit_kind {
    /// Tempered glass (side windows, backlight): shatters whole. Its pane is
    /// dropped and throws its shards.
    pub const TEMPERED: u8 = 0;
    /// The laminated windscreen: cracks where its beams yield or break and
    /// stays in its frame (the interlayer holds it).
    pub const LAMINATED: u8 = 1;
    /// A lamp: its lens shatters and its light goes dark.
    pub const LAMP: u8 = 2;
    /// A mirror: its glass shatters.
    pub const MIRROR: u8 = 3;
}

/// Tempered glass's break strain: characteristic bending strength ≈ 120 MPa
/// over E ≈ 70 GPa (engine/docs/beamng-realism-plan.md P6). Glass has no
/// plastic range, so a pane whose beams have yielded past this has broken: a
/// deformation trigger, as BeamNG's deform groups switch glass.
pub const TEMPERED_BREAK_STRAIN: f32 = 120.0e6 / 70.0e9;
/// Annealed glass (the windscreen's plies): ≈ 45 MPa over 70 GPa. Its cracks
/// start where its beams yield past this.
pub const LAMINATED_CRACK_STRAIN: f32 = 45.0e6 / 70.0e9;
/// A lamp lens (PMMA, the usual tail lamp and outer lens material) breaks at
/// about 3 % elongation (PMMA's elongation at break is 2–10 %): a lamp whose
/// mount has yielded past this has a broken lens.
pub const LENS_BREAK_STRAIN: f32 = 0.03;
/// Shards thrown per square metre of broken glass or lens (each a few
/// centimetres of granules; tempered glass's real ~1 cm cubes would be ~10⁴
/// a square metre, too many particles). Visual only.
pub const SHARD_DENSITY: f32 = 400.0;

/// A damage unit: a pane, lamp or mirror whose state follows its beams.
#[derive(Clone, Debug, Default)]
pub struct DamageUnit {
    pub name: String,
    pub kind: u8,
    /// This vehicle's solver beams (indices in the solver's order).
    pub beams: Vec<u32>,
    /// Deformation trigger: the unit is broken once one of its live beams'
    /// |rest − rest₀| / rest₀ passes this (`f32::INFINITY`: only by breaking).
    pub strain: f32,
    /// Its shard seeds (vertices with `shard` 1).
    pub shards: u32,
}

/// One solver beam's state (the GPU buffer's broken flag and rest length).
#[derive(Clone, Copy, Debug)]
pub struct BeamState {
    pub broken: bool,
    pub rest: f32,
}

/// A damage unit's state: broken (any of its beams broken, or its live
/// beams' largest plastic strain past the unit's trigger) and that strain.
/// The renderer's `units` pass computes exactly this on the GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct UnitState {
    pub broken: bool,
    pub strain: f32,
}

/// Every unit's state from the solver's beams (this vehicle's, in order).
pub fn unit_states(v: &ShowcaseVisual, beams: &[BeamState]) -> Vec<UnitState> {
    v.units
        .iter()
        .map(|u| {
            let mut s = UnitState::default();
            for &j in &u.beams {
                let b = beams[j as usize];
                if b.broken {
                    s.broken = true;
                } else {
                    let r0 = v.beam_rest0[j as usize];
                    s.strain = s.strain.max((b.rest - r0).abs() / r0.abs().max(1e-6));
                }
            }
            s.broken |= s.strain > u.strain;
            s
        })
        .collect()
}

/// Candidate binding triangles of one part, with a coarse grid over their centroids.
struct Binder {
    tris: Vec<[u32; 3]>,
    pos: Vec<Vec3>,
    cell: f32,
    grid: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl Binder {
    fn new(tris: Vec<[u32; 3]>, pos: &[Vec3], cell: f32) -> Self {
        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (ti, t) in tris.iter().enumerate() {
            let c = (pos[t[0] as usize] + pos[t[1] as usize] + pos[t[2] as usize]) / 3.0;
            grid.entry(key(c, cell)).or_default().push(ti);
        }
        Self {
            tris,
            pos: pos.to_vec(),
            cell,
            grid,
        }
    }

    /// The nearest triangle to `p`, and the binding coordinates.
    fn bind(&self, p: Vec3, normal: Vec3) -> Option<([u32; 3], [f32; 3], [f32; 3])> {
        self.bind_d(p, normal).map(|(t, c, n, _)| (t, c, n))
    }

    /// [`Self::bind`] and the distance to the triangle.
    fn bind_d(&self, p: Vec3, normal: Vec3) -> Option<([u32; 3], [f32; 3], [f32; 3], f32)> {
        if self.tris.is_empty() {
            return None;
        }
        let k = key(p, self.cell);
        let mut best: Option<(f32, usize)> = None;
        for r in 1..10 {
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if let Some(list) = self.grid.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                            for &ti in list {
                                let t = self.tris[ti];
                                let d = point_triangle(
                                    p,
                                    self.pos[t[0] as usize],
                                    self.pos[t[1] as usize],
                                    self.pos[t[2] as usize],
                                );
                                if best.is_none_or(|b| d < b.0) {
                                    best = Some((d, ti));
                                }
                            }
                        }
                    }
                }
            }
            // Everything within the searched block's inner distance is found.
            let lo = Vec3::new((k.0 - r) as f32, (k.1 - r) as f32, (k.2 - r) as f32) * self.cell;
            let hi = lo + Vec3::splat((2 * r + 1) as f32 * self.cell);
            let reach = (p - lo).min(hi - p).min_element();
            if best.is_some_and(|b| b.0 <= reach) {
                break;
            }
        }
        let (dist, ti) = best?;
        let t = self.tris[ti];
        let (p0, p1, p2) = (
            self.pos[t[0] as usize],
            self.pos[t[1] as usize],
            self.pos[t[2] as usize],
        );
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let n = e1.cross(e2).normalize_or_zero();
        let d = p - p0;
        let h = d.dot(n);
        let q = d - n * h;
        // Solve q = α e₁ + β e₂ in the plane.
        let (a11, a12, a22) = (e1.dot(e1), e1.dot(e2), e2.dot(e2));
        let (b1, b2) = (q.dot(e1), q.dot(e2));
        let det = a11 * a22 - a12 * a12;
        if det.abs() < 1e-12 {
            return None;
        }
        let alpha = (b1 * a22 - b2 * a12) / det;
        let beta = (a11 * b2 - a12 * b1) / det;
        let tt = e1.normalize();
        let bb = n.cross(tt);
        let nl = [normal.dot(tt), normal.dot(bb), normal.dot(n)];
        Some((t, [alpha, beta, h], nl, dist))
    }
}

/// A visual vertex within this of the skin (m) is on the physics surface:
/// it rides the skin triangle under it, whatever its parts, so it stays on
/// the collision skin however the panels move (the fine skin triangles span
/// the collision skin's nodes). The contact band: what is drawn there is
/// where contact happens. Off the skin (shut faces, returns, inner
/// stampings, trim) a vertex rides its own part's nodes, BeamNG's flexbody
/// binding: three nodes (centre, VX, VY) and an offset along their normal.
/// BeamNG's optional fourth (VZ) node is not used: on the skin the offset is
/// within the band, and off it the frame is the part's own nodes.
pub const SKIN_BAND: f32 = crate::explicit_beams::DEFAULT_COLLISION_RANGE;
/// Score penalty (m) of a node frame that is not one of the skin triangles.
/// A skin triangle is the physics surface itself: a vertex on it stays on
/// it however the panel buckles, and neighbouring vertices stay continuous
/// across its edges. Any other frame of skin nodes is a chord across the
/// surface, off it as soon as the panel folds. So a node frame is used only
/// where the part has no skin triangle within this (a one-row part, a
/// return past the part's last skin triangle). **Choice**: 3 cm.
pub const NODE_FRAME: f32 = 0.03;
/// Score penalty (m) per frame node of another part. A part's frame may take
/// one node across its border (a node sharing a skin triangle with the
/// part: BeamNG flexbodies likewise list their mount's group) only where its
/// own nodes cannot frame the vertex: a one-row part (a bumper's face, a strip
/// between the boot and the bumper) has no triangle of its own. Such a
/// frame stretches when the parts separate, so it must be much nearer.
/// **Choice**: 5 cm.
const FOREIGN: f32 = 0.05;
/// The farthest any visual vertex of the body (wheels and hubs, which ride
/// their rigid tyre and hub, aside) may lie from its binding frame (m): the
/// build fails on any farther, naming the piece, instead of binding it
/// somewhere arbitrary (Rigs of Rods' silent fallback to node 0). A rigid
/// piece's centroid is held to it.
/// A vertex at distance d from its frame moves up to d·θ when the frame
/// turns by θ relative to the panel around the vertex. On the skin d ≤
/// [`SKIN_BAND`]; off it, the deepest inner structure the car's pieces carry
/// over their own part's nodes. **Choice**: 0.3 m, the depth of a door card
/// or a boot floor inside its panel's nodes.
pub const BIND_MAX: f32 = 0.3;
/// [`BIND_MAX`] for the cabin's and bay's pieces on structure groups (dash,
/// console, seats, aprons): their nodes are the structure's members, and
/// trim stands farther off them than a panel's stampings off the panel.
/// **Choice**: 0.45 m, over the dash-top screen 0.40 m above the dash beam's
/// nodes and the rear bench's centroid 0.36 m from its seat frame (both
/// wait on nodes there in the skeleton).
pub const GROUP_BIND_MAX: f32 = 0.45;
/// The farthest a shut-line vertex's own-part fallback (`ShowcaseVisual::alt`)
/// may lie from it where the parts never come apart whole (a bumper's
/// face on the body: past it the shut line tears instead). The fallback
/// carries the vertex only once its shut line has died or torn, when the
/// alternative is a ribbon of skin stretched back to the car (or the vertex
/// dropped). **Choice**: [`GROUP_BIND_MAX`], the reach already accepted for
/// trim on structure nodes. Across a detachable part's border (a door, a
/// lid) the fallback is required and may lie anywhere within
/// [`FRAME_REACH`], as a BeamNG flexbody rides its own group's nodes
/// whatever their distance: the part comes off whole, so its frame turns
/// with the vertex. A door's belt line and a boot lid's rear edge lie
/// 0.30–0.34 m from three of their own nodes (sedan1), the two-tier
/// sedan's rear door glass at Low resolution farther.
pub const ALT_BIND_MAX: f32 = GROUP_BIND_MAX;
/// Node frames are built from a part's this many nearest nodes.
const FRAME_NODES: usize = 10;
/// ... within this radius (m): a frame never spans further.
const FRAME_REACH: f32 = 0.8;
/// A node frame's least shape quality, 2·area / (longest edge)² (a right
/// isosceles triangle 0.5, equilateral 0.87): no slivers, whose normal is
/// ill-defined. BeamNG's flexbody documentation likewise picks VX and VY
/// near right angles to the centre node.
const FRAME_QUALITY: f32 = 0.2;
/// Score penalty (m) for a frame whose plane is perpendicular to the
/// vertex's surface (scaled by 1 − |cos|): of two frames equally near, the
/// one lying along the surface. **Choice**: 1 cm.
const FRAME_TILT: f32 = 0.01;
/// Score per metre of a frame's farthest node from the vertex: of two frames
/// equally near, the more compact (its affine motion is the local one).
/// **Choice**: 0.05 (a frame whose nodes are 20 cm farther must be 1 cm
/// nearer).
const FRAME_SPREAD: f32 = 0.05;

/// Nodes on a coarse grid, for k-nearest queries.
struct NodeGrid {
    ids: Vec<u32>,
    cell: f32,
    grid: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl NodeGrid {
    fn new(ids: Vec<u32>, pos: &[Vec3]) -> Self {
        let cell = 0.15;
        let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for &i in &ids {
            grid.entry(key(pos[i as usize], cell)).or_default().push(i);
        }
        Self { ids, cell, grid }
    }

    /// The nearest [`FRAME_NODES`] nodes within [`FRAME_REACH`], nearest first.
    fn nearest(&self, pos: &[Vec3], p: Vec3) -> Vec<u32> {
        let k = key(p, self.cell);
        let mut found: Vec<(f32, u32)> = Vec::new();
        let rmax = (FRAME_REACH / self.cell).ceil() as i32 + 1;
        for r in 0..=rmax {
            // The shell at Chebyshev radius r.
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                            continue;
                        }
                        if let Some(list) = self.grid.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                            for &i in list {
                                let d = pos[i as usize].distance(p);
                                if d <= FRAME_REACH {
                                    found.push((d, i));
                                }
                            }
                        }
                    }
                }
            }
            // Every node within the shell's inner distance is found.
            let lo = Vec3::new((k.0 - r) as f32, (k.1 - r) as f32, (k.2 - r) as f32) * self.cell;
            let hi = lo + Vec3::splat((2 * r + 1) as f32 * self.cell);
            let reach = (p - lo).min(hi - p).min_element();
            if found.len() >= FRAME_NODES {
                found.sort_by(|a, b| a.0.total_cmp(&b.0));
                if found[FRAME_NODES - 1].0 <= reach {
                    break;
                }
            }
        }
        found.sort_by(|a, b| a.0.total_cmp(&b.0));
        found.truncate(FRAME_NODES);
        found.into_iter().map(|(_, i)| i).collect()
    }
}

/// Binding of one vertex: triangle, (α, β, h), its normal in the frame, and
/// its distance from the triangle.
type Bound = ([u32; 3], [f32; 3], [f32; 3], f32);

/// BeamNG-style flexbody binding, per physics part: a vertex rides three of
/// its own part's nodes (at most one across the part's border), the frame
/// nearest it: the part's skin triangle under it where there is one, else
/// the nearest well-shaped, compact frame of its nodes.
struct PartBinders {
    /// Per part, its nodes and the nodes across its border.
    frames: HashMap<u16, NodeGrid>,
    /// Per part, its own nodes only (the shut-line fallback's frames).
    own: HashMap<u16, NodeGrid>,
    /// Per part, its single-part skin triangles, and those with two of its
    /// nodes (the skin frame candidates).
    tris: HashMap<u16, Binder>,
    border: HashMap<u16, Binder>,
    /// Every skin triangle (sorted node ids), and a binder over them, to
    /// find the part whose skin is under a point.
    skin_set: std::collections::HashSet<[u32; 3]>,
    skin: Binder,
    part_of: Vec<u16>,
    pos: Vec<Vec3>,
    /// Parts held on by hinges and latches (doors, lids): no other part's
    /// frame takes a node of theirs (a BeamNG body flexbody lists only the
    /// body's nodes), or it would stretch after them when they come off.
    detachable: std::collections::HashSet<u16>,
}

impl PartBinders {
    fn new(
        sk: &super::skeleton::Skeleton,
        by_part: &HashMap<u16, Vec<[u32; 3]>>,
        pos: &[Vec3],
        detachable: std::collections::HashSet<u16>,
    ) -> Self {
        let part_of: Vec<u16> = (0..pos.len())
            .map(|i| sk.nodes.get(i).map(|n| n.part).unwrap_or(u16::MAX))
            .collect();
        let mut ids: HashMap<u16, std::collections::BTreeSet<u32>> = HashMap::new();
        for (i, nd) in sk.nodes.iter().enumerate() {
            ids.entry(nd.part).or_default().insert(i as u32);
        }
        let own = ids
            .iter()
            .map(|(&p, v)| (p, NodeGrid::new(v.iter().copied().collect(), pos)))
            .collect();
        // Across the border: nodes sharing a skin triangle with the part.
        for t in &sk.skin {
            for &a in t {
                let pa = part_of[a as usize];
                for &b in t {
                    if part_of[b as usize] != pa && pa != u16::MAX {
                        ids.entry(pa).or_default().insert(b);
                    }
                }
            }
        }
        let sorted = |t: [u32; 3]| {
            let mut s = t;
            s.sort_unstable();
            s
        };
        Self {
            own,
            frames: ids
                .into_iter()
                .map(|(p, v)| (p, NodeGrid::new(v.into_iter().collect(), pos)))
                .collect(),
            tris: by_part
                .iter()
                .map(|(&p, t)| (p, Binder::new(t.clone(), pos, 0.15)))
                .collect(),
            border: {
                let mut m: HashMap<u16, Vec<[u32; 3]>> = HashMap::new();
                for t in &sk.skin {
                    let ps = t.map(|i| part_of[i as usize]);
                    for (k, &p) in ps.iter().enumerate() {
                        let own = ps.iter().filter(|&&q| q == p).count();
                        if own == 2 && ps[..k].iter().all(|&q| q != p) && p != u16::MAX {
                            m.entry(p).or_default().push(*t);
                        }
                    }
                }
                m.into_iter()
                    .map(|(p, t)| (p, Binder::new(t, pos, 0.15)))
                    .collect()
            },
            skin_set: sk.skin.iter().map(|&t| sorted(t)).collect(),
            skin: Binder::new(sk.skin.clone(), pos, 0.15),
            part_of,
            pos: pos.to_vec(),
            detachable,
        }
    }

    fn has(&self, part: u16) -> bool {
        self.frames.get(&part).is_some_and(|g| {
            g.ids
                .iter()
                .filter(|&&i| self.part_of[i as usize] == part)
                .count()
                >= 2
        })
    }

    /// The frame score of triangle `t` for `p` (lower is better), and its
    /// distance from `p`; `None` for a sliver or a frame with fewer than two
    /// of the part's own nodes.
    /// [`Self::bind`] on the part's own nodes only (all three): the frame a
    /// shut line's vertex falls back to when the parts separate.
    fn bind_own(&self, part: u16, p: Vec3, normal: Vec3) -> Option<Bound> {
        let mut cands: Vec<[u32; 3]> = Vec::new();
        if let Some(b) = self.tris.get(&part).and_then(|b| b.bind_d(p, normal)) {
            cands.push(b.0);
        }
        if let Some(g) = self.own.get(&part) {
            triples(&g.nearest(&self.pos, p), &mut cands);
        }
        best_frame(
            &self.pos,
            &cands,
            p,
            normal,
            |t| {
                (t.iter().all(|&i| self.part_of[i as usize] == part))
                    .then(|| self.score(part, t, p, normal, true))
                    .flatten()
            },
            |t| self.is_skin(t),
        )
    }

    /// `strict`: no node of another, detachable part.
    fn score(
        &self,
        part: u16,
        t: [u32; 3],
        p: Vec3,
        normal: Vec3,
        strict: bool,
    ) -> Option<(f32, f32)> {
        let own = t
            .iter()
            .filter(|&&i| self.part_of[i as usize] == part)
            .count();
        if own < 2 {
            return None;
        }
        if strict && self.crosses_detachable(part, t) {
            return None;
        }
        let (score, d) = frame_score(&self.pos, t, p, normal)?;
        let chord = if self.is_skin(t) { 0.0 } else { NODE_FRAME };
        Some((score + FOREIGN * (3 - own) as f32 + chord, d))
    }

    /// `p` on part `part`: the best-scoring frame of the part's nearest nodes
    /// and its nearest skin triangle.
    fn bind(&self, part: u16, p: Vec3, normal: Vec3) -> Option<Bound> {
        let mut cands: Vec<[u32; 3]> = Vec::new();
        for set in [&self.tris, &self.border] {
            if let Some(b) = set.get(&part).and_then(|b| b.bind_d(p, normal)) {
                cands.push(b.0);
            }
        }
        if let Some(b) = self.skin.bind_d(p, normal) {
            cands.push(b.0);
        }
        if let Some(g) = self.frames.get(&part) {
            triples(&g.nearest(&self.pos, p), &mut cands);
        }
        // Skin triangles keep their (outward) winding; other frames are
        // wound so their normal faces the vertex's. A frame with a node of
        // another, detachable part (a door's, a lid's) only where no other
        // frame reaches within BIND_MAX (a body vertex beside a lid at low
        // resolution); such a vertex must have its own part's fallback.
        let pick = |strict: bool| {
            best_frame(
                &self.pos,
                &cands,
                p,
                normal,
                |t| self.score(part, t, p, normal, strict),
                |t| self.is_skin(t),
            )
        };
        match pick(true) {
            Some(b) if b.3 <= BIND_MAX => Some(b),
            strict => match pick(false) {
                Some(l) if strict.is_none_or(|s| l.3 < s.3) => Some(l),
                _ => strict,
            },
        }
    }

    /// Frame `t` of part `part` has a node of another, detachable part.
    fn crosses_detachable(&self, part: u16, t: [u32; 3]) -> bool {
        t.iter().any(|&i| {
            let q = self.part_of[i as usize];
            q != part && self.detachable.contains(&q)
        })
    }

    fn is_skin(&self, t: [u32; 3]) -> bool {
        let mut s = t;
        s.sort_unstable();
        self.skin_set.contains(&s)
    }

    /// Of triangle `t`'s nodes' parts, the one with the most barycentric
    /// weight at coordinates `c`.
    fn major(&self, t: [u32; 3], c: [f32; 3]) -> u16 {
        self.by_weight(t, c)
            .into_iter()
            .find(|e| e.0 != u16::MAX)
            .map_or(u16::MAX, |e| e.0)
    }

    fn by_weight(&self, t: [u32; 3], c: [f32; 3]) -> Vec<(u16, f32)> {
        let w = barycentric_clamped(c[0], c[1]);
        let mut by: Vec<(u16, f32)> = Vec::new();
        for k in 0..3 {
            let pt = self.part_of[t[k] as usize];
            match by.iter_mut().find(|e| e.0 == pt) {
                Some(e) => e.1 += w[k],
                None => by.push((pt, w[k])),
            }
        }
        by.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        by
    }

    /// The part whose skin `p` lies on, with its binding: the nearest skin
    /// triangle (the physics surface, a shut line's mixed triangles
    /// included), and of its nodes' parts the one with the most barycentric
    /// weight at the point nearest `p` (ties: the lower part id).
    fn nearest(&self, p: Vec3, normal: Vec3) -> Option<(u16, Bound)> {
        let (t, c, _, _) = self.skin.bind_d(p, normal)?;
        let by = self.by_weight(t, c);
        by.into_iter()
            .filter(|e| e.0 != u16::MAX)
            .find_map(|(pt, _)| self.bind(pt, p, normal).map(|b| (pt, b)))
    }
}

/// A frame's base score for `p` (lower is better: its distance, then its
/// tilt and spread) and its distance from `p`; `None` for a sliver, whose
/// normal is ill-defined.
fn frame_score(pos: &[Vec3], t: [u32; 3], p: Vec3, normal: Vec3) -> Option<(f32, f32)> {
    let (p0, p1, p2) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
    let cr = (p1 - p0).cross(p2 - p0);
    let longest = (p1 - p0)
        .length_squared()
        .max((p2 - p0).length_squared())
        .max((p2 - p1).length_squared());
    if longest <= 0.0 || cr.length() / longest < FRAME_QUALITY {
        return None;
    }
    let d = point_triangle(p, p0, p1, p2);
    let tilt = 1.0 - cr.normalize().dot(normal).abs();
    let spread = p.distance(p0).max(p.distance(p1)).max(p.distance(p2));
    Some((d + FRAME_TILT * tilt + FRAME_SPREAD * spread, d))
}

/// Every triple of `near`, as candidate frames.
fn triples(near: &[u32], out: &mut Vec<[u32; 3]>) {
    for a in 0..near.len() {
        for b in a + 1..near.len() {
            for c in b + 1..near.len() {
                out.push([near[a], near[b], near[c]]);
            }
        }
    }
}

/// The best-scoring of `cands` for `p`, wound so its normal faces `normal`
/// unless `keep` (an outward-wound skin triangle), with its coordinates and
/// distance.
fn best_frame(
    pos: &[Vec3],
    cands: &[[u32; 3]],
    p: Vec3,
    normal: Vec3,
    score: impl Fn([u32; 3]) -> Option<(f32, f32)>,
    keep: impl Fn([u32; 3]) -> bool,
) -> Option<Bound> {
    let mut best: Option<(f32, [u32; 3], f32)> = None;
    for &t in cands {
        if let Some((s, d)) = score(t) {
            if best.is_none_or(|b| s < b.0) {
                best = Some((s, t, d));
            }
        }
    }
    let (_, t, d) = best?;
    let (p0, p1, p2) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
    let t = if (p1 - p0).cross(p2 - p0).dot(normal) < 0.0 && !keep(t) {
        [t[0], t[2], t[1]]
    } else {
        t
    };
    let (c, nl) = frame_coords(pos, t, p, normal);
    Some((t, c, nl, d))
}

/// BeamNG-style flexbody binding on a structure group (the bay's aprons and
/// bulkhead, the dash and console): a vertex rides the best frame of the
/// group's nearest nodes or of its member triangles. The group's member
/// triangles alone (the old binding) can lie 0.3–0.5 m from a wall built
/// beside them, which then swings when they turn.
struct GroupFrames {
    nodes: NodeGrid,
    tris: Binder,
}

impl GroupFrames {
    fn new(nodes: Vec<u32>, tris: Vec<[u32; 3]>, pos: &[Vec3]) -> Self {
        Self {
            nodes: NodeGrid::new(nodes, pos),
            tris: Binder::new(tris, pos, 0.25),
        }
    }

    fn bind(&self, pos: &[Vec3], p: Vec3, normal: Vec3) -> Option<Bound> {
        let mut cands: Vec<[u32; 3]> = Vec::new();
        if let Some(b) = self.tris.bind_d(p, normal) {
            cands.push(b.0);
        }
        triples(&self.nodes.nearest(pos, p), &mut cands);
        best_frame(
            pos,
            &cands,
            p,
            normal,
            |t| frame_score(pos, t, p, normal),
            |_| false,
        )
    }
}

/// Barycentric weights (1 − α − β, α, β), clamped into the triangle.
fn barycentric_clamped(a: f32, b: f32) -> [f32; 3] {
    let w = [(1.0 - a - b).max(0.0), a.max(0.0), b.max(0.0)];
    let s = (w[0] + w[1] + w[2]).max(1e-9);
    [w[0] / s, w[1] / s, w[2] / s]
}

fn key(p: Vec3, cell: f32) -> (i32, i32, i32) {
    (
        (p.x / cell).floor() as i32,
        (p.y / cell).floor() as i32,
        (p.z / cell).floor() as i32,
    )
}

/// Distance from `p` to triangle (a, b, c) (Ericson, Real-Time Collision
/// Detection, 5.1.5).
fn point_triangle(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length();
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (p - (a + ab * v)).length();
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (p - (a + ac * w)).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (p - (b + (c - b) * w)).length();
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (p - (a + ab * v + ac * w)).length()
}

/// Build the visual skin of a built vehicle.
pub fn showcase_visual(built: &Built) -> Result<ShowcaseVisual, String> {
    showcase_visual_lod(built, 1.0)
}

/// The visual at a level of detail (1 full; 0.45 the distant LOD: the body
/// sampled about 2× coarser, single-step edge rolls, fewer wheel segments).
pub fn showcase_visual_lod(built: &Built, detail: f32) -> Result<ShowcaseVisual, String> {
    super::parts::with_detail(detail, || build_visual(built))
}

#[path = "skin_legacy.rs"]
mod legacy;

/// **Measurement only**: [`showcase_visual_lod`] with the binding as it was
/// before the flexbody rebinding (`skin_legacy.rs`), for a visual A/B on
/// identical physics.
pub fn showcase_visual_legacy_lod(built: &Built, detail: f32) -> Result<ShowcaseVisual, String> {
    super::parts::with_detail(detail, || legacy::build_visual_legacy(built))
}

fn build_visual(built: &Built) -> Result<ShowcaseVisual, String> {
    let def = &built.def;
    let sk = &built.skeleton;
    let paint = built.params.paint.unwrap_or(def.look.paint);
    // The solver's nodes (tyre nodes included), in definition coordinates.
    let sim = ExplicitBeams::from_truck(
        &built.truck,
        Vec3::ZERO,
        Vec3::ZERO,
        ExplicitBeamsSettings::default(),
    )?;
    let node_pos: Vec<Vec3> = sim.nodes.iter().map(|n| n.pos - built.shift).collect();
    let mut pieces = assemble(def, &sedan_styling(def));
    {
        let mut names: Vec<&String> = built.bind_tris.keys().collect();
        names.sort();
        let gi = |n: &str| names.iter().position(|g| *g == n).map(|i| i as u16);
        pieces.extend(super::parts::interior::interior(
            &gi,
            super::parts::interior::Cabin::of(def),
        ));
        let nodes = |n: &str| -> Option<Vec<Vec3>> {
            Some(built.groups.get(n)?.iter().map(|&i| sk.pos(i)).collect())
        };
        pieces.extend(super::parts::bay::engine_bay(&gi, &nodes));
    }
    // Binders: per part, all skin, per group, per tyre and hub.
    let mut by_part: HashMap<u16, Vec<[u32; 3]>> = HashMap::new();
    for t in &sk.skin {
        let p = sk.nodes[t[0] as usize].part;
        if t.iter().all(|&i| sk.nodes[i as usize].part == p) {
            by_part.entry(p).or_default().push(*t);
        }
    }
    let fallback = Binder::new(sk.skin.clone(), &node_pos, 0.15);
    // Single-part skin triangles of every part (votes and reach checks).
    let pure = Binder::new(
        by_part.values().flatten().copied().collect(),
        &node_pos,
        0.15,
    );
    let pb = PartBinders::new(
        sk,
        &by_part,
        &node_pos,
        built.hardware.iter().map(|h| h.part).collect(),
    );
    // Per skin triangle the beams whose break kills it (`skin_death`).
    let mut skin_index: HashMap<[u32; 3], u32> = sk
        .skin
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let mut u = *t;
            u.sort_unstable();
            (u, i as u32)
        })
        .collect();
    let mut coll: HashMap<[u32; 3], usize> = HashMap::new();
    for (i, t) in sim.triangles.iter().enumerate() {
        let mut u = t.nodes;
        u.sort_unstable();
        coll.insert(u, i);
    }
    let mut side: HashMap<(u32, u32), u32> = HashMap::new();
    // Only beams that hold their nodes together in tension: a support
    // (compression only) lets its parts part long before it breaks at its
    // long bound, and triggers and the records stored as beams (torsion,
    // L-beams, pressure and aero triangles) never break.
    for (i, b) in sim.beams.iter().enumerate() {
        use crate::explicit_beams::ExplicitKind as K;
        if !matches!(
            b.kind,
            K::Plain
                | K::LimitedSpring
                | K::Rope
                | K::Shock2
                | K::JbeamPlain
                | K::JbeamBounded
                | K::JbeamAniso
                | K::JbeamPressured
                | K::JbeamRope
        ) {
            continue;
        }
        side.entry((b.a.min(b.b), b.a.max(b.b))).or_insert(i as u32);
    }
    // Per part, one solver beam of each of its hinge and latch groups (up
    // to four): its joint to the car.
    let mut joint: HashMap<u16, Vec<u32>> = HashMap::new();
    for h in &built.hardware {
        if let Some(bm) = h.beams.first().and_then(|&j| sk.beams.get(j)) {
            if let Some(&j) = side.get(&(bm.a.min(bm.b), bm.a.max(bm.b))) {
                let e = joint.entry(h.part).or_default();
                if e.len() < 4 {
                    e.push(j);
                }
            }
        }
    }
    // A frame's death list: its collision triangle's break list (spec §18),
    // else its sides' beams across the part border, else (a node frame
    // across a shut line with no beam along it) its parts' joint: the first
    // hinge or latch to break.
    let side_death = |t: [u32; 3]| -> [u32; 4] {
        let part_of = |i: u32| sk.nodes.get(i as usize).map(|n| n.part);
        let mut u = t;
        u.sort_unstable();
        if let Some(l) = coll.get(&u).and_then(|&ci| sim.tri_breaks.get(ci)) {
            if l.iter().any(|&j| j != u32::MAX) {
                return *l;
            }
        }
        let mut l = [u32::MAX; 4];
        for (k, (a, b)) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])]
            .into_iter()
            .enumerate()
        {
            if part_of(a) != part_of(b) {
                if let Some(&j) = side.get(&(a.min(b), a.max(b))) {
                    l[k] = j;
                }
            }
        }
        if l.iter().all(|&j| j == u32::MAX) {
            if let Some(js) = t
                .iter()
                .filter_map(|&i| part_of(i))
                .find_map(|p| joint.get(&p))
            {
                for (k, &j) in js.iter().enumerate() {
                    l[k] = j;
                }
            }
        }
        l
    };
    let mut death: Vec<[u32; 4]> = sk.skin.iter().map(|&t| side_death(t)).collect();
    let is_shut = |t: [u32; 3]| {
        let ps = t.map(|i| sk.nodes.get(i as usize).map(|n| n.part));
        ps[0] != ps[1] || ps[1] != ps[2]
    };
    // The exterior: a vertex within the contact band of the skin rides the
    // skin triangle under it, whatever its parts: that is the physics
    // surface there (a shut line's triangles included, which the physics
    // breaks and the renderer tears past `TEAR_STRETCH`). The part's own
    // nodes do not reach there: a panel's drawn edge lies up to 0.15 m past
    // its last node row, and riding them leaves it 0.1 m off the collision
    // skin in a crash (measured, wall and pole).
    // A vertex within the contact band of its part's own skin (single-part
    // triangles) rides the triangle under it; an inner stamping, the one
    // under its outer point, else under itself (a return at the shut line).
    let on_skin = |pt: u16, p: Vec3, n: Vec3| {
        pb.tris
            .get(&pt)
            .and_then(|b| b.bind_d(p, n))
            .filter(|b| b.3 <= SKIN_BAND)
    };
    let on_skin_of = |pt: u16, inset: f32, p: Vec3, n: Vec3| {
        if inset > 0.0 {
            on_skin(pt, p - n * inset, -n).or_else(|| on_skin(pt, p, n))
        } else {
            on_skin(pt, p, n)
        }
    };
    // The farthest vertex of a piece from its binding: (distance, where, part).
    let far_of =
        |f: (f32, Vec3, u16), d: f32, p: Vec3, pt: u16| if d > f.0 { (d, p, pt) } else { f };
    let groups: Vec<String> = {
        let mut g: Vec<String> = built.bind_tris.keys().cloned().collect();
        g.sort();
        g
    };
    let tyre_centre: Vec<Vec3> = sim
        .tyres
        .iter()
        .map(|t| 0.5 * (node_pos[t.axle[0] as usize] + node_pos[t.axle[1] as usize]))
        .collect();
    let susp: Vec<u32> = built.groups.get("suspension").cloned().unwrap_or_default();
    let mut out = ShowcaseVisual {
        node_count: node_pos.len() as u32,
        node_rest: sim.nodes.iter().map(|n| n.pos.to_array()).collect(),
        tyre_axles: sim.tyres.iter().map(|t| t.axle).collect(),
        ..Default::default()
    };
    for piece in pieces.iter_mut() {
        piece.mesh.orient_by_normals();
    }
    // A voted piece that reaches well past its physics part (its part's skin
    // 8 cm farther than another's) is cut there, along the smooth zero line of
    // that distance difference (marching triangles), and the far side rides
    // the other part: it parts along a clean line instead of stretching.
    // The part a piece lies on: an area-weighted vote over its triangles (a
    // dense edge roll or shut face must not outvote the panel), each
    // triangle's centroid on the part whose skin is under it.
    let vote = |m: &Mesh| -> Option<u16> {
        let mut votes: HashMap<u16, f32> = HashMap::new();
        let tris = m.idx.len() / 3;
        let step = (tris / 1500).max(1);
        for t in (0..tris).step_by(step) {
            let [a, b, c] = [0, 1, 2].map(|k| m.idx[3 * t + k] as usize);
            let (pa, pb_, pc) = (m.pos[a], m.pos[b], m.pos[c]);
            let area = 0.5 * (pb_ - pa).cross(pc - pa).length();
            let n = (m.nrm[a] + m.nrm[b] + m.nrm[c]).normalize_or_zero();
            if let Some((pt, _)) = pb.nearest((pa + pb_ + pc) / 3.0, n) {
                *votes.entry(pt).or_default() += area;
            }
        }
        votes
            .into_iter()
            .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)))
            .map(|(p, _)| p)
    };
    // An inner stamping `X_inner` is its panel `X`'s (one flexbody group):
    // it takes the panel's part. A vote of its own runs along the panel's
    // edges, over the shut line, and at low resolution elects the body.
    let panel_part: HashMap<String, u16> = pieces
        .iter()
        .filter(|p| pieces.iter().any(|q| q.name == format!("{}_inner", p.name)))
        .filter_map(|p| Some((format!("{}_inner", p.name), vote(&p.mesh)?)))
        .collect();
    let mut queue: Vec<Piece> = pieces;
    let mut pieces: Vec<Piece> = Vec::new();
    let mut rounds = 0;
    while let Some(piece) = queue.pop() {
        rounds += 1;
        if let Bind::Inset(mm) = piece.bind {
            // Vote at the outer skin (the inner panel's normals face in).
            let mut probe = piece.mesh.clone();
            for k in 0..probe.len() {
                probe.pos[k] -= probe.nrm[k] * (mm as f32 * 1e-3);
                probe.nrm[k] = -probe.nrm[k];
            }
            if let Some(pt) = panel_part
                .get(&piece.name)
                .copied()
                .or_else(|| vote(&probe))
            {
                pieces.push(Piece {
                    bind: Bind::PartInset(pt, mm),
                    ..piece
                });
                continue;
            }
        }
        let pt = match piece.bind {
            Bind::Vote if rounds < 400 => vote(&piece.mesh),
            _ => None,
        };
        let Some(pt) = pt else {
            pieces.push(piece);
            continue;
        };
        let g: Vec<f32> = (0..piece.mesh.len())
            .map(|k| {
                let (p, n) = (piece.mesh.pos[k], piece.mesh.nrm[k]);
                match (pb.bind(pt, p, n), pb.nearest(p, n)) {
                    (Some(a), Some((zp, z))) if zp != pt => a.3 - z.3 - 0.08,
                    _ => -1.0,
                }
            })
            .collect();
        if g.iter().all(|&v| v < 0.0) {
            pieces.push(Piece {
                bind: Bind::Part(pt),
                ..piece
            });
            continue;
        }
        let main = super::parts::trim::clip(&piece.mesh, &g, 0.0).mesh;
        let neg: Vec<f32> = g.iter().map(|v| -v).collect();
        let rest = super::parts::trim::clip(&piece.mesh, &neg, 0.0).mesh;
        if !main.is_empty() {
            pieces.push(Piece {
                name: piece.name.clone(),
                mesh: main,
                bind: Bind::Part(pt),
            });
        }
        if !rest.is_empty() {
            queue.push(Piece {
                name: format!("{}+", piece.name),
                mesh: rest,
                bind: Bind::Vote,
            });
        }
    }
    let mut unbound: Vec<String> = Vec::new();
    // Vertices of a detachable part on a frame across its border with no
    // frame of its own part within ALT_BIND_MAX.
    let mut stranded: Vec<String> = Vec::new();
    for piece in &pieces {
        let m = &piece.mesh;
        if m.is_empty() {
            continue;
        }
        let first = out.positions.len() as u32;
        let centroid = m.pos.iter().copied().sum::<Vec3>() / m.len() as f32;
        let nearest_tyre = || {
            (0..tyre_centre.len())
                .min_by(|&a, &b| {
                    tyre_centre[a]
                        .distance(centroid)
                        .total_cmp(&tyre_centre[b].distance(centroid))
                })
                .unwrap_or(0)
        };
        // The binder for this piece.
        let own: Option<Binder> = match piece.bind {
            Bind::Wheel(_) => {
                let t = &sim.tyres[nearest_tyre()];
                let rays = (t.count / 2) as usize;
                let ring =
                    |i: usize, outer: bool| t.first + 2 * (i % rays) as u32 + u32::from(!outer);
                let mut tris = Vec::new();
                for i in 0..rays {
                    tris.push([t.axle[0], ring(i, true), ring(i + 1, true)]);
                    tris.push([t.axle[1], ring(i, false), ring(i + 1, false)]);
                }
                Some(Binder::new(tris, &node_pos, 0.1))
            }
            Bind::Hub(_) => {
                let t = &sim.tyres[nearest_tyre()];
                let (a0, a1) = (t.axle[0], t.axle[1]);
                let axis = (node_pos[a1 as usize] - node_pos[a0 as usize]).normalize_or_zero();
                let c = tyre_centre[nearest_tyre()];
                let k = susp
                    .iter()
                    .copied()
                    .filter(|&i| i != a0 && i != a1)
                    .filter(|&i| {
                        let d = node_pos[i as usize] - c;
                        (d - axis * d.dot(axis)).length() > 0.03
                    })
                    .min_by(|&a, &b| {
                        node_pos[a as usize]
                            .distance(c)
                            .total_cmp(&node_pos[b as usize].distance(c))
                    });
                k.map(|k| Binder::new(vec![[a0, a1, k]], &node_pos, 0.5))
            }
            _ => None,
        };
        // A structure group's piece rides its group's nodes.
        let group_frames: Option<GroupFrames> = match piece.bind {
            Bind::Group(g) | Bind::Groups([g, ..]) => {
                let gs = match piece.bind {
                    Bind::Groups(gs) => gs,
                    _ => [g, u16::MAX, u16::MAX],
                };
                let mut tris: Vec<[u32; 3]> = Vec::new();
                let mut nodes: Vec<u32> = Vec::new();
                for g in gs.into_iter().filter(|&g| g != u16::MAX) {
                    let Some(n) = groups.get(g as usize) else {
                        continue;
                    };
                    tris.extend(built.bind_tris.get(n).into_iter().flatten().copied());
                    nodes.extend(built.groups.get(n).into_iter().flatten().copied());
                }
                nodes.extend(tris.iter().flatten().copied());
                nodes.sort_unstable();
                nodes.dedup();
                (!nodes.is_empty()).then(|| GroupFrames::new(nodes, tris, &node_pos))
            }
            _ => None,
        };
        // A vote for the part.
        let group_rigid = match piece.bind {
            Bind::GroupRigid(g) => groups.get(g as usize).and_then(|n| {
                let tris = built.bind_tris.get(n)?.clone();
                let mut nodes: Vec<u32> = built.groups.get(n).cloned().unwrap_or_default();
                nodes.extend(tris.iter().flatten().copied());
                nodes.sort_unstable();
                nodes.dedup();
                Some(GroupFrames::new(nodes, tris, &node_pos))
            }),
            _ => None,
        };
        let rigid = piece.bind == Bind::Rigid || group_rigid.is_some();
        // An inner stamping binds where the outer panel over it binds: each
        // vertex picks its triangle at the outer skin (moved back out by its
        // depth), then is placed on that triangle at its own depth.
        let inset = if let Bind::PartInset(_, mm) = piece.bind {
            mm as f32 * 1e-3
        } else {
            0.0
        };
        let part: Option<u16> = if let Bind::Part(p) | Bind::PartInset(p, _) = piece.bind {
            Some(p)
        } else if piece.bind == Bind::Vote || rigid {
            vote(m)
        } else {
            None
        };
        if rigid {
            // One triangle (the voted part's nearest to the centroid) carries
            // the whole part rigidly.
            // The group's best frame at the piece's centroid.
            let b = group_rigid
                .as_ref()
                .and_then(|g| g.bind(&node_pos, centroid, Vec3::Y))
                .map(|b| (b.0, b.1, b.2))
                .or_else(|| {
                    // The part's best frame at the piece's centroid (compact,
                    // nearest: a small part's mount).
                    let n = m.nrm.iter().copied().sum::<Vec3>().normalize_or_zero();
                    part.and_then(|pt| pb.bind(pt, centroid, n))
                        .map(|b| (b.0, b.1, b.2))
                })
                .or_else(|| pure.bind(centroid, Vec3::Y));
            let Some((t, _, _)) = b else {
                return Err(format!(
                    "no skin triangle for the rigid part {}",
                    piece.name
                ));
            };
            let (p0, p1, p2) = (
                node_pos[t[0] as usize],
                node_pos[t[1] as usize],
                node_pos[t[2] as usize],
            );
            // A rigid piece moves with its frame: its centroid must be
            // within reach of it.
            let d = point_triangle(centroid, p0, p1, p2);
            if d > if group_rigid.is_some() {
                GROUP_BIND_MAX
            } else {
                BIND_MAX
            } {
                unbound.push(format!(
                    "{} (rigid, centroid {:.0} mm from its frame)",
                    piece.name,
                    d * 1e3
                ));
            }
            let tt = (p1 - p0).normalize();
            let nn = (p1 - p0).cross(p2 - p0).normalize();
            let bb = nn.cross(tt);
            for k in 0..m.len() {
                let d = m.pos[k] - p0;
                let n = m.nrm[k];
                out.bind_nodes.push(t);
                out.bind_coords.push([d.dot(tt), d.dot(bb), d.dot(nn)]);
                out.bind_normal.push([n.dot(tt), n.dot(bb), n.dot(nn)]);
                out.bind_rest.push(rest_edges(&node_pos, t));
                out.positions.push((m.pos[k] + built.shift).to_array());
                out.normals.push(n.to_array());
                out.uvs.push(m.uv[k]);
                out.material.push(m.mat[k]);
                out.colors.push(color(m.mat[k], paint));
                out.ao.push(m.ao[k].clamp(0.0, 1.0));
                out.smooth.push(0);
                out.rigid.push(1);
                out.alt.push(0);
                out.part.push(part.unwrap_or(u16::MAX));
                out.wheel.push(u32::MAX);
            }
            out.indices.extend(m.idx.iter().map(|i| i + first));
            out.pieces.push((piece.name.clone(), first, m.len() as u32));
            continue;
        }
        // Per vertex: its binding and its part. A voted piece rides its part,
        // except where that part's skin is much farther than another's (the
        // piece reaches past the physics panel): those vertices ride the
        // nearer part, and triangles are split along that boundary (vertices
        // duplicated), so the piece parts there instead of stretching.
        type B = ([u32; 3], [f32; 3], [f32; 3]);
        let mut vb: Vec<(B, u16)> = Vec::with_capacity(m.len());
        // The piece's farthest vertex from its binding (skin-riding pieces).
        let mut far = (0.0f32, Vec3::ZERO, u16::MAX);
        for k in 0..m.len() {
            let (p, n) = (m.pos[k], m.nrm[k]);
            let r: Option<(B, u16)> = if let Some(o) = &own {
                o.bind(p, n).map(|b| (b, u16::MAX))
            } else if let Some(g) = &group_frames {
                g.bind(&node_pos, p, n).map(|b| {
                    far = far_of(far, b.3, p, u16::MAX);
                    ((b.0, b.1, b.2), u16::MAX)
                })
            } else if let Some((pt, b)) = part.and_then(|pt| {
                // On its part's own skin: ride the triangle under it.
                on_skin_of(pt, inset, p, n).map(|b| (pt, b))
            }) {
                far = far_of(far, b.3, p, pt);
                let (c, nl) = frame_coords(&node_pos, b.0, p, n);
                Some(((b.0, c, nl), pt))
            } else if let Some(pt) = part {
                let (pp, np) = if inset > 0.0 {
                    (p - n * inset, -n)
                } else {
                    (p, n)
                };
                let at = |r: Option<([u32; 3], [f32; 3], [f32; 3], f32)>| {
                    r.map(|(t, c, nl, d)| {
                        if inset > 0.0 {
                            let (c, nl) = frame_coords(&node_pos, t, p, n);
                            (t, c, nl, d)
                        } else {
                            (t, c, nl, d)
                        }
                    })
                };
                // An inner stamping off the skin rides its panel's part
                // (its flexbody group): only an outer surface reaching past
                // its physics part is cut to the neighbour.
                let near = if inset > 0.0 {
                    None
                } else {
                    pb.nearest(pp, np)
                };
                let zpart = near.map(|z| z.0);
                let near = at(near.map(|z| z.1));
                let mine = at(pb.bind(pt, pp, np));
                match (mine, near) {
                    (Some(a), Some(z)) => {
                        let zp = zpart.unwrap_or(u16::MAX);
                        // 1.5 cm more than the vote's cut (8 cm): a piece cut
                        // there has vertices on the cut line itself, which
                        // must stay on their side, or the line opens as a
                        // sawtooth of triangles riding alternate parts.
                        if a.3 > z.3 + 0.095 && zp != pt {
                            far = far_of(far, z.3, p, zp);
                            Some(((z.0, z.1, z.2), zp))
                        } else {
                            far = far_of(far, a.3, p, pt);
                            Some(((a.0, a.1, a.2), pt))
                        }
                    }
                    (Some(a), None) => {
                        far = far_of(far, a.3, p, pt);
                        Some(((a.0, a.1, a.2), pt))
                    }
                    (None, Some(z)) => {
                        far = far_of(far, z.3, p, zpart.unwrap_or(u16::MAX));
                        Some(((z.0, z.1, z.2), zpart.unwrap_or(u16::MAX)))
                    }
                    _ => None,
                }
            } else {
                None
            };
            let r = r.or_else(|| {
                fallback.bind_d(p, n).map(|b| {
                    far = far_of(far, b.3, p, u16::MAX);
                    ((b.0, b.1, b.2), u16::MAX)
                })
            });
            let Some(r) = r else {
                return Err(format!("no skin triangle for a vertex at {p:?}"));
            };
            vb.push(r);
        }
        let mut idx: Vec<u32> = m.idx.clone();
        let mut extra: Vec<(usize, B, u16)> = Vec::new();
        if part.is_some() {
            let mut dup: HashMap<(u32, u16), u32> = HashMap::new();
            // Each vertex's binding on a part, and its distance.
            let mut cache: HashMap<(usize, u16), (B, f32)> = HashMap::new();
            for t in idx.chunks_mut(3) {
                let parts = [
                    vb[t[0] as usize].1,
                    vb[t[1] as usize].1,
                    vb[t[2] as usize].1,
                ];
                if parts[0] == parts[1] && parts[1] == parts[2] {
                    continue;
                }
                // The triangle rides one part: the majority's, else (its
                // nodes beyond [`BIND_MAX`] of a vertex: a one-row part at
                // low resolution) another of its parts.
                let major = if parts[1] == parts[2] {
                    parts[1]
                } else {
                    parts[0]
                };
                let mut order: Vec<u16> = vec![major];
                order.extend(parts.iter().copied().filter(|&q| q != major));
                order.dedup();
                let mut rebind = |k: usize, tp: u16| -> Option<(B, f32)> {
                    if let Some(&(b, d)) = cache.get(&(k, tp)) {
                        return Some((b, d));
                    }
                    let (pp, np) = if inset > 0.0 {
                        (m.pos[k] - m.nrm[k] * inset, -m.nrm[k])
                    } else {
                        (m.pos[k], m.nrm[k])
                    };
                    // On the skin it keeps the skin triangle under it.
                    let r = on_skin_of(tp, inset, m.pos[k], m.nrm[k])
                        .or_else(|| pb.bind(tp, pp, np))
                        .map(|(t, _, _, d)| {
                            let (c, nl) = frame_coords(&node_pos, t, m.pos[k], m.nrm[k]);
                            ((t, c, nl), d)
                        });
                    if let Some(r) = r {
                        cache.insert((k, tp), r);
                    }
                    r
                };
                let fits = |tp: u16, rebind: &mut dyn FnMut(usize, u16) -> Option<(B, f32)>| {
                    pb.has(tp)
                        && t.iter().all(|&v| {
                            vb[v as usize].1 == tp
                                || rebind(v as usize, tp).is_some_and(|(_, d)| d <= BIND_MAX)
                        })
                };
                let Some(tp) = order.into_iter().find(|&tp| fits(tp, &mut rebind)) else {
                    continue;
                };
                for v in t.iter_mut() {
                    if vb[*v as usize].1 == tp {
                        continue;
                    }
                    let key = (*v, tp);
                    let nv = if let Some(&nv) = dup.get(&key) {
                        nv
                    } else {
                        let k = *v as usize;
                        let (b, d) = rebind(k, tp).expect("fits");
                        far = far_of(far, d, m.pos[k], tp);
                        let nv = (m.len() + extra.len()) as u32;
                        extra.push((k, b, tp));
                        dup.insert(key, nv);
                        nv
                    };
                    *v = nv;
                }
            }
        }
        let all = (0..m.len())
            .map(|k| (k, vb[k].0, vb[k].1))
            .chain(extra.iter().map(|&(k, b, tp)| (k, b, tp)));
        for (k, (t, c, nl), tp) in all {
            // On a shut line's skin triangle: its own part's frame too.
            let mut alt = 0;
            let mut u = t;
            u.sort_unstable();
            // A frame across a part border (a shut line's skin triangle, or
            // a node frame with a node across it): its death list's index.
            let si = is_shut(t).then(|| {
                *skin_index.entry(u).or_insert_with(|| {
                    death.push(side_death(t));
                    death.len() as u32 - 1
                })
            });
            if let Some(si) = si {
                let pt = if tp != u16::MAX { Some(tp) } else { part };
                // Within ALT_BIND_MAX: a one-row part (a bumper's face) has
                // no frame of its own there, and its shut line tears instead.
                // Across a detachable part's border (a door, a lid) there
                // must be one, anywhere within FRAME_REACH: the part comes
                // off whole and its edge goes with it (or stays with the car).
                let detach =
                    pt.is_some_and(|q| pb.detachable.contains(&q) || pb.crosses_detachable(q, t));
                let reach = if detach { FRAME_REACH } else { ALT_BIND_MAX };
                if let Some((at, _, _, _)) = pt
                    .and_then(|pt| pb.bind_own(pt, m.pos[k], m.nrm[k]))
                    .filter(|b| b.3 <= reach)
                {
                    let (ac2, an2) = frame_coords(&node_pos, at, m.pos[k], m.nrm[k]);
                    out.alt_nodes.push(at);
                    out.alt_coords.push(ac2);
                    out.alt_normal.push(an2);
                    out.alt_rest.push(rest_edges(&node_pos, at));
                    out.alt_tri.push(si);
                    alt = out.alt_nodes.len() as u32;
                } else if let Some(pt) = pt.filter(|_| detach) {
                    stranded.push(format!(
                        "{} on {} at {:.2?}",
                        piece.name,
                        sk.parts.get(pt as usize).map_or("-", |s| s.as_str()),
                        m.pos[k]
                    ));
                }
            }
            out.alt.push(alt);
            out.wheel.push(match piece.bind {
                Bind::Wheel(_) => (nearest_tyre() as u32) << 1,
                Bind::Hub(_) => ((nearest_tyre() as u32) << 1) | 1,
                _ => u32::MAX,
            });
            out.part.push(if tp != u16::MAX {
                tp
            } else {
                part.unwrap_or(u16::MAX)
            });
            out.bind_nodes.push(t);
            out.bind_coords.push(c);
            out.bind_normal.push(nl);
            out.bind_rest.push(rest_edges(&node_pos, t));
            out.positions.push((m.pos[k] + built.shift).to_array());
            out.normals.push(m.nrm[k].to_array());
            out.uvs.push(m.uv[k]);
            out.material.push(m.mat[k]);
            out.colors.push(color(m.mat[k], paint));
            out.ao.push(m.ao[k].clamp(0.0, 1.0));
            // Smooth shading from node normals only where the piece rides the
            // body's skin (not on wheels or frames).
            out.smooth.push(if own.is_none() && group_frames.is_none() {
                m.smooth[k]
            } else {
                0
            });
            out.rigid.push(0);
        }
        let limit = if group_frames.is_some() {
            GROUP_BIND_MAX
        } else {
            BIND_MAX
        };
        if far.0 > limit {
            unbound.push(format!(
                "{} ({:.0} mm at {:.2?} on {})",
                piece.name,
                far.0 * 1e3,
                far.1,
                sk.parts.get(far.2 as usize).map_or("-", |s| s.as_str())
            ));
        }
        let count = out.positions.len() as u32 - first;
        out.indices.extend(idx.iter().map(|i| i + first));
        out.pieces.push((piece.name.clone(), first, count));
    }
    if !unbound.is_empty() {
        return Err(format!(
            "visual vertices farther from their nodes than BIND_MAX ({} mm) or, on structure groups, GROUP_BIND_MAX ({} mm): {}",
            BIND_MAX * 1e3,
            GROUP_BIND_MAX * 1e3,
            unbound.join(", ")
        ));
    }
    if !stranded.is_empty() {
        let n = stranded.len();
        stranded.truncate(8);
        return Err(format!(
            "{n} visual vertices ride a frame across a detachable part's border with no frame of their own part within FRAME_REACH ({} mm), e.g. {}",
            FRAME_REACH * 1e3,
            stranded.join(", ")
        ));
    }
    out.skin_death = death;
    // Node normals for smooth shading.
    out.normal_tris = sk.skin.clone();
    let nn = node_pos.len();
    let mut count = vec![0u32; nn + 1];
    for t in &out.normal_tris {
        for &i in t {
            count[i as usize + 1] += 1;
        }
    }
    for i in 0..nn {
        count[i + 1] += count[i];
    }
    let mut fill = count.clone();
    let mut list = vec![0u32; *count.last().unwrap() as usize];
    for (ti, t) in out.normal_tris.iter().enumerate() {
        for &i in t {
            list[fill[i as usize] as usize] = ti as u32;
            fill[i as usize] += 1;
        }
    }
    out.node_tri_start = count;
    out.node_tri_list = list;
    out.node_rest_normal = node_normals(&out, &node_pos)
        .iter()
        .map(|n| n.to_array())
        .collect();
    damage_units(built, &sim, &node_pos, &pure, &mut out);
    Ok(out)
}

/// The damage units, the per-node beam table and node materials.
///
/// Interim derivation (until the physics' own break groups land in `Built`:
/// engine/docs/beamng-realism-plan.md P1, P5, P6): a glass pane's unit is its
/// glass zone's beams (both nodes in the zone, on the pane's side); a lamp's,
/// a mirror's and a pane without a glass zone (the quarter light) is its
/// mount: the solver beams along the skin triangles under it.
fn damage_units(
    built: &Built,
    sim: &ExplicitBeams,
    node_pos: &[Vec3],
    pure: &Binder,
    out: &mut ShowcaseVisual,
) {
    use super::vehicle::Region;
    let sk = &built.skeleton;
    let def = &built.def;
    let nn = node_pos.len();
    let mut tyre_node = vec![false; nn];
    for t in &sim.tyres {
        for i in t.first..t.first + t.count {
            tyre_node[i as usize] = true;
        }
        for &a in &t.axle {
            tyre_node[a as usize] = true;
        }
    }
    // Beams by node pair, and per node (CSR, no tyre beams).
    let mut by_pair: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    let mut count = vec![0u32; nn + 1];
    for (j, b) in sim.beams.iter().enumerate() {
        by_pair
            .entry((b.a.min(b.b), b.a.max(b.b)))
            .or_default()
            .push(j as u32);
        if !tyre_node[b.a as usize] && !tyre_node[b.b as usize] {
            count[b.a as usize + 1] += 1;
            count[b.b as usize + 1] += 1;
        }
    }
    for i in 0..nn {
        count[i + 1] += count[i];
    }
    let mut fill = count.clone();
    let mut list = vec![0u32; count[nn] as usize];
    for (j, b) in sim.beams.iter().enumerate() {
        if !tyre_node[b.a as usize] && !tyre_node[b.b as usize] {
            for n in [b.a, b.b] {
                list[fill[n as usize] as usize] = j as u32;
                fill[n as usize] += 1;
            }
        }
    }
    out.node_beam_start = count;
    out.node_beam_list = list;
    out.beam_rest0 = sim.beams.iter().map(|b| b.rest0).collect();
    // Glass zones: per shell node its (zone, side) (side 0 for a zone across
    // the car, ±1 for a side zone).
    let len = def.profile.length;
    let glass: Vec<(usize, bool)> = def
        .zones
        .iter()
        .enumerate()
        .filter(|(_, z)| z.grade == "glass")
        .map(|(k, z)| (k, matches!(z.region, Region::Side(_))))
        .collect();
    let mut zone_of: HashMap<u32, (usize, i8)> = HashMap::new();
    for &i in &built.shell.nodes {
        let p = sk.pos(i);
        let Some(seg) = built.shell.seg.get(&i) else {
            continue;
        };
        if let Some(&(k, sided)) = glass
            .iter()
            .find(|(k, _)| def.zones[*k].region.contains(p, *seg, len))
        {
            let side = if sided {
                if p.x >= 0.0 {
                    1
                } else {
                    -1
                }
            } else {
                0
            };
            zone_of.insert(i, (k, side));
        }
    }
    let mut zone_beams: HashMap<(usize, i8), Vec<u32>> = HashMap::new();
    for (j, b) in sim.beams.iter().enumerate() {
        if let (Some(za), Some(zb)) = (zone_of.get(&b.a), zone_of.get(&b.b)) {
            if za == zb {
                zone_beams.entry(*za).or_default().push(j as u32);
            }
        }
    }
    let zone_centre = |key: &(usize, i8)| -> Vec3 {
        let ns: Vec<Vec3> = zone_of
            .iter()
            .filter(|(_, z)| *z == key)
            .map(|(&i, _)| node_pos[i as usize])
            .collect();
        ns.iter().copied().sum::<Vec3>() / ns.len().max(1) as f32
    };
    let centres: Vec<((usize, i8), Vec3)> =
        zone_beams.keys().map(|k| (*k, zone_centre(k))).collect();
    // Node materials.
    out.node_material = (0..nn)
        .map(|i| {
            if tyre_node[i] {
                node_mat::RUBBER
            } else if zone_of.contains_key(&(i as u32)) {
                node_mat::GLASS
            } else if i < sk.nodes.len()
                && sk.parts[sk.nodes[i].part as usize].starts_with("bumper")
            {
                node_mat::PLASTIC
            } else {
                node_mat::METAL
            }
        })
        .collect();
    // Units from the pieces.
    out.unit = vec![u32::MAX; out.positions.len()];
    out.shard = vec![0; out.positions.len()];
    let mut by_name: HashMap<String, u32> = HashMap::new();
    let pieces = out.pieces.clone();
    for (name, first, cnt) in &pieces {
        let base = name.trim_end_matches('+');
        let kind = if base == "windscreen" {
            unit_kind::LAMINATED
        } else if base == "backlight" || base.starts_with("glass_side") {
            unit_kind::TEMPERED
        } else if base.starts_with("headlight")
            || base.starts_with("taillight")
            || base.starts_with("foglamp")
            || base == "lightbar"
        {
            unit_kind::LAMP
        } else if base.starts_with("mirror") {
            unit_kind::MIRROR
        } else {
            continue;
        };
        let (first, cnt) = (*first as usize, *cnt as usize);
        let centroid = (first..first + cnt)
            .map(|k| Vec3::from_array(out.positions[k]) - built.shift)
            .sum::<Vec3>()
            / cnt.max(1) as f32;
        // The physics' own pane units (Built.damage_units: glass_windscreen,
        // glass_backlight, glass_side_{f,r,q}_{l,r}, each the pane's beams
        // and its mounts, broken when one of them breaks), mapped to the
        // solver's beams by node pair.
        let physics_unit = matches!(kind, unit_kind::TEMPERED | unit_kind::LAMINATED)
            .then(|| {
                let want = if base.starts_with("glass_") {
                    base.to_string()
                } else {
                    format!("glass_{base}")
                };
                built.damage_units.iter().find(|u| u.name == want)
            })
            .flatten()
            .map(|u| {
                let mut js: Vec<u32> = u
                    .beams
                    .iter()
                    .filter_map(|&j| sk.beams.get(j as usize))
                    .filter_map(|b| by_pair.get(&(b.a.min(b.b), b.a.max(b.b))))
                    .flatten()
                    .copied()
                    .collect();
                js.sort_unstable();
                js.dedup();
                js
            });
        let uid = *by_name.entry(base.to_string()).or_insert_with(|| {
            if let Some(beams) = physics_unit.clone() {
                out.units.push(DamageUnit {
                    name: base.to_string(),
                    kind,
                    beams,
                    strain: f32::INFINITY,
                    shards: 0,
                });
                return (out.units.len() - 1) as u32;
            }
            let glassy = matches!(kind, unit_kind::TEMPERED | unit_kind::LAMINATED);
            // A pane on its glass zone; otherwise (and every lamp and mirror)
            // its mount.
            let zone = glassy
                .then(|| {
                    centres
                        .iter()
                        .map(|(k, c)| (*k, c.distance(centroid)))
                        .filter(|(_, d)| *d < 0.4)
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|(k, _)| k)
                })
                .flatten();
            let beams = if let Some(z) = zone {
                zone_beams[&z].clone()
            } else {
                let mut set = std::collections::BTreeSet::new();
                let step = (cnt / 300).max(1);
                for k in (first..first + cnt).step_by(step) {
                    let p = Vec3::from_array(out.positions[k]) - built.shift;
                    if let Some((t, _, _)) = pure.bind(p, Vec3::Y) {
                        for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                            if let Some(js) = by_pair.get(&(a.min(b), a.max(b))) {
                                set.extend(js.iter().copied());
                            }
                        }
                    }
                }
                set.into_iter().collect()
            };
            // A pane on its own glass beams breaks at glass's strain; a pane
            // without them (the quarter light) only when its mount tears
            // (the body panel's beams' strain is not the glass's).
            let strain = match kind {
                unit_kind::TEMPERED if zone.is_some() => TEMPERED_BREAK_STRAIN,
                unit_kind::LAMP => LENS_BREAK_STRAIN,
                _ => f32::INFINITY,
            };
            out.units.push(DamageUnit {
                name: base.to_string(),
                kind,
                beams,
                strain,
                shards: 0,
            });
            (out.units.len() - 1) as u32
        });
        for k in first..first + cnt {
            out.unit[k] = uid;
        }
    }
    // Shard seeds: over each breaking unit's glass, lens and mirror faces,
    // area × SHARD_DENSITY of them (area weighted, one per marked vertex).
    let mut acc = vec![0.0f32; out.units.len()];
    for t in out.indices.chunks(3) {
        let (a, b, c) = (t[0] as usize, t[1] as usize, t[2] as usize);
        let u = out.unit[a];
        if u == u32::MAX || out.units[u as usize].kind == unit_kind::LAMINATED {
            continue;
        }
        if !matches!(
            out.material[a],
            mat::GLASS | mat::PRIVACY | mat::LENS | mat::TAIL_LENS | mat::CHROME
        ) {
            continue;
        }
        let (pa, pb, pc) = (
            Vec3::from_array(out.positions[a]),
            Vec3::from_array(out.positions[b]),
            Vec3::from_array(out.positions[c]),
        );
        acc[u as usize] += 0.5 * (pb - pa).cross(pc - pa).length() * SHARD_DENSITY;
        while acc[u as usize] >= 1.0 {
            let Some(&v) = [a, b, c].iter().find(|&&v| out.shard[v] == 0) else {
                break;
            };
            out.shard[v] = 1;
            out.units[u as usize].shards += 1;
            acc[u as usize] -= 1.0;
        }
    }
}

/// A point's binding coordinates (α, β, h) and its normal's (in the frame)
/// on triangle `t`.
pub fn frame_coords(pos: &[Vec3], t: [u32; 3], p: Vec3, normal: Vec3) -> ([f32; 3], [f32; 3]) {
    let (p0, p1, p2) = (pos[t[0] as usize], pos[t[1] as usize], pos[t[2] as usize]);
    let e1 = p1 - p0;
    let e2 = p2 - p0;
    let n = e1.cross(e2).normalize_or_zero();
    let d = p - p0;
    let h = d.dot(n);
    let q = d - n * h;
    let (a11, a12, a22) = (e1.dot(e1), e1.dot(e2), e2.dot(e2));
    let (b1, b2) = (q.dot(e1), q.dot(e2));
    let det = (a11 * a22 - a12 * a12).max(1e-12);
    let tt = e1.normalize_or_zero();
    let bb = n.cross(tt);
    (
        [(b1 * a22 - b2 * a12) / det, (a11 * b2 - a12 * b1) / det, h],
        [normal.dot(tt), normal.dot(bb), normal.dot(n)],
    )
}

fn rest_edges(pos: &[Vec3], t: [u32; 3]) -> [f32; 2] {
    let p0 = pos[t[0] as usize];
    [
        pos[t[1] as usize].distance(p0),
        pos[t[2] as usize].distance(p0),
    ]
}

/// A material's base colour (linear RGB).
pub fn color(m: u8, paint: [f32; 3]) -> [f32; 3] {
    match m {
        mat::PAINT => paint,
        mat::GLASS => [0.02, 0.025, 0.03],
        mat::CHROME => [0.75, 0.76, 0.78],
        mat::TRIM => [0.02, 0.02, 0.022],
        mat::HEADLIGHT => [0.85, 0.88, 0.9],
        mat::TAILLIGHT => [0.5, 0.015, 0.015],
        mat::TYRE => [0.025, 0.025, 0.025],
        // Gunmetal (the spokes' sides and the barrel's lip).
        mat::RIM => [0.10, 0.105, 0.115],
        // A shut line's wall: the paint, in shadow.
        mat::GAP => [paint[0] * 0.55, paint[1] * 0.55, paint[2] * 0.55],
        mat::UNDERBODY => [0.04, 0.04, 0.04],
        mat::PLATE => [0.85, 0.85, 0.8],
        mat::GRILLE => [0.015, 0.015, 0.015],
        mat::SEAT => [0.05, 0.045, 0.04],
        mat::DASH => [0.03, 0.03, 0.032],
        mat::DOORCARD => [0.1, 0.095, 0.09],
        mat::HEADLINER => [0.42, 0.4, 0.37],
        mat::CARPET => [0.05, 0.05, 0.05],
        mat::LENS => [0.9, 0.92, 0.95],
        mat::DRL => [1.0, 0.98, 0.95],
        mat::REFLECTOR => [0.8, 0.8, 0.82],
        mat::TAIL_LENS => [0.55, 0.02, 0.02],
        mat::TAIL_LED => [1.0, 0.06, 0.04],
        mat::BRAKE => [0.32, 0.32, 0.33],
        mat::CALIPER => [0.5, 0.03, 0.02],
        mat::TAIL_BACK => [0.12, 0.01, 0.01],
        mat::SCREEN => [0.02, 0.025, 0.03],
        mat::DISPLAY => [0.02, 0.03, 0.05],
        mat::PRIVACY => [0.01, 0.012, 0.014],
        mat::MACHINED => [0.80, 0.81, 0.83],
        mat::PILLAR => [0.018, 0.018, 0.02],
        // Inner stampings are painted in the body colour (jambs, the bay,
        // under the lids), satin and a shade darker.
        mat::INNER => [paint[0] * 0.8, paint[1] * 0.8, paint[2] * 0.8],
        mat::ENGINE => [0.16, 0.165, 0.17],
        _ => [1.0, 0.0, 1.0],
    }
}

/// The most the crease envelope may move a vertex off its binding triangle
/// beyond its rest offset (m): the physics surface is the flat skin
/// triangles, and beyond the contact band what is drawn is not where contact
/// happens. **Choice**: half the contact band ([`SKIN_BAND`]), the other
/// half left to a skin vertex's own rest offset (|h| ≤ the band) as its
/// triangle turns. It was half the triangle (5–10 cm), which drew crushed
/// panels up to that far off the collision skin. The renderer's
/// `ENVELOPE_MAX` is the same.
pub const ENVELOPE_MAX: f32 = 0.5 * SKIN_BAND;
/// Crease sharpness: the envelope's temperature over the binding triangle's
/// size L. A fold whose two sides' planes differ by θ switches from one to the
/// other over about `CREASE_SHARP · L / sin θ` (a 60° fold of a 10 cm
/// triangle: 1 cm); a bend of a few degrees stays smooth.
pub const CREASE_SHARP: f32 = 0.08;
/// A node's skin splits into two planes (a fold runs through the node) where
/// its most deviant triangle is more than 20° from its mean normal.
pub const SPLIT_COS: f32 = 0.939_692_6;

/// A node's tangent planes ([`node_planes`]): the first normal, and the
/// second where its skin triangles fall either side of a fold.
#[derive(Clone, Copy, Debug, Default)]
pub struct NodePlanes {
    pub a: Vec3,
    pub b: Option<Vec3>,
}

/// Every solver node's tangent planes from the skin triangles at `nodes`
/// (the renderer's `nodePlanes()`): its mean skin normal, or, where its most
/// deviant triangle is more than [`SPLIT_COS`] from that mean, two: the
/// triangles nearer that one, and those nearer the one most opposite it. A
/// node on a fold line so carries both panels' planes, not a bevel between
/// them. Zero off the skin.
pub fn node_planes(v: &ShowcaseVisual, nodes: &[Vec3]) -> Vec<NodePlanes> {
    let n = nodes.len().min(v.node_tri_start.len().saturating_sub(1));
    let mut out = vec![NodePlanes::default(); nodes.len()];
    for (i, o) in out.iter_mut().enumerate().take(n) {
        let tris = &v.node_tri_list[v.node_tri_start[i] as usize..v.node_tri_start[i + 1] as usize];
        let cr: Vec<Vec3> = tris
            .iter()
            .map(|&ti| {
                let t = v.normal_tris[ti as usize];
                let a = nodes[t[0] as usize];
                (nodes[t[1] as usize] - a).cross(nodes[t[2] as usize] - a)
            })
            .collect();
        let acc: Vec3 = cr.iter().copied().sum();
        if acc.length_squared() < 1e-24 {
            continue;
        }
        let m = (acc + Vec3::splat(1e-12)).normalize();
        let unit = |c: &Vec3| (c.length() >= 1e-12).then(|| *c / c.length());
        let mut lo = 2.0;
        let mut sb = m;
        for u in cr.iter().filter_map(unit) {
            if u.dot(m) < lo {
                lo = u.dot(m);
                sb = u;
            }
        }
        if lo > SPLIT_COS {
            *o = NodePlanes { a: m, b: None };
            continue;
        }
        let mut hi = 2.0;
        let mut sa = m;
        for u in cr.iter().filter_map(unit) {
            if u.dot(sb) < hi {
                hi = u.dot(sb);
                sa = u;
            }
        }
        let (mut aa, mut ab) = (Vec3::ZERO, Vec3::ZERO);
        for c in &cr {
            if c.dot(sa) >= c.dot(sb) {
                aa += *c;
            } else {
                ab += *c;
            }
        }
        *o = NodePlanes {
            a: (aa + Vec3::splat(1e-12)).normalize(),
            b: Some((ab + Vec3::splat(1e-12)).normalize()),
        };
    }
    out
}

/// The crease envelope of a body vertex (the renderer's `envelope()`). The
/// vertex's point on its binding triangle (nodes at `p`, barycentric `w`)
/// moves along u, the planes' barycentric mean normal, to the node tangent
/// plane nearest it: per node i and plane, t is the distance along u to the
/// plane and its weight s = wᵢ exp(−(|t| − min|t|) / (CREASE_SHARP · L)). At
/// a fold the two sides' planes cross along a straight line, so the fold is
/// straight and sharp across while smooth along it, wherever the binding
/// triangles' edges run; where the planes barely differ the weights are the
/// barycentric ones (Phong-like). Every term is invariant under rigid motion
/// and continuous across a shared edge (the opposite node's weight is zero
/// there). Returns the weighted normal, u and the offset along u.
pub fn crease_envelope(
    p: [Vec3; 3],
    planes: [NodePlanes; 3],
    w: Vec3,
    l: f32,
) -> (Vec3, Vec3, f32) {
    let pl = w.x * p[0] + w.y * p[1] + w.z * p[2];
    let k = 1.0 / (CREASE_SHARP * l.max(1e-4));
    let ww = w.max(Vec3::ZERO);
    let mut u = Vec3::splat(1e-9);
    for i in 0..3 {
        u += ww[i] * (planes[i].a + planes[i].b.unwrap_or(planes[i].a));
    }
    let u = u.normalize();
    let mut cand: Vec<(Vec3, f32, f32)> = Vec::with_capacity(6);
    for i in 0..3 {
        let lw = w[i].max(1e-6).ln();
        for nrm in [Some(planes[i].a), planes[i].b].into_iter().flatten() {
            let t = -(pl - p[i]).dot(nrm) / nrm.dot(u).max(0.2);
            cand.push((nrm, t, t.abs() * k - lw));
        }
    }
    let best = cand.iter().map(|c| c.2).fold(f32::INFINITY, f32::min);
    let (mut acc, mut off, mut sum) = (Vec3::ZERO, 0.0, 0.0);
    for (nrm, t, sc) in cand {
        let s = (best - sc).exp();
        acc += s * nrm;
        off += s * t;
        sum += s;
    }
    ((acc + Vec3::splat(1e-9)).normalize(), u, off / sum)
}

/// `v` turned by the shortest rotation taking unit `a` to unit `b`.
fn rot_to(a: Vec3, b: Vec3, v: Vec3) -> Vec3 {
    let k = a.cross(b);
    let c = a.dot(b);
    if c < -0.999 {
        let m = (a + b + Vec3::new(1e-6, 2e-6, 3e-6)).normalize();
        return v - 2.0 * v.dot(m) * m;
    }
    v * c + k.cross(v) + k * (k.dot(v) / (1.0 + c))
}

/// The visual's median rest triangle edge (m), over its opaque triangles:
/// the renderer picks, per car, the finest level of detail whose edges are
/// at least a pixel on screen (finer triangles only add overshading, the
/// image is the same).
pub fn median_edge(v: &ShowcaseVisual) -> f32 {
    let mut e: Vec<f32> = Vec::with_capacity(v.indices.len());
    for t in v.indices.chunks_exact(3) {
        if transparent(v.material[t[0] as usize]) {
            continue;
        }
        let p = |k: usize| Vec3::from_array(v.positions[t[k] as usize]);
        e.push(p(0).distance(p(1)));
        e.push(p(1).distance(p(2)));
        e.push(p(2).distance(p(0)));
    }
    if e.is_empty() {
        return 0.0;
    }
    let mid = e.len() / 2;
    *e.select_nth_unstable_by(mid, f32::total_cmp).1
}

/// Per solver node its skin normal (zero off the skin).
pub fn node_normals(v: &ShowcaseVisual, nodes: &[Vec3]) -> Vec<Vec3> {
    let mut out = vec![Vec3::ZERO; nodes.len()];
    for i in 0..nodes.len().min(v.node_tri_start.len().saturating_sub(1)) {
        let mut acc = Vec3::ZERO;
        for &ti in &v.node_tri_list[v.node_tri_start[i] as usize..v.node_tri_start[i + 1] as usize]
        {
            let t = v.normal_tris[ti as usize];
            let (a, b, c) = (
                nodes[t[0] as usize],
                nodes[t[1] as usize],
                nodes[t[2] as usize],
            );
            acc += (b - a).cross(c - a);
        }
        out[i] = acc.normalize_or_zero();
    }
    out
}

/// Deform the rest mesh by node positions (the solver's order): the GPU
/// shader's formula on the CPU, for tests and offline renders.
pub fn deform(v: &ShowcaseVisual, nodes: &[Vec3]) -> (Vec<Vec3>, Vec<Vec3>) {
    deform_with(v, nodes, None)
}

/// Whether a vertex of material `m` on a broken unit of `kind` is gone (a
/// shattered pane, lens or mirror glass; the laminated windscreen stays).
pub fn shatters(kind: u8, m: u8) -> bool {
    kind != unit_kind::LAMINATED
        && matches!(
            m,
            mat::GLASS | mat::PRIVACY | mat::LENS | mat::TAIL_LENS | mat::CHROME
        )
}

/// [`deform`] with the damage units' states ([`unit_states`]): a shattered
/// vertex ([`shatters`]) is dropped (NaN), as the renderer drops it.
pub fn deform_with(
    v: &ShowcaseVisual,
    nodes: &[Vec3],
    units: Option<&[UnitState]>,
) -> (Vec<Vec3>, Vec<Vec3>) {
    deform_beams(v, nodes, units, None)
}

/// Whether vertex `i` rides its own part's frame instead of its shut line's
/// skin triangle: that triangle has died (a beam of its [`ShowcaseVisual::skin_death`]
/// broke) or torn. The renderer's `skin()` is the same.
pub fn on_alt(
    v: &ShowcaseVisual,
    i: usize,
    nodes: &[Vec3],
    beams: Option<&[BeamState]>,
) -> Option<usize> {
    let a = *v.alt.get(i)?;
    if a == 0 {
        return None;
    }
    let a = a as usize - 1;
    let dead = beams.is_some_and(|bs| {
        v.skin_death[v.alt_tri[a] as usize]
            .iter()
            .any(|&j| j != u32::MAX && bs.get(j as usize).is_some_and(|b| b.broken))
    });
    let [n0, n1, n2] = v.bind_nodes[i].map(|k| nodes[k as usize]);
    let [r1, r2] = v.bind_rest[i];
    let torn = (n1 - n0).length() > TEAR_STRETCH * r1 || (n2 - n0).length() > TEAR_STRETCH * r2;
    (dead || torn).then_some(a)
}

/// [`deform_with`], with the solver's beams (shut-line vertices switch to
/// their own part's frame once their triangle dies: [`on_alt`]).
pub fn deform_beams(
    v: &ShowcaseVisual,
    nodes: &[Vec3],
    units: Option<&[UnitState]>,
    beams: Option<&[BeamState]>,
) -> (Vec<Vec3>, Vec<Vec3>) {
    let now_pl = node_planes(v, nodes);
    let rest: Vec<Vec3> = v.node_rest.iter().map(|q| Vec3::from_array(*q)).collect();
    let rest_pl = node_planes(v, &rest);
    let mut pos = Vec::with_capacity(v.positions.len());
    let mut nrm = Vec::with_capacity(v.positions.len());
    for i in 0..v.positions.len() {
        let alt = on_alt(v, i, nodes, beams);
        let ([a, b, c], [al, be, h], [r1, r2], bn) = match alt {
            Some(k) => (
                v.alt_nodes[k],
                v.alt_coords[k],
                v.alt_rest[k],
                v.alt_normal[k],
            ),
            None => (
                v.bind_nodes[i],
                v.bind_coords[i],
                v.bind_rest[i],
                v.bind_normal[i],
            ),
        };
        let (p0, p1, p2) = (nodes[a as usize], nodes[b as usize], nodes[c as usize]);
        let e1 = p1 - p0;
        let e2 = p2 - p0;
        let n = e1.cross(e2).normalize_or_zero();
        let t = e1.normalize_or_zero();
        let bb = n.cross(t);
        let rigid = v.rigid.get(i).copied().unwrap_or(0) == 1;
        let gone = units.is_some_and(|st| {
            let u = v.unit.get(i).copied().unwrap_or(u32::MAX);
            u != u32::MAX
                && st[u as usize].broken
                && shatters(v.units[u as usize].kind, v.material[i])
        });
        if gone || e1.length() > TEAR_STRETCH * r1 || e2.length() > TEAR_STRETCH * r2 {
            // Torn: the renderer cuts its triangles away (NaN here).
            pos.push(Vec3::splat(f32::NAN));
            nrm.push(Vec3::Y);
            continue;
        }
        if rigid {
            pos.push(p0 + t * al + bb * be + n * h);
        } else {
            pos.push(p0 + e1 * al + e2 * be + n * h);
        }
        let [x, y, z] = bn;
        let nf = (t * x + bb * y + n * z).normalize_or_zero();
        let ids = [a as usize, b as usize, c as usize];
        if v.smooth[i] == 1 && !rigid {
            let (pn, pr) = (ids.map(|k| now_pl[k]), ids.map(|k| rest_pl[k]));
            if pn
                .iter()
                .chain(pr.iter())
                .all(|q| q.a.length_squared() > 0.5)
            {
                // The crease envelope now and at rest (see [`crease_envelope`]).
                let w = Vec3::new(1.0 - al - be, al, be);
                let l = 0.5 * (r1 + r2);
                let rest_p = ids.map(|k| Vec3::from_array(v.node_rest[k]));
                let (nh, u, e) = crease_envelope([p0, p1, p2], pn, w, l);
                let (nh0, _, e0) = crease_envelope(rest_p, pr, w, l);
                let last = pos.len() - 1;
                pos[last] += (e - e0).clamp(-ENVELOPE_MAX, ENVELOPE_MAX) * u;
                nrm.push(rot_to(nh0, nh, Vec3::from_array(v.normals[i])).normalize_or_zero());
                continue;
            }
        }
        nrm.push(nf);
    }
    (pos, nrm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::authoring::{build, sedan, BuildParams, Resolution};
    use crate::explicit_beams::StaticBox;

    fn states(sim: &ExplicitBeams) -> Vec<BeamState> {
        sim.beams
            .iter()
            .map(|b| BeamState {
                broken: b.broken,
                rest: b.rest,
            })
            .collect()
    }

    fn spawn(b: &Built, vel: Vec3) -> ExplicitBeams {
        let st = ExplicitBeamsSettings {
            turbulence: false,
            ..Default::default()
        };
        let probe =
            ExplicitBeams::from_truck(&b.truck, Vec3::ZERO, Vec3::ZERO, st.clone()).unwrap();
        let low = probe.nodes.iter().map(|n| n.pos.y).fold(f32::MAX, f32::min);
        let mut s = ExplicitBeams::from_truck(
            &b.truck,
            Vec3::new(0.0, probe.spawn_clearance() - low, 0.0),
            vel,
            st,
        )
        .unwrap();
        s.roll_tyres(vel);
        s
    }

    /// The crease envelope follows the plane of the side a point is on, so a
    /// fold through the binding triangle is the planes' straight crossing
    /// line: a tent (ridge along z at x = 0, y = 1) bound by a triangle with
    /// a ridge node (both planes) and one node on each slope.
    #[test]
    fn crease_envelope_is_the_planes_crossing() {
        let na = Vec3::new(1.0, 1.0, 0.0).normalize();
        let nb = Vec3::new(-1.0, 1.0, 0.0).normalize();
        let p = [
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 1.0),
        ];
        let pl = [
            NodePlanes { a: na, b: Some(nb) },
            NodePlanes { a: na, b: None },
            NodePlanes { a: nb, b: None },
        ];
        let surface = |w: Vec3| {
            let (n, u, e) = crease_envelope(p, pl, w, 1.4);
            (w.x * p[0] + w.y * p[1] + w.z * p[2] + e * u, n)
        };
        let tent = |q: Vec3| 1.0 - q.x.abs();
        for &w in &[
            Vec3::new(0.2, 0.6, 0.2),
            Vec3::new(0.2, 0.2, 0.6),
            Vec3::new(0.5, 0.4, 0.1),
            Vec3::new(0.1, 0.3, 0.6),
        ] {
            let (q, n) = surface(w);
            // Off the ridge (the planes' |t| differ by much more than the
            // temperature) the point lies on its side's plane, with its normal.
            assert!((q.y - tent(q)).abs() < 2e-3, "{w:?}: {q:?}");
            let side = if q.x > 0.0 { na } else { nb };
            assert!(n.dot(side) > 0.999, "{w:?}: {n:?}");
        }
        // On the ridge line itself the point reaches the ridge, not a chamfer.
        let (q, _) = surface(Vec3::new(0.2, 0.4, 0.4));
        assert!(q.x.abs() < 1e-4 && (q.y - 1.0).abs() < 1e-3, "{q:?}");
        // On a flat triangle nothing moves.
        let up = NodePlanes {
            a: Vec3::Y,
            b: None,
        };
        let (_, _, e) = crease_envelope(
            p.map(|q| q * Vec3::new(1.0, 0.0, 1.0)),
            [up; 3],
            Vec3::splat(1.0 / 3.0),
            1.4,
        );
        assert!(e.abs() < 1e-6);
    }

    /// Every pane, lamp and mirror is a unit with beams; none breaks at rest
    /// (exact), and a unit's shatter is the CPU deform's dropped vertices.
    #[test]
    fn damage_units_follow_the_beams() {
        let b = build(
            &sedan::sedan(),
            &BuildParams {
                resolution: Resolution::Low,
                ..Default::default()
            },
        )
        .unwrap();
        let v = showcase_visual(&b).unwrap();
        let mut g: Vec<&String> = b.bind_tris.keys().collect();
        g.sort();
        eprintln!("groups: {g:?}");
        for u in &v.units {
            eprintln!(
                "unit {:>16} kind {} beams {:4} strain {:.4} shards {}",
                u.name,
                u.kind,
                u.beams.len(),
                u.strain,
                u.shards
            );
            assert!(!u.beams.is_empty(), "{} has no beams", u.name);
        }
        for want in [
            "windscreen",
            "backlight",
            "glass_side_f_l",
            "glass_side_r_r",
            "glass_side_q_l",
            "headlight_l",
            "taillight_r",
            "mirror_l",
        ] {
            assert!(v.units.iter().any(|u| u.name == want), "no unit {want}");
        }
        let mut s = spawn(&b, Vec3::ZERO);
        s.run(1.5);
        let st = unit_states(&v, &states(&s));
        for (u, x) in v.units.iter().zip(&st) {
            eprintln!(
                "rest: {:>16} strain {:.5} broken {}",
                u.name, x.strain, x.broken
            );
        }
        assert!(st.iter().all(|x| !x.broken), "a unit broke at rest");
        // A 56 km/h rigid wall.
        let vel = Vec3::new(0.0, 0.0, 56.0 / 3.6);
        let mut s = spawn(&b, vel);
        let front = s.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
        s.boxes.push(StaticBox::centered(
            Vec3::new(0.0, 1.0, front + 0.05 + 1.0),
            Vec3::new(3.0, 2.0, 1.0),
        ));
        s.run(1.0);
        let st = unit_states(&v, &states(&s));
        for (u, x) in v.units.iter().zip(&st) {
            eprintln!(
                "wall 56: {:>16} strain {:.5} broken {}",
                u.name, x.strain, x.broken
            );
        }
        let (p, _) = deform_with(&v, &s.positions(), Some(&st));
        for (k, q) in p.iter().enumerate() {
            let u = v.unit[k];
            if u != u32::MAX
                && st[u as usize].broken
                && shatters(v.units[u as usize].kind, v.material[k])
            {
                assert!(q.x.is_nan());
            }
        }
    }
}

#[cfg(test)]
mod inner_tests {
    use super::*;
    use crate::authoring::parts::design::{DOOR_SECTION, LID_SECTION};
    use crate::authoring::{build, sedan, BuildParams, Resolution};

    /// P7: each door's and lid's inner stamping sits at its authored section
    /// under the outer skin (the median distance from an inner vertex to the
    /// outer panel's nearest vertex, ±5 mm), rides the outer panel's part,
    /// and the engine bay is there.
    #[test]
    fn inner_panels_sit_at_their_section() {
        let b = build(
            &sedan::sedan(),
            &BuildParams {
                resolution: Resolution::Low,
                ..Default::default()
            },
        )
        .unwrap();
        let v = showcase_visual(&b).unwrap();
        let piece = |n: &str| v.pieces.iter().find(|p| p.0 == n).cloned();
        // **Choice**: 99 % (all four measure 100 %).
        const ON_OWN_PART: f32 = 0.99;
        // **Choice**: half on two of the panel's own nodes (the rest on the
        // shut line's skin triangles, as the outer edge over them; the
        // deck's inner frame runs along the lid's edge: 57 % at low
        // resolution, 68 % at medium).
        const ON_PART_AT_LEAST: f32 = 0.5;
        for (outer, depth) in [
            ("door_f_l", DOOR_SECTION),
            ("door_r_r", DOOR_SECTION),
            ("bonnet", LID_SECTION),
            ("deck", LID_SECTION),
        ] {
            let o = piece(outer).unwrap_or_else(|| panic!("no {outer}"));
            let i = piece(&format!("{outer}_inner")).unwrap_or_else(|| panic!("no {outer}_inner"));
            // The inner skin's vertices (normals facing in, against the outer
            // skin's): their depth along the nearest outer vertex's normal.
            let op: Vec<(Vec3, Vec3)> = (o.1..o.1 + o.2)
                .map(|k| {
                    (
                        Vec3::from_array(v.positions[k as usize]),
                        Vec3::from_array(v.normals[k as usize]),
                    )
                })
                .collect();
            let mut d: Vec<f32> = (i.1..i.1 + i.2)
                .step_by(5)
                .filter_map(|k| {
                    let p = Vec3::from_array(v.positions[k as usize]);
                    let n = Vec3::from_array(v.normals[k as usize]);
                    let (q, qn) = op
                        .iter()
                        .min_by(|a, b| a.0.distance(p).total_cmp(&b.0.distance(p)))?;
                    (n.dot(*qn) < -0.9).then(|| (*q - p).dot(*qn))
                })
                .collect();
            d.sort_by(f32::total_cmp);
            let med = d[d.len() / 2];
            eprintln!(
                "{outer}: inner {} vertices, median section {med:.4} m (authored {depth})",
                i.2
            );
            assert!(
                (med - depth).abs() < 0.005,
                "{outer}: section {med} against {depth}"
            );
            // It rides its panel's physics part, its flexbody group: two of
            // its frame's three nodes on it (at a shut line the third may be
            // across the border).
            let pname = if outer == "deck" { "bootlid" } else { outer };
            let main = b.skeleton.parts.iter().position(|p| p == pname).unwrap() as u16;
            let own = |k: usize| {
                v.bind_nodes[k]
                    .iter()
                    .filter(|&&i| {
                        b.skeleton
                            .nodes
                            .get(i as usize)
                            .is_some_and(|n| n.part == main)
                    })
                    .count()
                    >= 2
            };
            let on = (i.1..i.1 + i.2).filter(|&k| own(k as usize)).count() as f32 / i.2 as f32;
            // Or, at the panel's edge, the shut line's skin triangle under it,
            // as the outer skin over it.
            let skin: std::collections::HashSet<[u32; 3]> = b
                .skeleton
                .skin
                .iter()
                .map(|t| {
                    let mut u = *t;
                    u.sort_unstable();
                    u
                })
                .collect();
            let on_skin = (i.1..i.1 + i.2)
                .filter(|&k| {
                    let mut u = v.bind_nodes[k as usize];
                    u.sort_unstable();
                    own(k as usize) || skin.contains(&u)
                })
                .count() as f32
                / i.2 as f32;
            eprintln!(
                "  {:.1} % on {pname} or on the skin triangle under it",
                100.0 * on_skin
            );
            assert!(
                on_skin >= ON_OWN_PART,
                "{outer}_inner: {on_skin} on {pname} or the skin"
            );
            eprintln!("  {:.1} % on {pname}", 100.0 * on);
            assert!(on >= ON_PART_AT_LEAST, "{outer}_inner: {on} on {pname}");
        }
        for n in ["engine", "radiator", "apron_wall", "bulkhead"] {
            assert!(piece(n).is_some(), "no {n}");
        }
    }
}
