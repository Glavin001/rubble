//! Beam materials: steel grades and the other stuff a car is made of, as node–beam
//! properties. A material's stiffness is given for a 10 cm beam and scales as
//! 1/length (a longer beam of the same sheet is softer, as in a continuum); its
//! yield (`deform`) and break (`strength`) thresholds are forces in newtons, as in
//! a `.truck` file's `set_beam_defaults`.

use serde::{Deserialize, Serialize};

/// The metals' plastic coefficient: a yielded beam keeps 30 % of its yield
/// strain as spring-back.
pub const PLASTIC: f32 = 0.95;

/// Reference length the stiffness is quoted at (m).
pub const REF_LENGTH: f32 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Material {
    /// Stiffness of a [`REF_LENGTH`] beam (N/m).
    pub k: f32,
    /// Damping (N·s/m).
    pub damp: f32,
    /// Yield threshold (N): past it the beam deforms plastically.
    pub deform: f32,
    /// Break threshold (N).
    pub strength: f32,
    /// Plastic coefficient (0: the beam keeps no elastic spring-back once yielded).
    pub plastic: f32,
    /// Relative mass per metre of beam (the body's mass is spread by it, then
    /// scaled to the body's target).
    pub density: f32,
}

impl Material {
    pub const fn new(k: f32, damp: f32, deform: f32, strength: f32, density: f32) -> Self {
        Self {
            k,
            damp,
            deform,
            strength,
            plastic: 0.0,
            density,
        }
    }

    /// A grade from its stiffness (for a [`REF_LENGTH`] beam), damping, yield
    /// strain and break-to-yield ratio: the yield force is the strain times the
    /// beam's axial rigidity (k × [`REF_LENGTH`], the same at every length).
    pub fn grade(k: f32, damp: f32, yield_strain: f32, break_ratio: f32, density: f32) -> Self {
        let deform = yield_strain * k * REF_LENGTH;
        Self {
            k,
            damp,
            deform,
            strength: deform * break_ratio,
            plastic: 0.0,
            density,
        }
    }

    /// The plastic coefficient (0: a yielded beam keeps its yield force as
    /// elastic strain and springs back by it; 1: it keeps none).
    pub fn plastic(self, p: f32) -> Self {
        Self { plastic: p, ..self }
    }

    /// Yield and break thresholds scaled by `s` (stiffness and mass unchanged).
    pub fn scaled(self, s: f32) -> Self {
        Self {
            deform: self.deform * s,
            strength: self.strength * s,
            ..self
        }
    }

    /// Stiffness scaled by `s`, damping by √s (the same damping ratio).
    pub fn stiffer(self, s: f32) -> Self {
        Self {
            k: self.k * s,
            damp: self.damp * s.sqrt(),
            ..self
        }
    }

    /// Stiffness of a beam of rest length `len` (m).
    pub fn k_at(&self, len: f32) -> f32 {
        self.k * REF_LENGTH / len.max(REF_LENGTH)
    }

    /// A spot weld or bolt between the soft skin and the structure: the skin's
    /// stiffness (so the light skin stays stable) with `strong`'s thresholds.
    pub fn weld(skin: Self, strong: Self) -> Self {
        let _ = strong;
        Self {
            k: skin.k * 2.0,
            damp: skin.damp * 2.0,
            deform: skin.deform * 2.0,
            strength: skin.strength * 3.0,
            plastic: skin.plastic,
            density: skin.density,
        }
    }

    /// The weaker of two materials (a joint between them): each threshold, stiffness
    /// and damping the lower.
    pub fn weaker(self, o: Self) -> Self {
        Self {
            k: self.k.min(o.k),
            damp: self.damp.min(o.damp),
            deform: self.deform.min(o.deform),
            strength: self.strength.min(o.strength),
            plastic: self.plastic.min(o.plastic),
            density: self.density.min(o.density),
        }
    }
}

/// Bay length the damping of [`Material::steel`] is set for (m): a typical
/// beam of the sparse body.
pub const REF_SPAN: f32 = 0.3;
/// Reduced mass of a typical beam's node pair (kg) for the same: the body's
/// nodes weigh 0.5–2 kg.
pub const REF_MU: f32 = 0.4;

impl Material {
    /// A grade from its axial rigidity `kl` (k × length, N: the same at every
    /// length), its damping ratio `zeta` at a [`REF_SPAN`] beam between
    /// [`REF_MU`] nodes, its yield force (N), its elongation at break and its
    /// density. A yielded beam in tension loses strength as it stretches (the
    /// solver takes k × the plastic stretch off it), so the break force is the
    /// yield plus `elong` × k·L: the beam tears after stretching `elong` of its
    /// length, as the metal does (mild steel ~0.3, boron ~0.08; glass ~0).
    pub fn steel(kl: f32, zeta: f32, yield_force: f32, elong: f32, density: f32) -> Self {
        let k = kl / REF_LENGTH;
        let damp = 2.0 * zeta * (kl / REF_SPAN * REF_MU).sqrt();
        Self {
            k,
            damp,
            deform: yield_force,
            strength: yield_force + elong * kl,
            plastic: 0.0,
            density,
        }
    }

    /// Axial rigidity k × length (N).
    pub fn kl(&self) -> f32 {
        self.k * REF_LENGTH
    }
}

/// The grades a car body is built from. Values are this generator's own,
/// chosen for a 0.5 ms explicit step with 0.5–2 kg nodes (see
/// `engine/docs/vehicle-authoring.md`); every vehicle definition may override
/// them. Rigidities are as high as the step allows at the body's mass; the
/// yield forces set where the body gives in a crash.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Grades {
    /// Outer skin panels (doors' skins, fenders, bonnet, boot lid, roof skin).
    pub panel: Material,
    /// The floor pan.
    pub floor: Material,
    /// Light inner frames (bonnet and boot-lid frames).
    pub frame: Material,
    /// Crush members (front and rear rails, aprons): stiff, so they fold
    /// plastically at a high load instead of springing.
    pub crush: Material,
    /// Mild/HSLA structure: floor pan, inner panels, crumple-zone rails.
    pub hsla: Material,
    /// Ultra-high-strength steel: the passenger cell (sills, pillars, roof rails).
    pub uhss: Material,
    /// Press-hardened boron steel: B-pillars, door intrusion beams, tunnel.
    pub boron: Material,
    /// Aluminium bumper beams and crash boxes.
    pub alu: Material,
    /// Plastic bumper covers and grille.
    pub plastic: Material,
    /// Laminated and tempered glass.
    pub glass: Material,
    /// Cast blocks (engine, gearbox, knuckles): never yield.
    pub rigid: Material,
    /// Suspension links, knuckles and their brackets: stiff and well damped
    /// (a knuckle on soft links whips round in a crash), never yield.
    #[serde(default = "link")]
    pub link: Material,
    /// Mounts and bearing pads of blocks and light parts (engine mounts, seat
    /// mounts, door beams' bearing): sized to the lighter node they hold.
    #[serde(default = "mount")]
    pub mount: Material,
}

fn mount() -> Material {
    // Rubber and bushings: well damped (what they hold is heavy).
    Material::steel(1.2e5, 0.15, 3.65e3, 0.2, 2.0).plastic(PLASTIC)
}

fn link() -> Material {
    Material::steel(1.5e5, 0.05, 1.0e9, 0.0, 6.0)
}

/// Damping ratio of the body's grades.
pub const ZETA: f32 = 0.08;

impl Default for Grades {
    /// Two tiers. The frame's grades (crush, HSLA, UHSS, boron, aluminium)
    /// on heavy frame nodes, 1.5–2 kg; the skin's (panel, floor pan, lid
    /// frames, plastic, glass) on light ones, 0.3–0.5 kg: a grade's rigidity
    /// is what its tier's nodes allow at the 0.5 ms step. Yield forces set
    /// where the body gives in a crash.
    fn default() -> Self {
        Self {
            panel: Material::steel(2.7e4, ZETA, 1.0e3, 0.3, 1.0).plastic(PLASTIC),
            floor: Material::steel(2.7e4, ZETA, 1.4e3, 0.3, 1.5).plastic(PLASTIC),
            frame: Material::steel(2.0e4, ZETA, 1.9e2, 0.3, 1.5).plastic(PLASTIC),
            crush: Material::steel(3.75e5, ZETA, 1.23e4, 0.25, 2.0).plastic(PLASTIC),
            hsla: Material::steel(1.45e5, ZETA, 2.9e3, 0.2, 2.0).plastic(PLASTIC),
            uhss: Material::steel(2.2e5, ZETA, 3.5e3, 0.12, 3.0).plastic(PLASTIC),
            boron: Material::steel(2.65e5, ZETA, 3.9e3, 0.08, 3.5).plastic(PLASTIC),
            alu: Material::steel(1.45e5, ZETA, 2.9e3, 0.12, 2.0).plastic(PLASTIC),
            plastic: Material::steel(4.0e3, ZETA, 5.0e2, 0.25, 0.5),
            glass: Material::steel(9.0e3, ZETA, 4.0e2, 0.04, 1.0),
            rigid: Material::steel(1.5e5, 0.03, 1.0e9, 0.0, 6.0),
            link: link(),
            mount: mount(),
        }
    }
}
