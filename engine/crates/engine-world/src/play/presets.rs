//! The preset registry: every rubble demo by one name, the family of content it plays, and
//! the page that hosts it today (`engine-webgpu/<page>?<query>`). The playground's phases
//! move each family onto [`super`]'s scene API; a preset's name and its old URL stay the
//! same throughout (the page shell redirects old URLs by this table).

/// The content a preset plays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    /// Rigid stacks, rings, piles and joints (`demo.html`).
    Rigid,
    /// The fracture lab: a scene under any bond model ([`crate::lab`]).
    Lab,
    /// A baked GLB on the range with a scripted take ([`super::range`]).
    Range,
    /// Rigid vehicles (`vehicle.html`).
    Vehicle,
    /// Node–beam soft-body vehicles (`softbody.html`).
    Soft,
    /// CCD, characters and bullets (`motion.html`).
    Motion,
    /// The heightfield collider (`terrain.html`).
    Terrain,
    /// A meteor or a charge on deformable (MPM) ground (`impact.html`).
    Impact,
    /// The meteor run on deformable ground (`showcase.html`).
    Showcase,
    /// The large-maps city (`city.html`).
    City,
}

impl Family {
    pub const ALL: [Family; 10] = [
        Family::Rigid,
        Family::Lab,
        Family::Range,
        Family::Vehicle,
        Family::Soft,
        Family::Motion,
        Family::Terrain,
        Family::Impact,
        Family::Showcase,
        Family::City,
    ];

    /// The family named `name` ([`Self::name`]).
    pub fn from_name(name: &str) -> Option<Family> {
        Self::ALL.into_iter().find(|f| f.name() == name)
    }

    pub fn name(self) -> &'static str {
        match self {
            Family::Rigid => "rigid",
            Family::Lab => "lab",
            Family::Range => "range",
            Family::Vehicle => "vehicle",
            Family::Soft => "soft",
            Family::Motion => "motion",
            Family::Terrain => "terrain",
            Family::Impact => "impact",
            Family::Showcase => "showcase",
            Family::City => "city",
        }
    }
}

/// A demo: its name, title, family and the page that hosts it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Preset {
    /// Unique, URL-safe (`?preset=<name>`).
    pub name: &'static str,
    pub title: &'static str,
    pub family: Family,
    /// The page (relative to `engine-webgpu/`) and its query.
    pub page: &'static str,
    pub query: &'static str,
}

impl Preset {
    /// The page URL that hosts it (relative to `engine-webgpu/`).
    pub fn url(&self) -> String {
        if self.query.is_empty() {
            self.page.to_string()
        } else {
            format!("{}?{}", self.page, self.query)
        }
    }
}

const fn p(
    name: &'static str,
    title: &'static str,
    family: Family,
    page: &'static str,
    query: &'static str,
) -> Preset {
    Preset {
        name,
        title,
        family,
        page,
        query,
    }
}

use Family::*;

const PRESETS: &[Preset] = &[
    // demo.html
    p("wall", "Wall smash", Rigid, "demo.html", "scene=wall"),
    p("ring", "Brick ring", Rigid, "demo.html", "scene=ring"),
    p("gables", "Gables", Rigid, "demo.html", "scene=gables"),
    p("pyramid", "Pyramid", Rigid, "demo.html", "scene=pyramid"),
    p("pile", "Pile", Rigid, "demo.html", "scene=pile"),
    p("columns", "Columns", Rigid, "demo.html", "scene=columns"),
    p(
        "ring110k",
        "110k-brick ring",
        Rigid,
        "demo.html",
        "scene=ring110k",
    ),
    p(
        "gables506k",
        "506k-brick gables",
        Rigid,
        "demo.html",
        "scene=gables506k",
    ),
    p("joints", "Joints", Rigid, "demo.html", "scene=joints"),
    p("ragdolls", "Ragdolls", Rigid, "demo.html", "scene=ragdolls"),
    p(
        "ragdolls10k",
        "10k ragdolls",
        Rigid,
        "demo.html",
        "scene=ragdolls10k",
    ),
    p(
        "hingechains10k",
        "10k hinge chains",
        Rigid,
        "demo.html",
        "scene=hingechains10k",
    ),
    p(
        "motorfield10k",
        "10k motors",
        Rigid,
        "demo.html",
        "scene=motorfield10k",
    ),
    p(
        "legacy-pyramid",
        "Pyramid (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyPyramid",
    ),
    p(
        "legacy-stack",
        "Stack (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyStack",
    ),
    p(
        "legacy-stack-ratio",
        "Stack ratio (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyStackRatio",
    ),
    p(
        "legacy-dynamic-friction",
        "Dynamic friction (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyDynamicFriction",
    ),
    p(
        "legacy-static-friction",
        "Static friction (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyStaticFriction",
    ),
    p(
        "legacy-scatter",
        "Scatter (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyScatter",
    ),
    p(
        "legacy-scatter-boxes",
        "Scatter boxes (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyScatterBoxes",
    ),
    p(
        "legacy-grid-boxes",
        "Grid boxes (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyGridBoxes",
    ),
    p(
        "legacy-slanted-grid",
        "Slanted grid (legacy)",
        Rigid,
        "demo.html",
        "scene=legacySlantedGrid",
    ),
    p(
        "legacy-grid10k",
        "10k grid (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyGrid10k",
    ),
    p(
        "legacy-grid100k",
        "100k grid (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyGrid100k",
    ),
    p(
        "legacy-ground",
        "Ground (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyGround",
    ),
    p(
        "legacy-empty",
        "Empty (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyEmpty",
    ),
    // fracture/
    p("lab", "Fracture lab", Lab, "fracture/", ""),
    // broadside.html, timber.html
    p("broadside", "Broadside", Range, "broadside.html", "scene=broadside"),
    p("timber-yard", "Timber yard", Range, "timber.html", "scene=timber"),
    // vehicle.html
    p("vehicles", "Vehicles", Vehicle, "vehicle.html", ""),
    p("bridge", "Bridge", Vehicle, "vehicle.html", "scene=bridge"),
    p(
        "buggies",
        "Buggies",
        Vehicle,
        "vehicle.html",
        "scene=buggies",
    ),
    p("carry", "Carry", Vehicle, "vehicle.html", "scene=carry"),
    p(
        "destruction",
        "Destruction",
        Vehicle,
        "vehicle.html",
        "scene=destruction",
    ),
    p(
        "gauntlet",
        "Gauntlet",
        Vehicle,
        "vehicle.html",
        "scene=gauntlet",
    ),
    p(
        "meteors",
        "Meteors",
        Vehicle,
        "vehicle.html",
        "scene=meteors",
    ),
    p(
        "playground",
        "Playground",
        Vehicle,
        "vehicle.html",
        "scene=playground",
    ),
    p("run", "Meteor run", Vehicle, "vehicle.html", "scene=run"),
    p("seesaw", "Seesaw", Vehicle, "vehicle.html", "scene=seesaw"),
    p(
        "sidewall",
        "Side wall",
        Vehicle,
        "vehicle.html",
        "scene=sidewall",
    ),
    p("slab", "Slab", Vehicle, "vehicle.html", "scene=slab"),
    p(
        "tanklab",
        "Tank lab",
        Vehicle,
        "vehicle.html",
        "scene=tanklab",
    ),
    p(
        "traffic",
        "Traffic",
        Vehicle,
        "vehicle.html",
        "scene=traffic",
    ),
    p(
        "vibeland",
        "Vibe-land",
        Vehicle,
        "vehicle.html",
        "scene=vibeland",
    ),
    // softbody.html
    p("soft", "Soft-body vehicles", Soft, "softbody.html", ""),
    p(
        "soft-arena",
        "Explicit fleet arena",
        Soft,
        "softbody.html",
        "solver=explicit&xscene=arena",
    ),
    p(
        "soft-crates",
        "Crates",
        Soft,
        "softbody.html",
        "scene=crates",
    ),
    p(
        "soft-crusher",
        "Crusher",
        Soft,
        "softbody.html",
        "scene=crusher",
    ),
    p("soft-drop", "Drop", Soft, "softbody.html", "scene=drop"),
    p(
        "soft-headon",
        "Head-on",
        Soft,
        "softbody.html",
        "scene=headon",
    ),
    p(
        "soft-import",
        "Import",
        Soft,
        "softbody.html",
        "scene=import",
    ),
    p("soft-limbo", "Limbo", Soft, "softbody.html", "scene=limbo"),
    p("soft-pole", "Pole", Soft, "softbody.html", "scene=pole"),
    p("soft-ramp", "Ramp", Soft, "softbody.html", "scene=ramp"),
    p(
        "soft-rollover",
        "Rollover",
        Soft,
        "softbody.html",
        "scene=rollover",
    ),
    p(
        "soft-sandwich",
        "Sandwich",
        Soft,
        "softbody.html",
        "scene=sandwich",
    ),
    p(
        "soft-sidepole",
        "Side pole",
        Soft,
        "softbody.html",
        "scene=sidepole",
    ),
    p("soft-tbone", "T-bone", Soft, "softbody.html", "scene=tbone"),
    p(
        "soft-traffic",
        "Soft traffic",
        Soft,
        "softbody.html",
        "scene=traffic",
    ),
    p(
        "soft-wall",
        "Soft wall",
        Soft,
        "softbody.html",
        "scene=wall",
    ),
    p("soft-yard", "Yard", Soft, "softbody.html", "scene=yard"),
    // motion.html
    p(
        "ccd",
        "Continuous collision",
        Motion,
        "motion.html",
        "scene=ccd",
    ),
    p(
        "character",
        "Character",
        Motion,
        "motion.html",
        "scene=character",
    ),
    p("bullets", "Bullets", Motion, "motion.html", "scene=bullets"),
    // terrain.html, impact.html, showcase.html, city.html
    p(
        "terrain",
        "Heightfield terrain",
        Terrain,
        "terrain.html",
        "",
    ),
    p("impact", "Meteor impact", Impact, "impact.html", ""),
    p(
        "showcase",
        "Meteor run showcase",
        Showcase,
        "showcase.html",
        "",
    ),
    p(
        "showcase-play",
        "Meteor run (play)",
        Showcase,
        "showcase.html",
        "scene=play",
    ),
    p(
        "showcase-bts",
        "Meteor run (behind the scenes)",
        Showcase,
        "showcase.html",
        "scene=bts",
    ),
    p("city", "Large-maps city", City, "city.html", ""),
];

/// Every preset, in the landing page's order.
pub fn presets() -> &'static [Preset] {
    PRESETS
}

/// The preset named `name`.
pub fn preset(name: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_url_safe() {
        let mut seen = std::collections::HashSet::new();
        for p in presets() {
            assert!(seen.insert(p.name), "{} twice", p.name);
            assert!(
                p.name
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-'),
                "{} is not URL-safe",
                p.name
            );
        }
    }

    #[test]
    fn urls_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for p in presets() {
            assert!(seen.insert(p.url()), "{} twice", p.url());
        }
    }
}
