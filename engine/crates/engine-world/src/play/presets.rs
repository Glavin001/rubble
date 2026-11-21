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
    /// A demo with a page of its own, not (yet) a scene of the registry: the vehicle lab,
    /// one World, Why Rubble, the tools. Listed so the menu holds every demo.
    Page,
}

impl Family {
    pub const ALL: [Family; 11] = [
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
        Family::Page,
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
            Family::Page => "page",
        }
    }

    /// The playground hub the family's presets are listed under (play.html's menu).
    pub const fn hub(self) -> &'static str {
        match self {
            Family::Lab | Family::Range | Family::City => "Destruction",
            Family::Vehicle | Family::Soft => "Vehicles",
            Family::Terrain | Family::Impact | Family::Showcase => "Terrain",
            Family::Rigid | Family::Motion => "Scale & motion",
            Family::Page => "Studio",
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
    /// The playground hub it is listed under.
    pub hub: &'static str,
    /// How the menu lists it.
    pub menu: Menu,
}

/// How the playground menu lists a preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Menu {
    /// A main entry of its hub.
    Main,
    /// Under its hub's "More" (the legacy engine's scenes, diagnostics).
    More,
    /// Not listed: another name for a listed preset (an old URL that keeps working).
    Alias,
}

impl Menu {
    pub fn name(self) -> &'static str {
        match self {
            Menu::Main => "main",
            Menu::More => "more",
            Menu::Alias => "alias",
        }
    }
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
        hub: family.hub(),
        menu: Menu::Main,
    }
}

/// A preset listed under its hub's "More".
const fn more(name: &'static str, title: &'static str, family: Family, page: &'static str, query: &'static str) -> Preset {
    Preset { menu: Menu::More, ..p(name, title, family, page, query) }
}

/// Another name for a listed preset (an old URL): not listed.
const fn alias(name: &'static str, title: &'static str, family: Family, page: &'static str, query: &'static str) -> Preset {
    Preset { menu: Menu::Alias, ..p(name, title, family, page, query) }
}

/// A demo on a page of its own, listed under `hub`.
const fn page(name: &'static str, title: &'static str, hub: &'static str, page: &'static str, query: &'static str) -> Preset {
    Preset { hub, ..p(name, title, Family::Page, page, query) }
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
    more("legacy-pyramid",
        "Pyramid (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyPyramid",
    ),
    more("legacy-stack",
        "Stack (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyStack",
    ),
    more("legacy-stack-ratio",
        "Stack ratio (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyStackRatio",
    ),
    more("legacy-dynamic-friction",
        "Dynamic friction (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyDynamicFriction",
    ),
    more("legacy-static-friction",
        "Static friction (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyStaticFriction",
    ),
    more("legacy-scatter",
        "Scatter (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyScatter",
    ),
    more("legacy-scatter-boxes",
        "Scatter boxes (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyScatterBoxes",
    ),
    more("legacy-grid-boxes",
        "Grid boxes (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyGridBoxes",
    ),
    more("legacy-slanted-grid",
        "Slanted grid (legacy)",
        Rigid,
        "demo.html",
        "scene=legacySlantedGrid",
    ),
    more("legacy-grid10k",
        "10k grid (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyGrid10k",
    ),
    more("legacy-grid100k",
        "100k grid (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyGrid100k",
    ),
    more("legacy-ground",
        "Ground (legacy)",
        Rigid,
        "demo.html",
        "scene=legacyGround",
    ),
    more("legacy-empty",
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
    alias("vehicles", "Vehicles", Vehicle, "vehicle.html", ""),
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
    alias("soft", "Soft-body vehicles", Soft, "softbody.html", ""),
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
    // Demos on pages of their own (not yet scenes of the registry).
    page("why-rubble", "Why Rubble (compare)", "Destruction", "fracture/compare.html", ""),
    page("vehicle-lab", "Vehicle lab", "Vehicles", "vehicle-lab.html", ""),
    page("one-world", "Cars into buildings (one World)", "Vehicles", "one-world.html", ""),
    page("brick-ring-drive", "Drive through the 110k ring", "Vehicles", "brick-ring-drive.html", ""),
    page("crash-town", "Crash town (2 kHz fleet)", "Vehicles", "explicit-world.html", ""),
    page("showcase-car", "Showcase car test", "Studio", "showcase-car.html", ""),
    page("vehicle-designer", "Vehicle designer", "Studio", "vehicle-designer.html", ""),
    page("models", "Generated models", "Studio", "models/", ""),
    page("crash-orbit", "Crash replay orbit", "Studio", "crash-orbit.html", ""),
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
