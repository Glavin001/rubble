//! The large-maps city (`city.html`) as a [`super::SceneDef`]: [`crate::city`]'s houses,
//! map, sleep policy and piece pool from the page's switches, the world it needs, and its
//! placement (every house; the page places a few per frame). Its actions: a shell at a
//! point and a scenario started once the city has settled.

use crate::city::{city_house, City, CityDef, CityScenario};
use crate::*;
use serde::{Deserialize, Serialize};

/// `city.html`'s switches.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", default)]
pub struct CitySceneDef {
    /// Houses (`?k=`, default 1,024).
    pub houses: u32,
    /// Houses fall asleep once settled (`?sleep=0`: never).
    pub sleep: bool,
    /// Piece slots (`?pool=`; none: a slot per chunk). Default 131,072 on the dense town,
    /// 32,768 on the benchmark's map.
    pub pool: Option<u32>,
    /// The lot pitch (m, `?spacing=`, default 16); none: the benchmark's 2.2 km map.
    pub pitch: Option<f32>,
    /// Bodies below this height leave the world (`?kill=`, default −50; none: `off`).
    pub kill_below: Option<f32>,
    /// What happens once the city settles (`?scenario=still|drive|war`).
    pub scenario: String,
}

impl Default for CitySceneDef {
    fn default() -> Self {
        CitySceneDef {
            houses: 1024,
            sleep: true,
            pool: Some(131072),
            pitch: Some(16.0),
            kill_below: Some(-50.0),
            scenario: "still".into(),
        }
    }
}

impl CitySceneDef {
    /// The page's switches (`URLSearchParams` and `Number` as the page reads them).
    pub(crate) fn from_query(q: &super::scene::Query) -> anyhow::Result<Self> {
        let num = |k: &str| q.get(k).map(super::scene::js_number);
        let houses = num("k").unwrap_or(1024.0);
        anyhow::ensure!(
            houses.fract() == 0.0 && houses > 0.0,
            "k={}: a house count",
            q.get("k").unwrap_or("")
        );
        let spacing = q.get("spacing").unwrap_or("16");
        let pitch = if spacing == "bench" {
            None
        } else {
            let p = super::scene::js_number(spacing);
            anyhow::ensure!(
                p > 0.0,
                "spacing={spacing}: a lot pitch in metres, or bench"
            );
            Some(p as f32)
        };
        let pool = match q.get("pool") {
            Some("none") => None,
            Some(p) => Some(super::scene::js_number(p) as u32),
            None => Some(if pitch.is_none() { 32768 } else { 131072 }),
        };
        let kill_below = match q.get("kill") {
            Some("off") => None,
            Some(k) => Some(super::scene::js_number(k) as f32),
            None => Some(-50.0),
        };
        let scenario = q.get("scenario").unwrap_or("still").to_string();
        anyhow::ensure!(
            ["still", "drive", "war"].contains(&scenario.as_str()),
            "scenario={scenario}: still, drive, war"
        );
        Ok(CitySceneDef {
            houses: houses as u32,
            sleep: q.get("sleep") != Some("0"),
            pool,
            pitch,
            kill_below,
            scenario,
        })
    }

    /// [`crate::city`]'s definition: the map a `ceil(sqrt(K))²` grid of `pitch` m lots.
    pub fn city_def(&self) -> CityDef {
        let mut d = CityDef {
            houses: self.houses as usize,
            sleep: self.sleep,
            pool: self.pool,
            ..CityDef::default()
        };
        if let Some(p) = self.pitch {
            d.map = (self.houses as f64).sqrt().ceil() as f32 * p;
        }
        d
    }
}

pub fn world_def(def: &CitySceneDef) -> anyhow::Result<WorldDef> {
    let mut wd = def.city_def().world_def(&city_house()?)?;
    if let Some(y) = def.kill_below {
        wd.kill_bounds = Some(KillBounds::below(y));
    }
    Ok(wd)
}

/// Every house placed; the scenario starts later ([`super::Action::Start`]).
pub fn place(world: &mut World, def: &CitySceneDef) -> anyhow::Result<City> {
    let mut city = City::begin(world, &city_house()?, def.city_def())?;
    while city.remaining() > 0 {
        city.place(world, 64)?;
        world.settle_uploads();
    }
    Ok(city)
}

/// The city's actions: shells and its scenario.
pub struct CityDirector {
    city: City,
    scenario: CityScenario,
    pub shells: u32,
}

impl CityDirector {
    pub(crate) fn new(def: &CitySceneDef, city: City) -> anyhow::Result<Self> {
        Ok(CityDirector {
            city,
            scenario: CityScenario::from_name(&def.scenario)?,
            shells: 0,
        })
    }

    pub(crate) fn act(
        &mut self,
        world: &mut World,
        action: &super::Action,
        ev: &mut Vec<super::Event>,
    ) -> anyhow::Result<()> {
        use super::director::note;
        match action {
            super::Action::Start => {
                self.city.start(world, self.scenario)?;
                note(ev, world, "start");
                Ok(())
            }
            super::Action::Shell {
                origin,
                velocity,
                radius,
            } => {
                crate::city::city_shell(world, *origin, *velocity, *radius)?;
                self.shells += 1;
                note(ev, world, "shell");
                Ok(())
            }
            a => anyhow::bail!("the city has no {} action", a.name()),
        }
    }
}
