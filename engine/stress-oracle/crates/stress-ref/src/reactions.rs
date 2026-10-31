//! A scenario's destruction reactions: the world API's per-destructible switches (rubble
//! `DestructibleDef::support_loss`) a pack asks its host for, so a scene that needs the
//! weight its breaks leave without a path to break what it overloads says so itself
//! instead of each page and test choosing.
//!
//! A pack carries them as `defaults.reactions` ([`write`], read with [`reactions_of`]):
//!
//! ```json
//! {"supportLoss": true, "supportLossFactor": 1.0}
//! ```
//!
//! * `supportLoss`: after a frame's breaks, each anchored piece's weight flows down its
//!   intact bonds to its anchors and an overloaded bond breaks the next frame.
//!   `supportLossFactor` (1..2, default 1) is its dynamic factor.
//!
//! No `reactions`: off, the threshold model as the lab runs it. (Crushing and blast
//! loading are not carried yet: the scenario hosts neither draw a crushed chunk's
//! fragments nor detonate.)

use serde_json::{json, Value};

/// A pack's reactions (module doc).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reactions {
    pub support_loss: bool,
    pub support_loss_factor: f64,
}

impl Default for Reactions {
    fn default() -> Self {
        Reactions { support_loss: false, support_loss_factor: 1.0 }
    }
}

/// Write `reactions` into a pack as `defaults.reactions` (the default: the key is left out).
pub fn write(pack: &mut Value, reactions: &Reactions) {
    if *reactions != Reactions::default() {
        pack["defaults"]["reactions"] = json!({"supportLoss": reactions.support_loss, "supportLossFactor": reactions.support_loss_factor});
    }
}

/// The reactions a pack asks for (`defaults.reactions`; the default without them). An error
/// names a dynamic factor outside 1..2 and a reaction the hosts do not carry.
pub fn reactions_of(pack: &Value) -> Result<Reactions, String> {
    let r = &pack["defaults"]["reactions"];
    if r.is_null() {
        return Ok(Reactions::default());
    }
    if let Some(k) = r.as_object().and_then(|o| o.keys().find(|k| !["supportLoss", "supportLossFactor"].contains(&k.as_str()))) {
        return Err(format!("reactions: '{k}' is not carried (supportLoss and supportLossFactor are)"));
    }
    let support_loss = r["supportLoss"].as_bool().unwrap_or(false);
    let support_loss_factor = r["supportLossFactor"].as_f64().unwrap_or(1.0);
    if !(1.0..=2.0).contains(&support_loss_factor) {
        return Err(format!("reactions: supportLossFactor {support_loss_factor} outside 1..2"));
    }
    if support_loss_factor != 1.0 && !support_loss {
        return Err("reactions: supportLossFactor needs supportLoss".into());
    }
    Ok(Reactions { support_loss, support_loss_factor })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reactions_round_trip_through_a_pack() {
        let mut pack = json!({});
        write(&mut pack, &Reactions::default());
        assert!(pack.get("defaults").is_none(), "the default writes nothing");
        assert_eq!(reactions_of(&pack).unwrap(), Reactions::default());
        let r = Reactions { support_loss: true, support_loss_factor: 1.5 };
        write(&mut pack, &r);
        assert_eq!(reactions_of(&pack).unwrap(), r);
    }

    #[test]
    fn bad_reactions_are_errors() {
        let bad = |r: Value| reactions_of(&json!({"defaults": {"reactions": r}})).unwrap_err();
        assert!(bad(json!({"supportLoss": true, "supportLossFactor": 3.0})).contains("outside 1..2"));
        assert!(bad(json!({"supportLossFactor": 1.5})).contains("needs supportLoss"));
        assert!(bad(json!({"supportLoss": true, "crush": {}})).contains("'crush' is not carried"));
    }
}
