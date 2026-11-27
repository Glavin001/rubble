//! Which of a scenario's moments and cameras a reference frame is compared with, when the
//! frame does not name them: from its moment label ("peak crush") and view label ("side").
//! (Was `sheet.py`'s rule; the studio's server, `sheet.py` and `outlines.py` all ask here.)

use super::super::{Camera, Moment};

/// A reference moment's time after first contact (ms), to find the nearest moment.
const MOMENT_MS: &[(&str, f32)] = &[
    ("before", -10.0),
    ("setup", -10.0),
    ("first contact", -10.0),
    ("close first contact", -10.0),
    ("early crush", 20.0),
    ("mid crush", 40.0),
    ("peak crush", 60.0),
    ("peak", 60.0),
    ("peak intrusion", 70.0),
    ("rebound", 150.0),
    ("peak pitch", 250.0),
    ("rotation", 300.0),
];
/// A reference moment named by the scenario, when it has it.
const MOMENT_NAME: &[(&str, &str)] = &[
    ("first contact", "contact"),
    ("early crush", "early_crush"),
    ("mid crush", "mid_crush"),
    ("peak crush", "peak"),
    ("peak intrusion", "peak"),
    ("rebound", "rebound"),
];
/// Per scenario: a moment whose instant the generic table misses (S07's "peak" is the car
/// standing on its nose, not the 60 ms peak of crush).
const MOMENT_OVERRIDE: &[(&str, &str, &str)] = &[
    ("S07", "peak", "on_its_nose"),
    ("S10", "peak pitch", "rear_up"),
];
/// A reference view's camera, when the names differ (first that the scenario has).
const CAMERA_ALIAS: &[(&str, &[&str])] = &[
    ("wide", &["wide", "front_quarter", "side"]),
    ("side close", &["side close", "side"]),
    (
        "three-quarter",
        &[
            "three-quarter",
            "aftermath",
            "quarter",
            "front three-quarter",
        ],
    ),
    (
        "three-quarter rear",
        &[
            "three-quarter rear",
            "aftermath_rear",
            "aftermath",
            "quarter",
        ],
    ),
    (
        "front three-quarter",
        &["front three-quarter", "front_quarter", "quarter"],
    ),
];

/// Lower case, runs of space, '_' and '-' as one space.
pub fn norm(s: &str) -> String {
    s.trim()
        .to_lowercase()
        .split(|c: char| c.is_whitespace() || c == '_' || c == '-')
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn pick_camera<'a>(view: &str, cams: &'a [Camera]) -> Option<&'a str> {
    let v = norm(view);
    let alias = CAMERA_ALIAS
        .iter()
        .find(|(k, _)| norm(k) == v)
        .map_or(&[][..], |(_, a)| *a);
    std::iter::once(view)
        .chain(alias.iter().copied())
        .find_map(|cand| {
            cams.iter()
                .find(|c| norm(&c.name) == norm(cand))
                .map(|c| c.name.as_str())
        })
        .or(cams.first().map(|c| c.name.as_str()))
}

pub fn pick_moment<'a>(scenario: &str, label: &str, moments: &'a [Moment]) -> Option<&'a str> {
    let key = norm(label);
    let by = |name: &str| moments.iter().find(|m| m.name == name).map(|m| m.name.as_str());
    if let Some(m) = MOMENT_OVERRIDE
        .iter()
        .find(|(s, k, _)| s.eq_ignore_ascii_case(scenario) && norm(k) == key)
        .and_then(|(_, _, m)| by(m))
    {
        return Some(m);
    }
    if key == "aftermath" {
        return moments
            .iter()
            .max_by(|a, b| a.after_impact_ms.total_cmp(&b.after_impact_ms))
            .map(|m| m.name.as_str());
    }
    if let Some(m) = MOMENT_NAME
        .iter()
        .find(|(k, _)| *k == key)
        .and_then(|(_, n)| by(n))
    {
        return Some(m);
    }
    let ms = MOMENT_MS
        .iter()
        .find(|(k, _)| *k == key)
        .map_or(60.0, |(_, v)| *v);
    moments
        .iter()
        .min_by(|a, b| {
            (a.after_impact_ms - ms)
                .abs()
                .total_cmp(&(b.after_impact_ms - ms).abs())
        })
        .map(|m| m.name.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moments() -> Vec<Moment> {
        vec![
        Moment {
            name: "before".into(),
            after_impact_ms: -10.0,
        },
        Moment {
            name: "peak".into(),
            after_impact_ms: 70.0,
        },
        Moment {
            name: "on_its_nose".into(),
            after_impact_ms: 400.0,
        },
        Moment {
            name: "rest".into(),
            after_impact_ms: 900.0,
        },
    ]
    }
    fn cams() -> Vec<Camera> {
        vec![
        Camera {
            name: "side".into(),
            eye: [0.0; 3],
            target: [1.0, 0.0, 0.0],
        },
        Camera {
            name: "front_quarter".into(),
            eye: [0.0; 3],
            target: [1.0, 0.0, 0.0],
        },
    ]
    }

    #[test]
    fn moments_by_name_by_time_and_by_override() {
        assert_eq!(pick_moment("S01", "peak crush", &moments()), Some("peak"));
        assert_eq!(pick_moment("S01", "first contact", &moments()), Some("before"));
        assert_eq!(pick_moment("S01", "Before", &moments()), Some("before"));
        assert_eq!(pick_moment("S01", "aftermath", &moments()), Some("rest"));
        assert_eq!(pick_moment("S07", "Peak", &moments()), Some("on_its_nose"));
        assert_eq!(pick_moment("S01", "peak", &moments()), Some("peak"));
    }

    #[test]
    fn cameras_by_name_and_alias() {
        assert_eq!(pick_camera("Side", &cams()), Some("side"));
        assert_eq!(pick_camera("wide", &cams()), Some("front_quarter"));
        assert_eq!(
            pick_camera("front-three-quarter", &cams()),
            Some("front_quarter")
        );
        assert_eq!(pick_camera("top", &cams()), Some("side"));
    }
}
