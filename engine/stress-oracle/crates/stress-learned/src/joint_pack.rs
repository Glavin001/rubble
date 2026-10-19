//! Pack / unpack `JointState` into the schema's 22-float vector.

use stress_ref::joint::JointState;

use crate::constants::JOINT_STATE_DIM;

/// Finite ductility is stored as-is (clamped). Non-finite (= unset / infinite in the
/// oracle) becomes 0 so the network is not fed 1e6 sentinels.
fn pack_ductility(x: f64) -> f32 {
    if x.is_finite() { (x as f32).clamp(0.0, 1.0e3) } else { 0.0 }
}

/// Pack oracle joint state → `[JOINT_STATE_DIM]` f32 (schema order).
pub fn pack_joint_state(st: &JointState) -> [f32; JOINT_STATE_DIM] {
    let mut o = [0.0f32; JOINT_STATE_DIM];
    o[0] = st.damage as f32;
    o[1] = st.crush as f32;
    o[2] = st.kappa as f32;
    o[3] = st.kappa_c as f32;
    o[4] = pack_ductility(st.ductility);
    o[5] = pack_ductility(st.ductility_c);
    o[6] = st.fatigue as f32;
    o[7] = st.plastic.lin[0] as f32;
    o[8] = st.plastic.lin[1] as f32;
    o[9] = st.plastic.lin[2] as f32;
    o[10] = st.plastic.ang[0] as f32;
    o[11] = st.plastic.ang[1] as f32;
    o[12] = st.plastic.ang[2] as f32;
    o[13] = st.rebar_plastic as f32;
    o[14] = st.rebar_slip[0] as f32;
    o[15] = st.rebar_slip[1] as f32;
    o[16] = st.rebar_work as f32;
    o[17] = if st.rebar_broken { 1.0 } else { 0.0 };
    o[18] = st.strain_rate as f32;
    o[19] = st.governing_stress as f32;
    o[20] = st.dissipated as f32;
    o[21] = st.utilization as f32;
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use stress_ref::joint::JointState;

    #[test]
    fn pack_round_utilization() {
        let mut st = JointState::new();
        st.utilization = 1.25;
        st.damage = 0.5;
        let p = pack_joint_state(&st);
        assert_eq!(p.len(), JOINT_STATE_DIM);
        assert!((p[21] - 1.25).abs() < 1e-6);
        assert!((p[0] - 0.5).abs() < 1e-6);
        // JointState::new leaves ductility = ∞ → packed as 0.
        assert_eq!(p[4], 0.0);
        assert_eq!(p[5], 0.0);
    }
}
