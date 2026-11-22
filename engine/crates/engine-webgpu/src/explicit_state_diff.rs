//! Two explicit states compared word by word (the A/Bs' verdict lines).

/// Two [`super::StagedState::ab_words`] compared: one line per run of consecutive differing words
/// (its name, the run, the first pair), at most `cap` lines; empty when bit for bit.
#[doc(hidden)]
pub fn state_differences(a: &[(&'static str, Vec<u32>)], b: &[(&'static str, Vec<u32>)], cap: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut more = 0usize;
    for ((name, x), (name_b, y)) in a.iter().zip(b) {
        if name != name_b || x.len() != y.len() {
            out.push(format!("{name}: {} words / {name_b}: {} words", x.len(), y.len()));
            continue;
        }
        let total = x.iter().zip(y).filter(|(p, q)| p != q).count();
        let mut i = 0;
        while i < x.len() {
            if x[i] == y[i] {
                i += 1;
                continue;
            }
            let start = i;
            while i < x.len() && x[i] != y[i] {
                i += 1;
            }
            if out.len() < cap {
                out.push(format!(
                    "{name} [{start}..{i}] of {} ({total} differ in all): {:#x} / {:#x} ({} / {})",
                    x.len(),
                    x[start],
                    y[start],
                    f32::from_bits(x[start]),
                    f32::from_bits(y[start])
                ));
            } else {
                more += 1;
            }
        }
    }
    if a.len() != b.len() {
        out.push(format!("{} runs of words / {}", a.len(), b.len()));
    }
    if more > 0 {
        out.push(format!("… {more} more runs"));
    }
    out
}
