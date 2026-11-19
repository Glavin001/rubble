//! The explicit vehicles' kernels without strict float arithmetic: the WGSL of the
//! fast setting (`bit_exact` off; `vehicle-perf-first-principles.md` §8.38).
//!
//! The kernels' strict helpers (`slang/physics/strict.slang`) give IEEE single precision
//! in the CPU reference's order: every intermediate passes through `nf`, an XOR with a
//! uniform the compilers cannot know is zero, and division and square root take one
//! exact fma correction. That is what makes the GPU's results the CPU reference's bit
//! for bit on every target. It also doubles every dependent arithmetic chain (§8.35: a
//! dependent ALU step is about 8 ns on one workgroup).
//!
//! With bit-exact results off (the default since 2026-11-19, the owner's decision:
//! bit-identical results are for testing and reproduction), the host creates its
//! pipelines from this rewrite of the same modules: `nf(x)` returns `x`, `sdiv(a, b)` is
//! `a / b` and `ssqrt(x)` is `sqrt(x)`. The equations, their operands and every branch
//! stay as written; only the rounding of the intermediates is the compiler's (it may
//! now fuse a multiply and an add, or reassociate). `noFuseBits` and `opaqueCount` keep
//! their uniform: `opaqueCount` keeps a loop a loop for the drivers' compile time
//! (`kernel-compile-time.md`), not for the bits.
//!
//! Strict mode's modules are not touched: the host passes the build's WGSL as before.

use std::borrow::Cow;

/// The strict helpers this rewrite replaces, as (Slang's WGSL name, the new body from
/// the parameters' names).
const REWRITES: [(&str, fn(&[&str]) -> String); 3] = [
    ("nf_0", |p| format!("return {};", p[0])),
    ("sdiv_0", |p| format!("return {} / {};", p[0], p[1])),
    ("ssqrt_0", |p| format!("return sqrt({});", p[0])),
];

/// `src` (an explicit module's WGSL, before or after the uniform-load rewrite) with the
/// strict helpers' bodies replaced (see the module's documentation). A module without
/// them comes back unchanged (borrowed). Panics if a helper's definition is not in the
/// shape Slang emits: a change of the emitter must fail loudly, not run strict code
/// under the fast setting's name.
pub fn fast_float_wgsl(src: &str) -> Cow<'_, str> {
    if !src.contains("fn nf_0(") {
        return Cow::Borrowed(src);
    }
    let mut out = src.to_string();
    for (name, body) in REWRITES {
        let head = format!("fn {name}(");
        let Some(at) = out.find(&head) else { continue };
        let open_paren = at + head.len() - 1;
        let close_paren = open_paren
            + out[open_paren..]
                .find(')')
                .unwrap_or_else(|| panic!("fast_float_wgsl: {name}'s parameters"));
        let params: Vec<&str> = out[open_paren + 1..close_paren]
            .split(',')
            .map(|p| p.split(':').next().unwrap_or("").trim())
            .collect();
        assert!(
            params.iter().all(|p| !p.is_empty()),
            "fast_float_wgsl: {name}'s parameters {params:?}"
        );
        let open = close_paren
            + out[close_paren..]
                .find('{')
                .unwrap_or_else(|| panic!("fast_float_wgsl: {name}'s body"));
        let mut depth = 0usize;
        let mut close = None;
        for (i, c) in out[open..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(open + i);
                        break;
                    }
                }
                _ => {}
            }
        }
        let close = close.unwrap_or_else(|| panic!("fast_float_wgsl: {name}'s body is not closed"));
        let new_body = format!("{{\n    {}\n}}", body(&params));
        out.replace_range(open..=close, &new_body);
        assert_eq!(
            out.matches(&head).count(),
            1,
            "fast_float_wgsl: {name} defined more than once"
        );
    }
    Cow::Owned(out)
}

/// The engine-wide default of the bit-exact setting for a World and the vehicle lab
/// (`ExplicitVehiclesDef::bit_exact`, `LabOptions::bit_exact`): off, unless the
/// environment's `RUBBLE_BIT_EXACT` is `1` (the bit-for-bit harnesses set it:
/// `perf-verify.sh`, `fast-check.sh`, the scenario matrix, the Chrome A/B). Native only;
/// in a browser it is off.
pub fn bit_exact_default() -> bool {
    std::env::var("RUBBLE_BIT_EXACT").as_deref() == Ok("1")
}

/// A module's WGSL for the given setting: `src` itself with bit-exact results on,
/// [`fast_float_wgsl`] of it off.
pub fn explicit_wgsl(src: &str, bit_exact: bool) -> Cow<'_, str> {
    if bit_exact {
        Cow::Borrowed(src)
    } else {
        fast_float_wgsl(src)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STRICT: &str = "fn noFuseBits_0() -> u32
{
    return ((gP_0.vehicles_0) >> (u32(31)));
}

fn nf_0( x_1 : f32) -> f32
{
    return (bitcast<f32>((((bitcast<u32>((x_1))) ^ ((noFuseBits_0()))))));
}

fn ssqrt_0( x_2 : f32) -> f32
{
    if(x_2 <= 0.0f)
    {
        return sqrt(x_2);
    }
    var s_0 : f32 = nf_0(sqrt(x_2));
    return nf_0(fma(fma(- s_0, s_0, x_2), 0.5f / s_0, s_0));
}

fn sdiv_0( a_5 : f32,  b_5 : f32) -> f32
{
    var r_0 : f32 = 1.0f / b_5;
    var q_0 : f32 = nf_0(a_5 * r_0);
    return nf_0(fma(fma(- q_0, b_5, a_5), r_0, q_0));
}

fn opaqueCount_0( n_5 : u32) -> u32
{
    return n_5 + noFuseBits_0();
}
";

    #[test]
    fn rewrites_the_three_helpers_only() {
        let fast = fast_float_wgsl(STRICT);
        assert!(fast.contains("fn nf_0( x_1 : f32) -> f32\n{\n    return x_1;\n}"));
        assert!(fast.contains("fn ssqrt_0( x_2 : f32) -> f32\n{\n    return sqrt(x_2);\n}"));
        assert!(
            fast.contains("fn sdiv_0( a_5 : f32,  b_5 : f32) -> f32\n{\n    return a_5 / b_5;\n}")
        );
        // The uniform stays for opaqueCount (compile time, not bits).
        assert!(fast.contains("return n_5 + noFuseBits_0();"));
        assert!(fast.contains("return ((gP_0.vehicles_0) >> (u32(31)));"));
        assert!(!fast.contains("fma("));
    }

    #[test]
    fn strict_setting_and_plain_modules_are_untouched() {
        assert!(matches!(explicit_wgsl(STRICT, true), Cow::Borrowed(s) if s == STRICT));
        let plain = "fn main() { }";
        assert!(matches!(fast_float_wgsl(plain), Cow::Borrowed(s) if s == plain));
    }

    /// Every shipped explicit module either has no strict helpers or has all of `nf`
    /// rewritten, and the rewrite leaves no XOR with `noFuseBits` behind.
    #[test]
    fn every_shipped_module_rewrites() {
        for (name, _, wgsl) in crate::explicit_beams_shipped_wgsl() {
            let fast = fast_float_wgsl(&wgsl);
            assert!(
                !fast.contains("^ ((noFuseBits_0()))"),
                "{name}: an nf XOR left"
            );
            if wgsl.contains("fn nf_0(") {
                assert!(fast.contains("fn nf_0("), "{name}");
            }
        }
    }
}
