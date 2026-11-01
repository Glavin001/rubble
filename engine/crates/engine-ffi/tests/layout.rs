//! The header's struct layouts are Rust's: a C program built against `include/rubble.h`
//! prints sizes, alignments and field offsets of the destruction-reaction types, and they
//! must equal `size_of`, `align_of` and `offset_of!` here. Needs a C compiler (`cc`, or
//! `$CC`); without one the test fails with that reason rather than passing unchecked.

use rubble_ffi::*;
use std::mem::{align_of, offset_of, size_of};
use std::process::Command;

/// (type, field or "" for the type itself, Rust's value: size and align, or offset).
fn rust_layout() -> Vec<(&'static str, &'static str, usize, usize)> {
    macro_rules! ty {
        ($t:ident) => {
            (stringify!($t), "", size_of::<$t>(), align_of::<$t>())
        };
    }
    macro_rules! field {
        ($t:ident, $f:ident) => {
            (stringify!($t), stringify!($f), offset_of!($t, $f), 0)
        };
    }
    vec![
        ty!(RbBlastDef),
        field!(RbBlastDef, position),
        field!(RbBlastDef, tnt_kg),
        field!(RbBlastDef, radial),
        field!(RbBlastDef, radius),
        field!(RbBlastDef, falloff),
        field!(RbBlastDef, impulse_per_area),
        field!(RbBlastDef, mask_bits),
        field!(RbBlastDef, internal_value),
        ty!(RbCrushMaterial),
        field!(RbCrushMaterial, name),
        field!(RbCrushMaterial, cap_pressure),
        field!(RbCrushMaterial, cohesion),
        field!(RbCrushMaterial, friction_slope),
        field!(RbCrushMaterial, crush_energy),
        field!(RbCrushMaterial, crush_viscosity),
        ty!(RbCrushDef),
        field!(RbCrushDef, enabled),
        field!(RbCrushDef, mass),
        field!(RbCrushDef, fragments),
        field!(RbCrushDef, budget),
        field!(RbCrushDef, max_per_step),
        field!(RbCrushDef, materials),
        field!(RbCrushDef, material_count),
        ty!(RbDestructibleDef),
        field!(RbDestructibleDef, support_loss),
        field!(RbDestructibleDef, support_loss_dynamic_factor),
        field!(RbDestructibleDef, blast_loading),
        field!(RbDestructibleDef, crush),
        field!(RbDestructibleDef, internal_value),
        ty!(RbDestructionEvent),
        field!(RbDestructionEvent, type_),
        field!(RbDestructionEvent, step),
        field!(RbDestructionEvent, mass),
        field!(RbDestructionEvent, bonds),
        field!(RbDestructionEvent, children),
        field!(RbDestructionEvent, name),
        field!(RbDestructionEvent, break_cause),
        field!(RbDestructionEvent, crush_cause),
        field!(RbDestructionEvent, volume),
        field!(RbDestructionEvent, transform),
        field!(RbDestructionEvent, linear_velocity),
        field!(RbDestructionEvent, angular_velocity),
        field!(RbDestructionEvent, peak_pressure),
        field!(RbDestructionEvent, peak_deviator),
        ty!(RbBreakCause),
        ty!(RbCrushCause),
        ty!(RbCrushMass),
    ]
}

#[test]
fn the_headers_layouts_are_rusts() {
    let rows = rust_layout();
    let mut c = String::from(
        "#include <stdio.h>\n#include <stddef.h>\n#include \"rubble.h\"\nint main(void) {\n",
    );
    for (t, f, _, _) in &rows {
        if f.is_empty() {
            c += &format!("  printf(\"{t} - %zu %zu\\n\", sizeof({t}), _Alignof({t}));\n");
        } else {
            c += &format!("  printf(\"{t} {f} %zu 0\\n\", offsetof({t}, {f}));\n");
        }
    }
    c += "  return 0;\n}\n";
    let dir = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("rubble_layout");
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("layout.c");
    let exe = dir.join("layout");
    std::fs::write(&src, c).unwrap();
    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".into());
    let include = concat!(env!("CARGO_MANIFEST_DIR"), "/include");
    let built = Command::new(&cc)
        .args(["-std=c11", "-I", include, "-o"])
        .arg(&exe)
        .arg(&src)
        .output()
        .unwrap_or_else(|e| panic!("the layout test needs a C compiler ({cc}): {e}"));
    assert!(
        built.status.success(),
        "{cc} failed:\n{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let out = Command::new(&exe).output().unwrap();
    let text = String::from_utf8(out.stdout).unwrap();
    let mut wrong = Vec::new();
    for ((t, f, a, b), line) in rows.iter().zip(text.lines()) {
        let w: Vec<&str> = line.split(' ').collect();
        let (ca, cb): (usize, usize) = (w[2].parse().unwrap(), w[3].parse().unwrap());
        if (ca, cb) != (*a, *b) {
            wrong.push(format!("{t}.{f}: C ({ca}, {cb}), Rust ({a}, {b})"));
        }
    }
    assert_eq!(text.lines().count(), rows.len());
    assert!(
        wrong.is_empty(),
        "layouts differ:\n  {}",
        wrong.join("\n  ")
    );
}
