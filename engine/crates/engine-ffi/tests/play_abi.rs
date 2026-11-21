//! A preset through the C ABI: created by name with page switches over it, stepped, its
//! counts read; an unknown preset or one placed elsewhere fails with its reason.

use rubble_ffi::play::*;
use rubble_ffi::*;
use std::ffi::{CStr, CString};

fn last_error() -> String {
    unsafe {
        CStr::from_ptr(rbGetLastError())
            .to_string_lossy()
            .into_owned()
    }
}

#[test]
fn a_preset_steps_through_the_c_abi() {
    unsafe {
        let mut play: *mut RbPlay = std::ptr::null_mut();
        let name = CString::new("bullets").unwrap();
        let query = CString::new("lookahead=0").unwrap();
        assert_eq!(
            rbCreatePlay(name.as_ptr(), query.as_ptr(), &mut play),
            RbResult::RB_OK,
            "{}",
            last_error()
        );
        assert_eq!(rbPlay_Step(play, 5), RbResult::RB_OK, "{}", last_error());
        let mut s = RbPlayStats::default();
        assert_eq!(rbPlay_GetStats(play, &mut s), RbResult::RB_OK);
        assert_eq!(s.steps, 5);
        assert_eq!(s.bodies, 9);
        assert_eq!(s.overflow, 0);
        rbDestroyPlay(play);

        for bad in ["no-such-preset", "lab"] {
            let name = CString::new(bad).unwrap();
            assert_ne!(
                rbCreatePlay(name.as_ptr(), std::ptr::null(), &mut play),
                RbResult::RB_OK
            );
            assert!(!last_error().is_empty(), "{bad}");
        }
    }
}
