//! Compile-time layout validation for GPU structs.
//!
//! Ensures all repr(C) types match expected WGSL storage buffer layouts.

/// Assert that a type has the expected size at compile time.
macro_rules! assert_gpu_layout {
    ($ty:ty, $expected_size:expr, $expected_align:expr) => {
        const _: () = {
            assert!(
                std::mem::size_of::<$ty>() == $expected_size,
                concat!(
                    "GPU layout mismatch: ",
                    stringify!($ty),
                    " size mismatch"
                )
            );
            assert!(
                std::mem::align_of::<$ty>() >= $expected_align,
                concat!(
                    "GPU layout mismatch: ",
                    stringify!($ty),
                    " alignment mismatch"
                )
            );
        };
    };
}

pub(crate) use assert_gpu_layout;
