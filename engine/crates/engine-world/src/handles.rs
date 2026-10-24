//! Generational handles (Box3D `b3BodyId` / `b3ShapeId` / `b3JointId`): a
//! slot index and the generation it was handed out with, packed in a u64.
//! Destroying an object bumps its slot's generation, so every copy of its
//! handle is refused instead of reaching whatever reuses the slot. Generation
//! 0 is never issued, so a zeroed handle is null.

use engine_schema::Handle;

macro_rules! handle {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name(pub(crate) u64);

        impl $name {
            pub const NULL: Self = Self(0);

            pub(crate) fn new(index: u32, generation: u32) -> Self {
                debug_assert!(generation != 0);
                Self(u64::from(generation) << 32 | u64::from(index))
            }

            /// The slot index (stable for the handle's lifetime).
            pub fn index(self) -> u32 {
                self.0 as u32
            }

            pub fn generation(self) -> u32 {
                (self.0 >> 32) as u32
            }

            pub fn is_null(self) -> bool {
                self.generation() == 0
            }

            /// Pack into a `u64` (user data, FFI, saving) and back (Box3D
            /// `b3StoreBodyId` / `b3LoadBodyId`).
            pub fn to_bits(self) -> u64 {
                self.0
            }

            pub fn from_bits(bits: u64) -> Self {
                Self(bits)
            }

            pub fn handle(self) -> Handle {
                Handle::new(self.index(), self.generation())
            }
        }
    };
}

handle!(
    /// A rigid body.
    BodyId
);
handle!(
    /// A shape (collider) attached to a body.
    ShapeId
);
handle!(
    /// A joint, including breakable bonds.
    JointId
);
