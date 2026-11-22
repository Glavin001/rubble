//! A kernel module's WGSL by name: [`Kernel`] is what the crate's kernel constants and
//! tables hold. The text is embedded in the binary (release, the C library, wasm), or,
//! in the fast profile on native, compiled at run time on first use into a store keyed
//! by the Slang sources ([`crate::kernel_store`]), so a kernel edit there compiles no
//! Rust. The same compile (src/kernel_build.rs) either way: the same bytes.

/// A kernel module: its name (the WGSL file's) and its text ([`Kernel::text`], or the
/// `str` it dereferences to).
#[derive(Clone, Copy)]
pub struct Kernel {
    name: &'static str,
    // (Empty with the kernels compiled at run time.)
    #[cfg_attr(kernels_at_runtime, allow(dead_code))]
    embedded: &'static str,
}

impl Kernel {
    /// A module embedded in the binary.
    pub const fn embedded(name: &'static str, text: &'static str) -> Self {
        Self { name, embedded: text }
    }

    /// A module compiled at run time ([`crate::kernel_store`]).
    pub const fn runtime(name: &'static str) -> Self {
        Self { name, embedded: "" }
    }

    pub const fn name(&self) -> &'static str {
        self.name
    }

    /// The module's WGSL.
    pub fn text(&self) -> &'static str {
        #[cfg(kernels_at_runtime)]
        {
            crate::kernel_store::wgsl(self.name)
        }
        #[cfg(not(kernels_at_runtime))]
        {
            self.embedded
        }
    }
}

impl std::ops::Deref for Kernel {
    type Target = str;
    fn deref(&self) -> &str {
        self.text()
    }
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Kernel({})", self.name)
    }
}

impl PartialEq<str> for Kernel {
    fn eq(&self, other: &str) -> bool {
        self.text() == other
    }
}

/// `kernel!("name")`: the module `name.wgsl` of this build, embedded or at run time.
#[cfg(not(kernels_at_runtime))]
macro_rules! kernel {
    ($name:literal) => {
        $crate::kernel::Kernel::embedded($name, include_str!(concat!(env!("OUT_DIR"), "/wgsl/", $name, ".wgsl")))
    };
}
#[cfg(kernels_at_runtime)]
macro_rules! kernel {
    ($name:literal) => {
        $crate::kernel::Kernel::runtime($name)
    };
}
pub(crate) use kernel;
