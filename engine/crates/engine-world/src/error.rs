//! Errors a host can match on. Everything else is a plain `anyhow` error with
//! a message.

/// A definition field or call the solver does not support yet. Returned
/// instead of silently ignoring the request; downcast with
/// `err.downcast_ref::<Unsupported>()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unsupported(pub &'static str);

impl std::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "not supported yet: {}", self.0)
    }
}

impl std::error::Error for Unsupported {}

pub(crate) fn unsupported<T>(what: &'static str) -> anyhow::Result<T> {
    Err(Unsupported(what).into())
}

/// A handle that is null, was removed, or belongs to a reused slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaleHandle(pub &'static str, pub u64);

impl std::fmt::Display for StaleHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stale or invalid {} handle {:#x}", self.0, self.1)
    }
}

impl std::error::Error for StaleHandle {}
