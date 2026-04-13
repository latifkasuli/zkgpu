//! Conditional Send + Sync bounds for cross-platform compatibility.
//!
//! On native targets, GPU types must be `Send + Sync` for safe
//! multi-threaded use. On wasm32 (single-threaded), these bounds
//! are impossible to satisfy for wgpu's `WebBuffer` (which uses
//! `RefCell` internally), so we relax them.

/// Marker trait that equals `Send + Sync` on native, no-op on wasm32.
#[cfg(not(target_arch = "wasm32"))]
pub trait MaybeSendSync: Send + Sync {}

#[cfg(not(target_arch = "wasm32"))]
impl<T: Send + Sync> MaybeSendSync for T {}

/// Marker trait that equals nothing on wasm32 (single-threaded).
#[cfg(target_arch = "wasm32")]
pub trait MaybeSendSync {}

#[cfg(target_arch = "wasm32")]
impl<T> MaybeSendSync for T {}
