//! GPU capability detection, classification, and driver safety.
//!
//! Submodules:
//! - [`types`]: Enums and structs for device tier, GPU family, platform, quirks
//! - [`profile`]: `CapabilityProfile` — the single capability snapshot
//! - [`classify`]: Tier, platform, and memory model classification
//! - [`detect`]: GPU family detection from vendor IDs and device names
//! - [`quirks`]: Driver blocklist and safety checks

mod classify;
mod detect;
pub mod profile;
pub mod quirks;
pub(crate) mod scoring;
pub mod types;

pub use profile::{CapabilityProfile, RenderAttachmentPolicy};
pub use quirks::{driver_quirks, is_gpu_usable};
pub use types::{
    DetectionSource, DeviceTier, DriverQuirks, GpuFamily, MemoryModel, PlatformClass,
};
