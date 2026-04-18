//! Goldilocks GPU NTT module.
//!
//! # Phase status
//!
//! **Phase A (this commit): scaffolding only.** The resolver lives in
//! [`resolve`], `FieldStorageAbi` lives in
//! [`crate::field_codec`], but there is **no Goldilocks NTT plan yet**.
//! Nothing here is reachable from [`WgpuNttPlan`]; everything is
//! `pub(crate)` and exists so that Phase B can slot in the portable
//! `u32x2` Stockham kernels + `WgpuGoldilocksNttPlan` without a second
//! round of structural churn.
//!
//! [`WgpuNttPlan`]: crate::WgpuNttPlan
//!
//! # Design spec
//!
//! This module implements **option C** from the architecture decision:
//! native-int64 kernels may exist as an opt-in fast path on Vulkan
//! (behind the `goldilocks-vulkan-int64` Cargo feature), but the
//! default — and the only supported path on the web / wasm target —
//! is a portable `u32x2` limb representation. The choice is resolved
//! once at plan-construction time and encoded in the plan; shaders
//! never branch on variant at runtime.
//!
//! # Why this matters now
//!
//! - WGSL (10-March-2026 CRD) defines only `i32` / `u32` scalar
//!   integer types. There is no standard `u64` shader type today.
//! - WebGPU's feature registry contains `shader-f16` and `subgroups`,
//!   but no `shader-int64`.
//! - `wgpu::Features::SHADER_INT64` exists but is documented as
//!   native-only (Vulkan / DX12-DXC / Metal 2.3+). Browser-WebGPU
//!   sessions can't rely on it.
//!
//! A Goldilocks kernel that leans on native `u64` math therefore
//! breaks the project's browser/portable goal. The resolver keeps
//! the portable limb kernel as the mandatory baseline and treats
//! native int64 as a proven-on-a-specific-(backend, driver) fast path.

pub(crate) mod arith_test;
pub(crate) mod plan;
pub(crate) mod resolve;

pub use plan::WgpuGoldilocksNttPlan;
