//! Goldilocks GPU NTT module.
//!
//! # Status after Phase B.3
//!
//! - [`plan::WgpuGoldilocksNttPlan`] — concrete, portable-u32x2 Stockham
//!   NTT plan for [`zkgpu_goldilocks::Goldilocks`]. Dispatches the R4
//!   kernel when `log_n` is even, falls back to R2 when odd. Forward +
//!   inverse, up to `MAX_GOLDILOCKS_LOG_N = 31`. 2D-folded dispatch via
//!   [`crate::dispatch::plan_linear_dispatch`] so large `log_n` respect
//!   WebGPU's baseline `max_compute_workgroups_per_dimension = 65535`.
//!   Canary-validated on Metal + Vulkan against
//!   [`zkgpu_ntt::ntt_cpu_reference::<Goldilocks>`] at log_n ∈
//!   {3,4,5,6,10,18} and GPU-determinism at log_n = 20.
//! - [`resolve`] — kernel-variant resolver (Auto / Portable /
//!   NativeVulkan). `Auto` always resolves to `PortableU32x2` until an
//!   allowlist of proven `(backend, gpu_family, driver)` fingerprints
//!   for the native-int64 path lands.
//! - [`arith_test`] — Phase B.1 differential-test harness for the
//!   portable u32x2 arithmetic primitives.
//!
//! Not yet wired into [`WgpuNttPlan`] — Phase E adds a `field`
//! parameter to the harness and routes Goldilocks suites here.
//! `WgpuGoldilocksNttPlan::execute()` is also blocking; an
//! async/browser-safe variant lands alongside Phase E.
//!
//! [`WgpuNttPlan`]: crate::WgpuNttPlan
//! [`zkgpu_ntt::ntt_cpu_reference::<Goldilocks>`]: zkgpu_ntt::ntt_cpu_reference
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
//! # Why this matters
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

pub use plan::{WgpuGoldilocksNttPlan, MAX_GOLDILOCKS_LOG_N};
