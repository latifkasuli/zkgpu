//! Field storage ABI — how a field element is laid out in GPU storage.
//!
//! Most of `zkgpu-wgpu` has always assumed that the on-GPU representation
//! of a field element is bit-identical to `F::Repr`: `WgpuDevice::upload`
//! casts `&[F]` to bytes via `bytemuck`, and `WgpuBuffer::download` maps
//! back the same way. That's fine for BabyBear and KoalaBear (both
//! `Repr = u32`), and WGSL has a native `u32`, so the shader sees the
//! same 32-bit word the Rust side produced.
//!
//! Goldilocks breaks that assumption. `Repr = u64`, and WGSL has no
//! standard `u64` type (the 10-March-2026 spec defines only `i32` and
//! `u32`; `shader-f16` is the only extra numeric feature in the current
//! WebGPU REC). Until a `shader-int64` shader feature appears, a
//! portable Goldilocks kernel has to represent each field element as a
//! pair of 32-bit limbs — canonically `vec2<u32>` on the shader side.
//!
//! `FieldStorageAbi` is the single knob that tells the rest of the
//! crate which layout to assume for a given plan. It is deliberately
//! scoped to the layouts the Goldilocks plan needs today — if a
//! future BabyBear/KoalaBear path starts carrying this enum, add a
//! `NativeU32` variant at that point (don't speculate now).

/// How a field element is stored in a GPU storage buffer.
///
/// Scope: currently only the Goldilocks plan carries this. Both
/// variants use **8 bytes per element**; they differ only in how the
/// shader interprets those bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldStorageAbi {
    /// Native `u64` per element. Used by the future native-int64
    /// Goldilocks variant (behind `goldilocks-vulkan-int64`). Requires
    /// `wgpu::Features::SHADER_INT64` at device creation.
    NativeU64,

    /// One `vec2<u32>` per element, laid out little-endian:
    /// `[lo32, hi32]` so `value = lo + (hi << 32)`. Used by the
    /// portable Goldilocks kernel path. Works on every WebGPU target
    /// including browser / wasm.
    Limb32x2Le,
}

impl FieldStorageAbi {
    /// Byte width of a single element in this ABI.
    ///
    /// Both current variants are 8 bytes (a `u64` and a `vec2<u32>`
    /// are the same size). The method stays for forward compatibility
    /// — if a `NativeU32` variant is added for BabyBear/KoalaBear,
    /// the caller's storage-sizing code still works unchanged.
    ///
    /// Dead in Phase A; the Goldilocks plan (Phase B) is the first
    /// caller — it needs this to size the on-GPU scratch buffers.
    #[allow(dead_code)]
    pub const fn element_size_bytes(self) -> u32 {
        match self {
            Self::NativeU64 => 8,
            Self::Limb32x2Le => 8,
        }
    }

    /// Stable human-readable tag, used in report JSON. Matches the
    /// strings consumed by the Goldilocks resolver and test harness.
    ///
    /// Dead in Phase A; Phase E (harness/report integration) is the
    /// first caller.
    #[allow(dead_code)]
    pub const fn label(self) -> &'static str {
        match self {
            Self::NativeU64 => "NativeU64",
            Self::Limb32x2Le => "Limb32x2Le",
        }
    }
}
