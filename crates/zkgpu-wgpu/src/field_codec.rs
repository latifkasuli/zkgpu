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
//! crate which layout to assume for a given plan. It's deliberately
//! non-generic over the field itself: the plan carries the ABI
//! explicitly so shaders, buffers, and upload/download code can agree
//! without threading a type parameter through every layer.

/// How a field element is stored in a GPU storage buffer.
///
/// Chosen per-plan, not per-field: Goldilocks plans may ship both a
/// portable `Limb32x2Le` variant and (eventually, under the
/// `goldilocks-vulkan-int64` Cargo feature) a `NativeRepr` variant that
/// relies on `wgpu::Features::SHADER_INT64`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldStorageAbi {
    /// The shader sees the same bytes as `F::Repr` — one u32 for
    /// BabyBear / KoalaBear; the native 64-bit shader type for a
    /// future native-int64 Goldilocks variant.
    NativeRepr,

    /// The shader sees one `vec2<u32>` per element, laid out
    /// little-endian: `[lo32, hi32]` so that
    /// `value = lo + (hi << 32)`.
    ///
    /// Element byte width is always 8. Uniform encoding for any
    /// 64-bit constant (twiddles, inverse-scale factors, etc.) uses
    /// the same two-word layout in the same order.
    Limb32x2Le,
}

impl FieldStorageAbi {
    /// Byte width of a single element in this ABI. Always a multiple
    /// of 4 so bind-group alignment rules are satisfied without
    /// additional padding.
    ///
    /// Dead in Phase A; the Goldilocks plan (Phase B) is the first
    /// caller — it needs this to size the on-GPU scratch buffers.
    #[allow(dead_code)]
    pub const fn element_size_bytes(self) -> u32 {
        match self {
            Self::NativeRepr => 4,
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
            Self::NativeRepr => "NativeRepr",
            Self::Limb32x2Le => "Limb32x2Le",
        }
    }
}
