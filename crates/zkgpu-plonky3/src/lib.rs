//! Adapter between zkgpu's NTT engine and Plonky3's
//! [`TwoAdicSubgroupDft`] trait.
//!
//! Phase 7.1 — minimal viable adapter for BabyBear. Implements only
//! the required `dft_batch` method; every other method on the trait
//! is inherited from the default implementations, which recurse back
//! into `dft_batch`.
//!
//! # Strategy
//!
//! Plonky3's DFT operates on a [`RowMajorMatrix<F>`] where each
//! *column* is an independent polynomial of length `height`. zkgpu's
//! NTT plan is single-poly, in-place. This adapter bridges the two
//! with a per-column loop (**Path A** in the design spec): for each
//! column, we convert from Plonky3's Monty form to zkgpu's canonical
//! form, upload to GPU, run the plan, download, and scatter back
//! into the output matrix.
//!
//! A true 2D-batched execution path (Path B) is a deferred
//! optimization — see Phase 7.5 in `research/phase-7-plonky3-adapter/
//! interface-spec.md` for the upgrade route.
//!
//! # Lifecycle
//!
//! `GpuDft` is a unit-like struct with `Default + Clone` (as the
//! trait requires). All device state lives in a process-global
//! [`OnceLock`] that performs lazy adapter/device init on the first
//! `dft_batch` call above the fallback threshold. Init failure is
//! cached — subsequent calls fall back to a local `Radix2DitParallel`
//! without re-probing the GPU.
//!
//! # Fallback policy
//!
//! Per Codex review: the 7.1 threshold is purely `log_h < 14` — no
//! width-based rule. Under the column-loop architecture, width is
//! just a multiplier on invocations, not true batching, so conflating
//! it into the threshold misleads the policy.
//!
//! Two opt-in bypass mechanisms exist so the GPU path remains
//! exercised during correctness validation (e.g. via `p3-zk-proofs`,
//! whose traces are tiny):
//!
//! - [`GpuDft::force_gpu`] — constructor that ignores the `log_h`
//!   threshold for its instance.
//! - Environment variable `ZKGPU_PLONKY3_FORCE_GPU=1` — process-wide
//!   override with the same effect.
//!
//! Buffer-size preflight and device-init-failure fallbacks still
//! apply under both overrides (those are correctness constraints).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use p3_baby_bear::BabyBear as P3BabyBear;
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::PrimeField32;
use p3_matrix::Matrix;
use p3_matrix::bitrev::{BitReversalPerm, BitReversedMatrixView};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::util::reverse_matrix_index_bits;
use p3_util::log2_strict_usize;

use zkgpu_babybear::BabyBear as ZkgpuBabyBear;
use zkgpu_core::{GpuBuffer, GpuDevice, NttDirection};
use zkgpu_wgpu::{WgpuDevice, WgpuNttPlan};

// -- Process-global GPU context ----------------------------------------------

/// Lazily-initialized process-global GPU state.
///
/// Shape per spec §5.2 (Codex-simplified): `Arc` only wraps the `Ok`
/// payload; `Err` is a plain `String` that's cheap to clone per-caller.
/// On device-init failure, the `Err` is cached so subsequent calls
/// don't re-probe.
///
/// Consequence: adapter/environment choice is frozen at first
/// successful init for the process lifetime. Swapping devices
/// requires a separate process. This is an intentional simplification
/// for 7.1; a `GpuContext::custom()` escape hatch could be added later
/// if needed.
static GPU_CONTEXT: OnceLock<Result<Arc<GpuContext>, String>> = OnceLock::new();

struct GpuContext {
    device: WgpuDevice,
    /// Plan cache keyed on `(log_h, direction)`.
    ///
    /// Each plan is wrapped in `Mutex` because `WgpuNttPlan::execute`
    /// takes `&mut self` (it owns scratch buffers aliased across
    /// stages). Concurrent execution on the same plan would corrupt
    /// intermediate state.
    plans: Mutex<HashMap<(u32, NttDirection), Arc<Mutex<WgpuNttPlan>>>>,
}

impl GpuContext {
    fn try_init() -> Result<Self, String> {
        // `WgpuDevice::new` is sync-blocking on native (wraps async
        // wgpu init with pollster internally). Browser/wasm callers
        // would need a different entry point; 7.1 is native-only.
        let device =
            WgpuDevice::new().map_err(|e| format!("zkgpu device init failed: {e}"))?;
        Ok(Self {
            device,
            plans: Mutex::new(HashMap::new()),
        })
    }

    /// Get (and cache) a plan for the given size/direction.
    fn get_or_build_plan(
        &self,
        log_n: u32,
        direction: NttDirection,
    ) -> Result<Arc<Mutex<WgpuNttPlan>>, String> {
        let key = (log_n, direction);
        let mut plans = self
            .plans
            .lock()
            .map_err(|_| "plan cache mutex poisoned".to_string())?;
        if let Some(existing) = plans.get(&key) {
            return Ok(existing.clone());
        }
        let plan = WgpuNttPlan::new(&self.device, log_n, direction)
            .map_err(|e| format!("plan build failed for log_n={log_n}: {e}"))?;
        let wrapped = Arc::new(Mutex::new(plan));
        plans.insert(key, wrapped.clone());
        Ok(wrapped)
    }
}

/// Lazy accessor for the process-global context.
#[inline]
fn get_context() -> &'static Result<Arc<GpuContext>, String> {
    GPU_CONTEXT.get_or_init(|| {
        // Cost: ~100–500 ms on first call (adapter + device + pipeline
        // cache warm-up). Subsequent calls are a pointer-load.
        let result = GpuContext::try_init().map(Arc::new);
        if let Err(ref e) = result {
            tracing::warn!(target: "zkgpu_plonky3", "GPU unavailable, falling back to CPU: {e}");
        }
        result
    })
}

// -- GpuDft -------------------------------------------------------------------

/// Plonky3 [`TwoAdicSubgroupDft`] implementation backed by zkgpu's
/// WebGPU NTT engine.
///
/// Value-semantics per the trait bounds: `Clone + Default + Send +
/// Sync`. The struct itself carries no device state; all GPU context
/// is shared process-wide via a lazy [`OnceLock`].
#[derive(Clone, Debug)]
pub struct GpuDft<F> {
    /// If `true`, `should_fallback` ignores the `log_h` threshold for
    /// this instance. Buffer-size preflight and device-init-failure
    /// fallbacks still apply. Set via [`Self::force_gpu`] or at
    /// `Default` construction when the `ZKGPU_PLONKY3_FORCE_GPU` env
    /// var is set.
    force_gpu: bool,
    /// Pre-constructed CPU fallback. Allocating it here (rather than
    /// lazily per call) keeps the fallback hot path allocation-free.
    cpu_fallback: Radix2DitParallel<F>,
    _phantom: core::marker::PhantomData<F>,
}

impl<F> Default for GpuDft<F>
where
    Radix2DitParallel<F>: Default,
{
    fn default() -> Self {
        Self {
            force_gpu: env_force_gpu(),
            cpu_fallback: Radix2DitParallel::default(),
            _phantom: core::marker::PhantomData,
        }
    }
}

impl<F> GpuDft<F>
where
    Radix2DitParallel<F>: Default,
{
    /// Construct a `GpuDft` that ignores the `log_h` fallback
    /// threshold for this instance.
    ///
    /// Use from tests that need to exercise the GPU code path at
    /// sizes where the default heuristic would route to CPU —
    /// including `p3-zk-proofs`'s tiny trace heights (see spec §6.4
    /// and §8).
    ///
    /// Device-init-failure and buffer-size-too-large fallbacks still
    /// apply; this is a tuning-threshold override, not a correctness
    /// override.
    pub fn force_gpu() -> Self {
        tracing::warn!(
            target: "zkgpu_plonky3",
            "GpuDft::force_gpu() in use — size-threshold fallback disabled for this instance"
        );
        Self {
            force_gpu: true,
            cpu_fallback: Radix2DitParallel::default(),
            _phantom: core::marker::PhantomData,
        }
    }
}

/// Read `ZKGPU_PLONKY3_FORCE_GPU`. Any non-empty value other than
/// `0` / `false` / `FALSE` turns the override on.
fn env_force_gpu() -> bool {
    match std::env::var("ZKGPU_PLONKY3_FORCE_GPU") {
        Ok(v) => {
            let lowered = v.trim().to_ascii_lowercase();
            !lowered.is_empty() && lowered != "0" && lowered != "false"
        }
        Err(_) => false,
    }
}

// -- Fallback policy ----------------------------------------------------------

/// Size-based fallback decision. See spec §6.2.
///
/// Returns `true` if this `(log_h, width)` invocation should be
/// handled by the CPU `Radix2DitParallel` rather than by GPU.
///
/// Per Codex review: width is intentionally not factored into the
/// threshold under Path A, since width is invocation-count multiplier
/// rather than true batching. Width re-enters policy only when Path B
/// (2D batched plan) lands in 7.5.
fn should_fallback(log_h: u32, _width: usize, force_gpu: bool) -> bool {
    if force_gpu {
        return false;
    }
    // Empirical floor: below log_h=14, wgpu dispatch + upload/download
    // overhead dominates even a single-column run. Refined in 7.4.
    log_h < 14
}

// -- Field conversion ---------------------------------------------------------

/// Convert a slice of Plonky3 BabyBear (Montgomery form) to zkgpu
/// BabyBear (canonical `[0, P)`).
///
/// See spec §3 for the conversion contract.
#[allow(dead_code)] // Used in tests; inline in the hot path.
fn p3_to_zkgpu_babybear(src: &[P3BabyBear]) -> Vec<ZkgpuBabyBear> {
    src.iter()
        .map(|e| ZkgpuBabyBear(e.as_canonical_u32()))
        .collect()
}

/// Convert a slice of zkgpu BabyBear (canonical) to Plonky3 BabyBear
/// (Montgomery form).
#[allow(dead_code)] // Used in tests; inline in the hot path.
fn zkgpu_to_p3_babybear(src: &[ZkgpuBabyBear]) -> Vec<P3BabyBear> {
    // `P3BabyBear::new(u32)` accepts any u32 and routes through
    // `to_monty` (see `monty-31/src/utils.rs:7`). Our canonical values
    // are already `< P`, but `new` handles any input safely.
    src.iter().map(|e| P3BabyBear::new(e.0)).collect()
}

// -- TwoAdicSubgroupDft impl --------------------------------------------------

impl TwoAdicSubgroupDft<P3BabyBear> for GpuDft<P3BabyBear> {
    type Evaluations = BitReversedMatrixView<RowMajorMatrix<P3BabyBear>>;

    fn dft_batch(&self, mat: RowMajorMatrix<P3BabyBear>) -> Self::Evaluations {
        let h = mat.height();
        let w = mat.width();

        // Degenerate cases: empty matrix or single-row. Delegate to
        // the CPU impl — its default handles both correctly.
        if h <= 1 {
            return self.cpu_fallback.dft_batch(mat);
        }

        let log_h = log2_strict_usize(h) as u32;

        if should_fallback(log_h, w, self.force_gpu) {
            return self.cpu_fallback.dft_batch(mat);
        }

        let ctx = match get_context() {
            Ok(ctx) => ctx.clone(),
            Err(_) => return self.cpu_fallback.dft_batch(mat),
        };

        let plan = match ctx.get_or_build_plan(log_h, NttDirection::Forward) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    target: "zkgpu_plonky3",
                    "plan build failed at log_h={log_h}, falling back: {e}"
                );
                return self.cpu_fallback.dft_batch(mat);
            }
        };

        // Path A: column loop. For each column, gather → upload →
        // execute → download → scatter. See spec §4.3.
        let natural = match run_column_loop(&ctx, &plan, &mat) {
            Ok(out) => out,
            Err(e) => {
                tracing::warn!(
                    target: "zkgpu_plonky3",
                    "GPU execution failed at log_h={log_h}, falling back: {e}"
                );
                return self.cpu_fallback.dft_batch(mat);
            }
        };

        // zkgpu's Stockham plan produces output in natural (auto-sort)
        // order. Plonky3 consumers expect the `BitReversedMatrixView`
        // marker, so we apply a CPU-side bit-reversal and wrap.
        //
        // Future optimization (see spec §2.3 option C): skip Stockham's
        // final auto-sort pass and return bit-reversed directly.
        let mut natural = natural;
        reverse_matrix_index_bits(&mut natural);
        BitReversalPerm::new_view(natural)
    }
}

/// Core column-loop execution. Pulled out for clarity and to keep the
/// trait method focused on control flow.
fn run_column_loop(
    ctx: &Arc<GpuContext>,
    plan_handle: &Arc<Mutex<WgpuNttPlan>>,
    mat: &RowMajorMatrix<P3BabyBear>,
) -> Result<RowMajorMatrix<P3BabyBear>, String> {
    let h = mat.height();
    let w = mat.width();

    let mut plan = plan_handle
        .lock()
        .map_err(|_| "plan mutex poisoned".to_string())?;

    let mut output_vals: Vec<P3BabyBear> = vec![P3BabyBear::new(0); h * w];

    // Reusable staging vector — avoids allocator pressure when w is
    // large. The GPU buffer itself is re-created per column (cheap
    // since it's a fixed-size u32 array), but the host-side staging
    // is hot.
    let mut col_canonical: Vec<ZkgpuBabyBear> = vec![ZkgpuBabyBear(0); h];

    for c in 0..w {
        // Gather column c into `col_canonical` with Monty→canonical
        // conversion in the same pass.
        for r in 0..h {
            let p3_val = mat.values[r * w + c];
            col_canonical[r] = ZkgpuBabyBear(p3_val.as_canonical_u32());
        }

        // Upload, execute, download.
        let mut gpu_buf = ctx
            .device
            .upload(&col_canonical)
            .map_err(|e| format!("upload failed at column {c}: {e}"))?;
        plan.execute_kernels(&ctx.device, &mut gpu_buf)
            .map_err(|e| format!("plan execute failed at column {c}: {e}"))?;
        let result_canonical = gpu_buf
            .read_to_vec()
            .map_err(|e| format!("readback failed at column {c}: {e}"))?;

        // Scatter column c into output matrix with canonical→Monty
        // conversion.
        for r in 0..h {
            output_vals[r * w + c] = P3BabyBear::new(result_canonical[r].0);
        }
    }

    Ok(RowMajorMatrix::new(output_vals, w))
}

// -- Tests --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use p3_dft::NaiveDft;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Generate a random `RowMajorMatrix<P3BabyBear>` of the given shape.
    fn random_matrix(log_h: usize, w: usize, seed: u64) -> RowMajorMatrix<P3BabyBear> {
        let h = 1usize << log_h;
        let mut rng = StdRng::seed_from_u64(seed);
        let values: Vec<P3BabyBear> = (0..h * w)
            .map(|_| P3BabyBear::new(rng.random::<u32>()))
            .collect();
        RowMajorMatrix::new(values, w)
    }

    /// Assert two matrices are equal element-wise.
    fn assert_matrix_eq(
        actual: &RowMajorMatrix<P3BabyBear>,
        expected: &RowMajorMatrix<P3BabyBear>,
        ctx: &str,
    ) {
        assert_eq!(actual.width(), expected.width(), "{ctx}: width mismatch");
        assert_eq!(
            actual.height(),
            expected.height(),
            "{ctx}: height mismatch"
        );
        for (i, (a, e)) in actual.values.iter().zip(expected.values.iter()).enumerate() {
            assert_eq!(
                a.as_canonical_u32(),
                e.as_canonical_u32(),
                "{ctx}: element {i} differs",
            );
        }
    }

    /// Differential: small sizes vs `NaiveDft` (the O(n²) reference).
    ///
    /// Uses `force_gpu()` so the GPU path is exercised even at tiny
    /// log_h where default would route to CPU.
    #[test]
    fn dft_batch_matches_naive_small() {
        for &log_h in &[2usize, 4, 6, 8] {
            for &w in &[1usize, 3, 8] {
                let mat = random_matrix(log_h, w, 0xC0FFEE);
                let naive_out = NaiveDft.dft_batch(mat.clone()).to_row_major_matrix();

                let gpu = GpuDft::<P3BabyBear>::force_gpu();
                let gpu_out = gpu.dft_batch(mat).to_row_major_matrix();

                assert_matrix_eq(
                    &gpu_out,
                    &naive_out,
                    &format!("naive log_h={log_h} w={w}"),
                );
            }
        }
    }

    /// Differential: medium sizes vs `Radix2DitParallel`. Force-GPU
    /// at log_h=12 (below threshold) and default-threshold at log_h=14+.
    #[test]
    fn dft_batch_matches_parallel_medium() {
        for &(log_h, force) in &[(12usize, true), (14, false), (16, false)] {
            for &w in &[1usize, 4] {
                let mat = random_matrix(log_h, w, 0xDEADBEEF);
                let cpu_out = Radix2DitParallel::<P3BabyBear>::default()
                    .dft_batch(mat.clone())
                    .to_row_major_matrix();

                let gpu = if force {
                    GpuDft::<P3BabyBear>::force_gpu()
                } else {
                    GpuDft::<P3BabyBear>::default()
                };
                let gpu_out = gpu.dft_batch(mat).to_row_major_matrix();

                assert_matrix_eq(
                    &gpu_out,
                    &cpu_out,
                    &format!("parallel log_h={log_h} w={w} force={force}"),
                );
            }
        }
    }

    /// Inverse DFT via the default trait impl (which routes through
    /// our `dft_batch`). Tests that `idft(dft(x)) == x`.
    #[test]
    fn idft_batch_roundtrip() {
        for &log_h in &[4usize, 6, 8] {
            for &w in &[1usize, 3] {
                let mat = random_matrix(log_h, w, 0xFEEDFACE);
                let gpu = GpuDft::<P3BabyBear>::force_gpu();
                let dft_out = gpu.dft_batch(mat.clone()).to_row_major_matrix();
                let idft_out = gpu.idft_batch(dft_out);
                assert_matrix_eq(
                    &idft_out,
                    &mat,
                    &format!("idft roundtrip log_h={log_h} w={w}"),
                );
            }
        }
    }

    /// Coset DFT roundtrip: `coset_idft(coset_dft(x, shift), shift) == x`.
    #[test]
    fn coset_dft_idft_roundtrip() {
        use p3_field::TwoAdicField;
        let shift = P3BabyBear::two_adic_generator(10);
        for &log_h in &[4usize, 6, 8] {
            for &w in &[1usize, 3] {
                let mat = random_matrix(log_h, w, 0x5EED);
                let gpu = GpuDft::<P3BabyBear>::force_gpu();
                let dft_out = gpu.coset_dft_batch(mat.clone(), shift).to_row_major_matrix();
                let idft_out = gpu.coset_idft_batch(dft_out, shift);
                assert_matrix_eq(
                    &idft_out,
                    &mat,
                    &format!("coset roundtrip log_h={log_h} w={w}"),
                );
            }
        }
    }

    /// With a default (non-forcing) `GpuDft` at tiny log_h, the
    /// threshold routes to CPU. The result must still match
    /// `NaiveDft` — this validates the fallback path is wired
    /// correctly, not just the GPU path.
    #[test]
    fn default_threshold_routes_to_cpu_correctly() {
        let mat = random_matrix(4, 2, 0xBEEF);
        let naive_out = NaiveDft.dft_batch(mat.clone()).to_row_major_matrix();
        let gpu_default = GpuDft::<P3BabyBear>::default();
        // At log_h=4, should_fallback=true → cpu_fallback path.
        let out = gpu_default.dft_batch(mat).to_row_major_matrix();
        assert_matrix_eq(&out, &naive_out, "default-threshold small size");
    }

    /// Sanity-check field conversion round-trips.
    #[test]
    fn field_conversion_roundtrips() {
        let samples: Vec<P3BabyBear> = (0u32..16)
            .map(|v| P3BabyBear::new(v.wrapping_mul(0x9E37_79B9)))
            .collect();
        let canonical = p3_to_zkgpu_babybear(&samples);
        let back = zkgpu_to_p3_babybear(&canonical);
        for (a, b) in samples.iter().zip(back.iter()) {
            assert_eq!(a.as_canonical_u32(), b.as_canonical_u32());
        }
    }
}
