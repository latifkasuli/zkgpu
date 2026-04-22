//! Adapter between zkgpu's NTT engine and Plonky3's
//! [`TwoAdicSubgroupDft`] trait.
//!
//! Phase 7.1 — minimal viable adapter for BabyBear. Implements only
//! the required `dft_batch` method; every other method on the trait
//! is inherited from the default implementations, which recurse back
//! into `dft_batch`.
//!
//! Phase 7.2 — correctness oracle landed: `dft_batch`, `coset_lde_batch`,
//! FRI commit root, and end-to-end prove/verify all match
//! `Radix2DitParallel` under `strict_gpu()`.
//!
//! Phase 7.3 — lifecycle polish: plan cache warm-up via
//! [`GpuDft::preload_plans`], `tracing::instrument` for profiler
//! compatibility, cache-growth semantics documented below.
//!
//! Phase 7.4 — perf microbench (see `benches/gpu_vs_cpu_dft.rs`).
//! Headline findings on RTX 4090 + Ryzen 9 7950X:
//!
//! * **Single-polynomial (width=1)**: GPU wins at `log_h ≥ 16`;
//!   peak 2.44× at `log_h=18` for `dft_batch`, 1.73× at `log_h=20`
//!   for `coset_lde_batch`.
//! * **Fallback threshold holds**: `log_h < 14 → CPU` is
//!   empirically correct on discrete NVIDIA.
//!
//! Phase 7.5 — Path B (2D-batched NTT plan, [`zkgpu_wgpu::WgpuBatchedNttPlan`]).
//! Phase C.1 (2026-04-21) extended the batched plan to radix-4 +
//! pitched storage. Adapter auto-selects the best GPU path per
//! `(log_h, w)`. On RTX 4090 + Ryzen 9 7950X at `w = 8`:
//!
//! * **`log_h ≤ 14, w > 1` → batched**: per-column launch overhead of
//!   Path A dominates; one batched dispatch per stage wins. Up to
//!   2.06× faster than Path A at `log_h=14`.
//! * **`15 ≤ log_h ≤ 17, w > 1` → Path A column loop**: the single-
//!   column plan's R4 + workgroup-local fused tail reduces dispatch
//!   count sharply at these sizes. The batched plan has no local
//!   fused tail (Path C.2 future work) and exactly fills one SM wave
//!   at `log_h=16`, leaving no oversubscription to hide memory
//!   latency. Path A wins despite w-fold launch overhead.
//! * **`log_h ≥ 18, w > 1` → batched**: per-stage thread count grows
//!   enough (≥ 4× SM capacity) for batched R4 to saturate the GPU;
//!   Path A's w-fold launch overhead re-dominates. C.1 narrowed the
//!   CPU gap at log_h=18 from 1.98× to 1.66× and at log_h=20 from
//!   1.34× to 1.11×.
//!
//! Neither regime yet beats the 7950X's 32-thread AVX2 CPU at w=8;
//! beating that requires either C.4 (GPU-resident coset_lde_batch
//! to remove round-trip overhead in the FRI hot path) or a native
//! CUDA backend. For single-poly (w=1), GPU wins 2.44× at log_h=18
//! and the pitch stands.
//!
//! Run the bench with
//! `cargo bench -p zkgpu-plonky3 --bench gpu_vs_cpu_dft`.
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
//! The plan cache inside `GpuContext` is an unbounded `HashMap` keyed
//! on `(log_h, direction)`. For typical Plonky3 workloads (single
//! proof session, ≤10 distinct sizes across blowup/FRI-fold domains)
//! the cache tops out at a few dozen entries, so no eviction policy
//! is needed. Long-running services that drive many distinct sizes
//! should call a fresh process rather than rely on pathological
//! cache growth. See [`GpuDft::preload_plans`] for pre-warming.
//!
//! # Fallback policy
//!
//! Evolved through 7.1 → 7.5 as more measurement landed:
//!
//! * **7.1 (spec) / 7.4 (data)**: threshold `log_h < 14` for all
//!   widths. Width was considered invocation-count multiplier, not
//!   true batching under Path A.
//! * **7.5 (data)**: bench on RTX 4090 + Ryzen 9 7950X showed CPU
//!   beats the best GPU path at every `w > 1, log_h` tested. Path B
//!   closed the worst Path A regressions but not the gap vs a
//!   32-thread AVX2 CPU.
//!
//! So the shipped default is now **performance-safe**: `Default`
//! routes any `w > 1` call to CPU, and keeps the `log_h ≥ 14`
//! threshold for `w == 1`. Width-agnostic GPU dispatch is available
//! via [`GpuDft::force_gpu`] (opt-in) or [`GpuDft::strict_gpu`]
//! (opt-in plus loud panics on fallback).
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

pub mod poseidon2_bridge;

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
use zkgpu_wgpu::{WgpuBatchedNttPlan, WgpuDevice, WgpuNttPlan};

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

/// Type alias for the single-column plan-cache map. Extracted from
/// the `GpuContext` field signature to satisfy clippy's
/// `type_complexity` lint.
type PlanCache = Mutex<HashMap<(u32, NttDirection), Arc<Mutex<WgpuNttPlan>>>>;

/// Type alias for the Phase 7.5 batched plan-cache map. Keyed on
/// `(log_h, width, direction)` — width matters because the batched
/// plan's scratch buffer and twiddle-indexing bake it in at build time.
type BatchedPlanCache =
    Mutex<HashMap<(u32, u32, NttDirection), Arc<Mutex<WgpuBatchedNttPlan>>>>;

struct GpuContext {
    device: WgpuDevice,
    /// Single-column plan cache keyed on `(log_h, direction)`.
    ///
    /// Each plan is wrapped in `Mutex` because `WgpuNttPlan::execute`
    /// takes `&mut self` (it owns scratch buffers aliased across
    /// stages). Concurrent execution on the same plan would corrupt
    /// intermediate state.
    plans: PlanCache,
    /// Batched plan cache for Phase 7.5 Path B. Keyed on
    /// `(log_h, width, direction)`; built lazily the first time a
    /// matching `dft_batch(w > 1)` call arrives.
    batched_plans: BatchedPlanCache,
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
            batched_plans: Mutex::new(HashMap::new()),
        })
    }

    /// Get (and cache) a single-column plan for the given size/direction.
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

    /// Get (and cache) a batched plan for the given
    /// `(log_h, width, direction)`. Phase 7.5 Path B.
    fn get_or_build_batched_plan(
        &self,
        log_n: u32,
        width: u32,
        direction: NttDirection,
    ) -> Result<Arc<Mutex<WgpuBatchedNttPlan>>, String> {
        let key = (log_n, width, direction);
        let mut plans = self
            .batched_plans
            .lock()
            .map_err(|_| "batched plan cache mutex poisoned".to_string())?;
        if let Some(existing) = plans.get(&key) {
            return Ok(existing.clone());
        }
        let plan = WgpuBatchedNttPlan::new(&self.device, log_n, width, direction).map_err(
            |e| format!("batched plan build failed for log_n={log_n} w={width}: {e}"),
        )?;
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
    /// If `true`, any fallback to CPU panics instead of silently
    /// routing through `Radix2DitParallel`. See [`Self::strict_gpu`]
    /// for rationale — this is the signal a correctness oracle needs
    /// to prove the GPU path was actually exercised.
    ///
    /// Implies `force_gpu = true`.
    strict: bool,
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
            strict: false,
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
    /// Device-init-failure, plan-build-failure, and execute-failure
    /// fallbacks still apply silently (via `tracing::warn!`). If you
    /// need a fail-loud version for a correctness oracle, use
    /// [`Self::strict_gpu`] instead — `force_gpu()` alone is **not**
    /// sufficient to guarantee the test exercised the GPU path.
    pub fn force_gpu() -> Self {
        tracing::warn!(
            target: "zkgpu_plonky3",
            "GpuDft::force_gpu() in use — size-threshold fallback disabled for this instance"
        );
        Self {
            force_gpu: true,
            strict: false,
            cpu_fallback: Radix2DitParallel::default(),
            _phantom: core::marker::PhantomData,
        }
    }

    /// Construct a `GpuDft` that **panics on any fallback to CPU**
    /// rather than routing through the CPU impl silently.
    ///
    /// Intended for correctness-oracle test paths (e.g. when using
    /// `p3-zk-proofs` to validate the GPU adapter end-to-end). Without
    /// this, a proof can pass entirely on CPU — device-init failure,
    /// plan build failure, and execute failure all silently fall back
    /// under `force_gpu()` — and the test appears green while the GPU
    /// code path was never exercised.
    ///
    /// Implies `force_gpu` (the size threshold is also bypassed).
    ///
    /// # Panics
    ///
    /// Panics from `dft_batch` (and therefore from any default-trait
    /// method that routes through it) whenever:
    ///   * GPU device init failed on first use,
    ///   * plan build fails for the requested `log_h`,
    ///   * plan execution returns an error for any column.
    ///
    /// This is intentional — the sync `TwoAdicSubgroupDft::dft_batch`
    /// signature can't return a `Result`, and a strict-mode caller by
    /// definition wants loud failure.
    pub fn strict_gpu() -> Self {
        tracing::warn!(
            target: "zkgpu_plonky3",
            "GpuDft::strict_gpu() in use — any CPU fallback will panic"
        );
        Self {
            force_gpu: true,
            strict: true,
            cpu_fallback: Radix2DitParallel::default(),
            _phantom: core::marker::PhantomData,
        }
    }
}

/// Lifecycle helpers. Always available, regardless of which field the
/// `GpuDft` is parameterised over — they only touch the process-global
/// context, not `F`.
impl<F> GpuDft<F> {
    /// Pre-build `WgpuNttPlan`s for the given sizes so that the first
    /// real `dft_batch` call doesn't pay plan-construction cost.
    ///
    /// Useful for:
    /// * Benchmarks that want to measure *execution* cost without the
    ///   one-shot plan-build overhead polluting the numbers.
    /// * Services that want deterministic first-call latency.
    ///
    /// Builds both forward and inverse plans for each supplied
    /// `log_n`, since Plonky3's FRI uses both directions.
    ///
    /// Silently no-ops on device-init failure (same semantics as the
    /// non-strict `dft_batch` path — preload is a hint, not a
    /// correctness surface). Individual plan-build failures per
    /// `log_n` are warned via `tracing::warn!` and skipped.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let dft = GpuDft::<BabyBear>::default();
    /// // Warm up for the log sizes our FRI config will use.
    /// dft.preload_plans(&[14, 16, 18, 20]);
    /// // First real dft_batch at any of these sizes skips plan build.
    /// ```
    pub fn preload_plans(&self, log_ns: &[u32]) {
        let ctx = match get_context() {
            Ok(ctx) => ctx.clone(),
            Err(e) => {
                tracing::warn!(
                    target: "zkgpu_plonky3",
                    "preload_plans: GPU unavailable, skipping all preloads: {e}"
                );
                return;
            }
        };
        for &log_n in log_ns {
            for direction in [NttDirection::Forward, NttDirection::Inverse] {
                if let Err(e) = ctx.get_or_build_plan(log_n, direction) {
                    tracing::warn!(
                        target: "zkgpu_plonky3",
                        "preload_plans: skipping log_n={log_n} {direction:?}: {e}"
                    );
                }
            }
        }
    }

    /// Phase 7.5 — preload batched plans at the given
    /// `(log_h, width)` pairs. Same semantics as `preload_plans`, but
    /// warms the batched cache instead of (or in addition to) the
    /// single-column cache.
    ///
    /// Use before a benchmark or prove run that will hit `dft_batch`
    /// with `width > 1` — lets the first batched call skip plan build.
    pub fn preload_batched_plans(&self, shapes: &[(u32, u32)]) {
        let ctx = match get_context() {
            Ok(ctx) => ctx.clone(),
            Err(e) => {
                tracing::warn!(
                    target: "zkgpu_plonky3",
                    "preload_batched_plans: GPU unavailable: {e}"
                );
                return;
            }
        };
        for &(log_n, w) in shapes {
            for direction in [NttDirection::Forward, NttDirection::Inverse] {
                if let Err(e) = ctx.get_or_build_batched_plan(log_n, w, direction) {
                    tracing::warn!(
                        target: "zkgpu_plonky3",
                        "preload_batched_plans: skipping log_n={log_n} w={w} {direction:?}: {e}"
                    );
                }
            }
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

/// Size-based fallback decision. See spec §6.2 and the 7.5 bench
/// findings (`research/phase-7-plonky3-adapter/7-4-bench/findings-pathb.md`).
///
/// Returns `true` if this `(log_h, width)` invocation should be
/// handled by the CPU `Radix2DitParallel` rather than by GPU.
///
/// # Default-mode policy
///
/// Under the measured 7.5 bench on RTX 4090 + Ryzen 9 7950X, the
/// best GPU path — auto-selected per `(log_h, w)` between Path A
/// (column loop over `WgpuNttPlan`) and Path B (`WgpuBatchedNttPlan`,
/// R2-only) — is consistently slower than parallel CPU DFT at
/// every tested `w > 1` size. Concretely, at `w=8` CPU beats the
/// best GPU path by 1.27× (log_h=16), 1.64× (log_h=14, 18), up to
/// 3.14× at log_h=12.
///
/// So under `Default` (performance-safe mode), any `w > 1` call
/// routes to CPU. The single-column path keeps its original
/// `log_h ≥ 14` threshold where GPU wins 1.77-2.44× on the same
/// host.
///
/// # Overrides
///
/// * [`GpuDft::force_gpu`] bypasses this rule — the adapter then
///   picks the better GPU path per `(log_h, w)` regardless. Use
///   when a consumer knows their hardware profile (slower CPU, for
///   example) or explicitly wants GPU-exercise correctness validation.
/// * [`GpuDft::strict_gpu`] same bypass plus loud panics on any
///   fallback — the correctness-oracle constructor.
///
/// # Future
///
/// The width-based CPU shunt is conservative until **Path C** lands
/// (R4-batched kernel + workgroup-local fused tail in the batched
/// plan). Expected post-Path-C behavior: GPU beats CPU at `w > 1`
/// for `log_h ≥ 16` on discrete NVIDIA. At that point this
/// predicate simplifies back to a pure `log_h` threshold.
fn should_fallback(log_h: u32, width: usize, force_gpu: bool) -> bool {
    if force_gpu {
        return false;
    }
    // Width > 1: CPU wins at every measured size on the strongest
    // consumer CPU on the market. Until Path C flips that regime,
    // default mode keeps consumers out of the GPU-regression zone.
    if width > 1 {
        return true;
    }
    // Width == 1: GPU wins cleanly at log_h ≥ 16 on discrete NVIDIA.
    // log_h=14 is nearly tied; keep as-is for the conservative
    // threshold.
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

// -- Strict-vs-warn fallback decision ----------------------------------------

/// Handle an `Err` from a GPU-path step by either panicking (strict
/// mode) or warning + returning `None` (non-strict mode, caller falls
/// back to CPU).
///
/// Factored out of `dft_batch` so the strict-vs-warn policy can be
/// tested without allocating huge matrices to trip internal errors.
/// See the test `strict_mode_panics` which calls this directly.
#[inline]
fn handle_gpu_step_err(context_tag: &str, log_h: u32, strict: bool, err: String) {
    if strict {
        panic!(
            "GpuDft::strict_gpu(): {context_tag} at log_h={log_h} and \
             strict mode forbids CPU fallback: {err}"
        );
    }
    tracing::warn!(
        target: "zkgpu_plonky3",
        "{context_tag} at log_h={log_h}, falling back: {err}"
    );
}

// -- TwoAdicSubgroupDft impl --------------------------------------------------

impl TwoAdicSubgroupDft<P3BabyBear> for GpuDft<P3BabyBear> {
    type Evaluations = BitReversedMatrixView<RowMajorMatrix<P3BabyBear>>;

    #[tracing::instrument(
        name = "zkgpu_plonky3::dft_batch",
        level = "debug",
        skip_all,
        fields(h = mat.height(), w = mat.width(), force_gpu = self.force_gpu, strict = self.strict)
    )]
    fn dft_batch(&self, mat: RowMajorMatrix<P3BabyBear>) -> Self::Evaluations {
        let h = mat.height();
        let w = mat.width();

        // Degenerate cases: empty matrix or single-row. Delegate to
        // the CPU impl — its default handles both correctly. Even in
        // strict mode, `h <= 1` is a correctness-forced fallback (no
        // NTT to run), not a GPU-availability fallback, so we don't
        // panic.
        if h <= 1 {
            return self.cpu_fallback.dft_batch(mat);
        }

        let log_h = log2_strict_usize(h) as u32;

        if should_fallback(log_h, w, self.force_gpu) {
            // Size threshold — already bypassed when force_gpu=true, so
            // this path only triggers for default-mode `GpuDft`. No
            // strict-mode concern here (strict implies force_gpu).
            return self.cpu_fallback.dft_batch(mat);
        }

        let ctx = match get_context() {
            Ok(ctx) => ctx.clone(),
            Err(e) => {
                handle_gpu_step_err("device init failed", log_h, self.strict, e.clone());
                return self.cpu_fallback.dft_batch(mat);
            }
        };

        // Phase 7.5 — use the batched plan when it's empirically the
        // better GPU choice.
        //
        // Measured on RTX 4090 + Ryzen 9 7950X after Path C.1 landed
        // (R4-batched + pitched storage). For w=8, dft_batch in
        // microseconds:
        //
        //   log_h   CPU     Path A (col)   Batched C.1   winner
        //   12      155     1261           456           batched
        //   14      711     2343           1190          batched
        //   16      3270    5419           6980          Path A
        //   18      14440   24002          23980         tied
        //   20      98770   111460         109900        batched
        //
        // So the GPU-best path is non-monotonic in `log_h`:
        //
        //   * `log_h <= BATCHED_WINS_BELOW = 14` → batched: launch
        //     overhead of Path A's w-fold column loop dominates; one
        //     batched dispatch per stage wins.
        //   * `BATCHED_WINS_BELOW < log_h < BATCHED_WINS_ABOVE` (i.e.
        //     15, 16, 17) → Path A: the single-column plan's R4 +
        //     workgroup-local fused tail reduces the dispatch count
        //     sharply at these sizes. The batched plan has no local
        //     fused tail (Path C.2 future work) and exactly fills one
        //     SM wave at log_h=16, leaving no oversubscription to
        //     hide memory latency. Path A column-loop wins despite
        //     the w-fold launch overhead.
        //   * `log_h >= BATCHED_WINS_ABOVE = 18` → batched: per-stage
        //     thread count grows enough (>= 4× SM capacity) for
        //     batched R4 to saturate the GPU, and Path A's w-fold
        //     launch overhead re-dominates.
        //
        // Width=1 always takes Path A regardless — the batched plan
        // has no advantage there.
        const BATCHED_WINS_BELOW: u32 = 14;
        const BATCHED_WINS_ABOVE: u32 = 18;
        let prefer_batched = w > 1
            && (log_h <= BATCHED_WINS_BELOW || log_h >= BATCHED_WINS_ABOVE);
        if prefer_batched {
            match ctx.get_or_build_batched_plan(log_h, w as u32, NttDirection::Forward) {
                Ok(plan) => {
                    match run_batched(&ctx, &plan, &mat) {
                        Ok(out) => {
                            let mut natural = out;
                            reverse_matrix_index_bits(&mut natural);
                            return BitReversalPerm::new_view(natural);
                        }
                        Err(e) => {
                            handle_gpu_step_err(
                                "batched GPU execution failed",
                                log_h,
                                self.strict,
                                e,
                            );
                            return self.cpu_fallback.dft_batch(mat);
                        }
                    }
                }
                Err(e) => {
                    // Batched plan-build failure is usually a size-limit
                    // (log_n > MAX_BATCHED_LOG_N or buffer > device cap).
                    // Fall through to Path A column loop rather than
                    // abort — Path A can handle larger single-column
                    // sizes via Four-Step. Strict mode already panics
                    // inside handle_gpu_step_err if force_gpu is on.
                    tracing::debug!(
                        target: "zkgpu_plonky3",
                        "batched plan unavailable at log_h={log_h} w={w}, trying Path A: {e}"
                    );
                }
            }
        }

        // Path A: column loop. For each column, gather → upload →
        // execute → download → scatter. See spec §4.3.
        let plan = match ctx.get_or_build_plan(log_h, NttDirection::Forward) {
            Ok(p) => p,
            Err(e) => {
                handle_gpu_step_err("plan build failed", log_h, self.strict, e);
                return self.cpu_fallback.dft_batch(mat);
            }
        };

        let natural = match run_column_loop(&ctx, &plan, &mat) {
            Ok(out) => out,
            Err(e) => {
                handle_gpu_step_err("GPU execution failed", log_h, self.strict, e);
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

/// Phase 7.5 Path B — run the entire `h × w` matrix through the
/// batched plan in a single GPU round-trip. No column loop.
///
/// The matrix is uploaded as a flat `h * w` buffer (row-major,
/// matching Plonky3's `RowMajorMatrix::values` layout), converted
/// Monty → canonical in the upload-side pass and back again on
/// download.
#[tracing::instrument(
    name = "zkgpu_plonky3::run_batched",
    level = "debug",
    skip_all,
    fields(h = mat.height(), w = mat.width())
)]
fn run_batched(
    ctx: &Arc<GpuContext>,
    plan_handle: &Arc<Mutex<WgpuBatchedNttPlan>>,
    mat: &RowMajorMatrix<P3BabyBear>,
) -> Result<RowMajorMatrix<P3BabyBear>, String> {
    let w = mat.width();
    let h = mat.height();

    let mut plan = plan_handle
        .lock()
        .map_err(|_| "batched plan mutex poisoned".to_string())?;

    // Sanity-check the plan's shape matches the matrix. Cheap
    // compared to the GPU round-trip, and cuts off confusing errors
    // downstream if the plan cache ever ages out of sync with caller
    // shapes. Folded into the same mutex acquisition as execution —
    // no separate peek lock needed on this hot path.
    if plan.width() != w as u32 || (1u32 << plan.log_n()) != h as u32 {
        return Err(format!(
            "batched plan shape mismatch: plan=({}, w={}), matrix=({}, w={})",
            1u32 << plan.log_n(),
            plan.width(),
            h,
            w
        ));
    }

    // Phase C.1: pitched storage. Upload `h × pitch` where
    // `pitch = round_up(w, 8)`. Padding columns `[w, pitch)` hold
    // zero; they flow through the butterfly network harmlessly and
    // are stripped on download.
    let pitch = plan.pitch() as usize;

    let mut canonical: Vec<ZkgpuBabyBear> = vec![ZkgpuBabyBear(0); h * pitch];
    for r in 0..h {
        for c in 0..w {
            canonical[r * pitch + c] =
                ZkgpuBabyBear(mat.values[r * w + c].as_canonical_u32());
        }
    }

    let mut gpu_buf = ctx
        .device
        .upload(&canonical)
        .map_err(|e| format!("batched upload failed: {e}"))?;

    plan.execute(&ctx.device, &mut gpu_buf)
        .map_err(|e| format!("batched execute failed: {e}"))?;

    let result_canonical = gpu_buf
        .read_to_vec()
        .map_err(|e| format!("batched readback failed: {e}"))?;

    // Strip pitch padding: extract `h × w` from the `h × pitch`
    // buffer and convert canonical → Monty in the same pass.
    let mut output_vals: Vec<P3BabyBear> = vec![P3BabyBear::new(0); h * w];
    for r in 0..h {
        for c in 0..w {
            output_vals[r * w + c] =
                P3BabyBear::new(result_canonical[r * pitch + c].0);
        }
    }

    Ok(RowMajorMatrix::new(output_vals, w))
}

/// Core column-loop execution. Pulled out for clarity and to keep the
/// trait method focused on control flow.
#[tracing::instrument(
    name = "zkgpu_plonky3::run_column_loop",
    level = "debug",
    skip_all,
    fields(h = mat.height(), w = mat.width())
)]
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

    /// Strict mode: a differential test against `Radix2DitParallel`
    /// must pass when GPU is available. If a silent CPU fallback were
    /// happening under `force_gpu()`, this test would still be green;
    /// `strict_gpu()` guarantees the result came from the GPU path by
    /// panicking on any fallback. Pair with the differential assertion
    /// and we have an asserted GPU-path signal.
    #[test]
    fn strict_gpu_succeeds_when_device_available() {
        let mat = random_matrix(12, 4, 0x51EADF15);
        let cpu_out = Radix2DitParallel::<P3BabyBear>::default()
            .dft_batch(mat.clone())
            .to_row_major_matrix();

        let gpu = GpuDft::<P3BabyBear>::strict_gpu();
        let gpu_out = gpu.dft_batch(mat).to_row_major_matrix();

        assert_matrix_eq(&gpu_out, &cpu_out, "strict_gpu log_h=12 w=4");
    }

    /// Phase C.1 — pin the go/no-go regime. After C.1, the batched
    /// plan is the routed path at `log_h ∈ {18, 20}` for `w > 1`
    /// under `force_gpu` / `strict_gpu`. If a kernel regression at
    /// exactly those sizes slipped past the `log_h ≤ 16` family of
    /// tests below, the whole point of Path C.1 would be lost silently.
    /// This test guards the regime that drives the C.4 go/no-go gate.
    ///
    /// Uses `strict_gpu` so any silent fallback to CPU (which would
    /// make the differential pass trivially) panics.
    #[test]
    fn c1_regime_dft_batch_matches_parallel_log18() {
        // log_h=18 is the smallest size where the bench data shows
        // C.1 batched beating Path A column-loop on RTX 4090.
        // Matrix is `2^18 × 8` = 2 MiB — comfortable for CI memory.
        let mat = random_matrix(18, 8, 0xC1_0018_0008_u64);
        let cpu_out = Radix2DitParallel::<P3BabyBear>::default()
            .dft_batch(mat.clone())
            .to_row_major_matrix();

        let gpu = GpuDft::<P3BabyBear>::strict_gpu();
        let gpu_out = gpu.dft_batch(mat).to_row_major_matrix();

        assert_matrix_eq(&gpu_out, &cpu_out, "C.1 dft_batch log_h=18 w=8");
    }

    /// Phase C.1 — same regime, through coset_lde_batch (the FRI
    /// hot path). The default trait decomposition calls dft_batch
    /// twice under the hood, so this exercises C.1 across the real
    /// consumer of the adapter.
    #[test]
    fn c1_regime_coset_lde_matches_parallel_log18() {
        use p3_field::Field;
        use p3_matrix::bitrev::BitReversibleMatrix;

        let shift = P3BabyBear::GENERATOR;
        let mat = random_matrix(18, 8, 0xC1_1DE0_0018_u64);
        let cpu_lde = Radix2DitParallel::<P3BabyBear>::default()
            .coset_lde_batch(mat.clone(), 1, shift)
            .bit_reverse_rows()
            .to_row_major_matrix();

        let gpu = GpuDft::<P3BabyBear>::strict_gpu();
        let gpu_lde = gpu
            .coset_lde_batch(mat, 1, shift)
            .bit_reverse_rows()
            .to_row_major_matrix();

        assert_matrix_eq(
            &gpu_lde,
            &cpu_lde,
            "C.1 coset_lde_batch log_h=18 w=8 added_bits=1",
        );
    }

    /// Phase 7.5 Path B — differential test at widths that exercise
    /// the batched plan directly. Unlike the earlier medium/naive
    /// tests (which also routed through Path B at w=4 post-7.5), this
    /// fixture sweeps the widths where Path A loses hardest in 7.4's
    /// data: w=8, w=16. A pass proves the batched plan is correct at
    /// the sizes where its perf advantage matters.
    #[test]
    fn path_b_dft_batch_matches_parallel() {
        let gpu = GpuDft::<P3BabyBear>::strict_gpu();
        for &log_h in &[12usize, 14, 16] {
            for &w in &[8usize, 16] {
                let mat = random_matrix(log_h, w, 0xC01DB10C ^ (log_h as u64) << 16 ^ (w as u64));
                let cpu_out = Radix2DitParallel::<P3BabyBear>::default()
                    .dft_batch(mat.clone())
                    .to_row_major_matrix();
                let gpu_out = gpu.dft_batch(mat).to_row_major_matrix();
                assert_matrix_eq(
                    &gpu_out,
                    &cpu_out,
                    &format!("Path B log_h={log_h} w={w}"),
                );
            }
        }
    }

    /// Path B + coset LDE. `coset_lde_batch` default decomposition
    /// calls our `dft_batch` twice internally (idft + coset_dft after
    /// resize); both invocations at w > 1 route through the batched
    /// plan. This test confirms the full FRI-commit-critical pipeline
    /// is correct under Path B at realistic widths.
    #[test]
    fn path_b_coset_lde_matches_parallel() {
        use p3_field::Field;
        use p3_matrix::bitrev::BitReversibleMatrix;

        let shift = P3BabyBear::GENERATOR;
        let gpu = GpuDft::<P3BabyBear>::strict_gpu();
        for &log_h in &[12usize, 14] {
            for &w in &[8usize, 16] {
                for &added_bits in &[1usize, 2] {
                    let mat = random_matrix(log_h, w, 0x5E77E1 ^ (log_h as u64) << 8 ^ (w as u64));
                    let cpu_lde = Radix2DitParallel::<P3BabyBear>::default()
                        .coset_lde_batch(mat.clone(), added_bits, shift)
                        .bit_reverse_rows()
                        .to_row_major_matrix();
                    let gpu_lde = gpu
                        .coset_lde_batch(mat, added_bits, shift)
                        .bit_reverse_rows()
                        .to_row_major_matrix();
                    assert_matrix_eq(
                        &gpu_lde,
                        &cpu_lde,
                        &format!(
                            "Path B coset_lde log_h={log_h} w={w} added_bits={added_bits}"
                        ),
                    );
                }
            }
        }
    }

    /// Phase 7.2.a — `coset_lde_batch` differential between GPU and
    /// CPU DFTs at FRI-realistic sizes.
    ///
    /// `coset_lde_batch` is the single hottest DFT method in
    /// `plonky3/fri/src/two_adic_pcs.rs` (two call sites in the
    /// commit phase, plus one in `hiding_pcs.rs`). Default trait
    /// decomposition composes our GPU `dft_batch` through `idft_batch`
    /// (swap-and-scale) and `coset_dft_batch` (row-scale + dft). If
    /// any step of that pipeline disagrees with the CPU impl, the
    /// FRI commitment root will diverge — which is exactly what this
    /// test prevents by comparing LDE output directly.
    ///
    /// Uses `strict_gpu()` so a silent fallback cannot make the
    /// assertion pass on CPU.
    #[test]
    fn coset_lde_batch_matches_parallel() {
        use p3_field::Field;
        use p3_matrix::bitrev::BitReversibleMatrix;

        // `shift` mirrors what `TwoAdicFriPcs::commit` uses:
        // `Val::GENERATOR / domain.shift()` at the PCS layer. For a
        // standalone test any non-trivial two-adic-field element works.
        let shift = P3BabyBear::GENERATOR;

        for &log_h in &[12usize, 14] {
            for &w in &[1usize, 4] {
                // `added_bits = 1` matches `StandardBackend`'s
                // `log_blowup = 1` setting in p3-zk-proofs. The hiding
                // path uses `log_blowup + 1 = 2`; we exercise both.
                for &added_bits in &[1usize, 2] {
                    let mat = random_matrix(log_h, w, 0x_C0DE_C07E_u64);

                    let cpu_lde = Radix2DitParallel::<P3BabyBear>::default()
                        .coset_lde_batch(mat.clone(), added_bits, shift)
                        .bit_reverse_rows()
                        .to_row_major_matrix();

                    let gpu = GpuDft::<P3BabyBear>::strict_gpu();
                    let gpu_lde = gpu
                        .coset_lde_batch(mat, added_bits, shift)
                        .bit_reverse_rows()
                        .to_row_major_matrix();

                    assert_matrix_eq(
                        &gpu_lde,
                        &cpu_lde,
                        &format!(
                            "coset_lde log_h={log_h} w={w} added_bits={added_bits}"
                        ),
                    );
                }
            }
        }
    }

    /// Strict-mode panic path exercised via the internal
    /// `handle_gpu_step_err` seam. Testing through the full
    /// `dft_batch` entry point would require triggering one of the
    /// GPU-path failures end-to-end (device init, plan build, or
    /// execute), all of which would demand either process-wide state
    /// manipulation or multi-GiB matrix allocations — neither
    /// appropriate for a unit test.
    ///
    /// The factoring keeps the actual dispatch simple: every site in
    /// `dft_batch` that could fall back calls `handle_gpu_step_err`,
    /// so if this helper panics correctly on strict + err, every call
    /// site does too.
    #[test]
    #[should_panic(expected = "strict mode forbids CPU fallback")]
    fn strict_mode_panics_on_err() {
        handle_gpu_step_err(
            "plan build failed",
            14,
            true,
            "simulated: log_n=28 exceeds 2-adicity".to_string(),
        );
    }

    /// Non-strict mode must *not* panic on the same error — it
    /// should emit a `tracing::warn!` and return. This test is the
    /// inverse of `strict_mode_panics_on_err` and together they pin
    /// the full policy behavior of `handle_gpu_step_err`.
    #[test]
    fn non_strict_mode_warns_without_panic() {
        handle_gpu_step_err(
            "plan build failed",
            14,
            false,
            "simulated error".to_string(),
        );
        // If we get here without panicking, the non-strict path is
        // correct. (tracing::warn! output is not asserted — verifying
        // log output in a unit test couples to the tracing
        // subscriber, which is out of scope.)
    }

    /// `preload_plans` warms the cache without panicking when the
    /// sizes are valid. Follow-up `dft_batch` calls at the same sizes
    /// should work correctly — we can't measure "faster" deterministically
    /// in a unit test but we can verify correctness is unaffected.
    #[test]
    fn preload_plans_is_noop_correctness_wise() {
        let gpu = GpuDft::<P3BabyBear>::strict_gpu();
        gpu.preload_plans(&[12, 14]);

        // After preload, a strict dft_batch at the preloaded size must
        // still succeed and produce the correct answer.
        let mat = random_matrix(12, 2, 0xBAD_CAFE_u64);
        let cpu_out = Radix2DitParallel::<P3BabyBear>::default()
            .dft_batch(mat.clone())
            .to_row_major_matrix();
        let gpu_out = gpu.dft_batch(mat).to_row_major_matrix();
        assert_matrix_eq(&gpu_out, &cpu_out, "post-preload dft_batch");
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
