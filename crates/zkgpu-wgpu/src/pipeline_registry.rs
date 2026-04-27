use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{Arc, Mutex};

/// Device-scoped registry for reusing compiled compute pipelines.
///
/// All WGSL shader sources are `include_str!`'d into the binary, so
/// pointer equality on the source `&'static str` is a valid identity key.
/// Forward + inverse NTT plans share the same shaders and pipeline layouts
/// (they differ only in twiddle values and scale params), so reusing
/// pipelines across plans eliminates redundant shader compilation.
///
/// On desktop drivers this saves ~50ms per duplicate pipeline. On mobile
/// Vulkan without a driver cache, it saves ~200ms per pipeline.
///
/// # Cache-key shape
///
/// As of the foundation commit for the speed-opportunities phase
/// ([`docs/research/zkgpu-wgpu-speed-opportunities.md`]), the cache
/// keys distinguish more than just `{source, entry_point, bgl_label}`.
/// The expanded shape allows upcoming items in that doc — `PipelineCompilationOptions`
/// zero-init opt-in (item #2), `Features::IMMEDIATES` for small per-dispatch
/// params (item #3), `Device::create_shader_module_trusted` for audited
/// kernels (item #7) — to specialize pipelines without silently colliding
/// in the cache.
///
/// Today every existing caller uses defaulted values for the new fields,
/// which preserves the previous cache identity exactly. New low-level
/// methods that take an explicit [`PipelineSpec`] will be added by the
/// items above as they land.
pub(crate) struct PipelineRegistry {
    modules: Mutex<HashMap<ShaderKey, Arc<wgpu::ShaderModule>>>,
    pipelines: Mutex<HashMap<PipelineKey, Arc<wgpu::ComputePipeline>>>,
    bgls: Mutex<HashMap<BglKey, Arc<wgpu::BindGroupLayout>>>,
}

// ---------------------------------------------------------------------------
// Cache-key types
// ---------------------------------------------------------------------------

/// Identity of a WGSL source.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Debug)]
enum SourceId {
    /// Static `include_str!`'d source identified by pointer identity.
    /// Two `&'static str` values with the same content but different
    /// pointers (e.g. duplicate `include_str!` of the same file from
    /// two crates) deliberately produce separate cache entries — they
    /// may have been wrapped with different post-processing.
    Static(usize),
    /// Generated / runtime-built WGSL identified by content hash.
    /// Reserved for item #6 (Poseidon constants migration via
    /// generated module-constant shaders); not yet used.
    #[allow(dead_code)]
    Generated(u64),
}

/// How the shader module was created — controls runtime safety checks.
///
/// `pub(crate)` because [`PipelineSpec::runtime_check_mode`] exposes
/// this type to in-crate callers (items #2-7 of the speed-opportunities
/// doc); not exposed outside the crate.
#[derive(Hash, Eq, PartialEq, Clone, Copy, Default, Debug)]
pub(crate) enum RuntimeCheckMode {
    /// `Device::create_shader_module` — full validation. Today's path
    /// for every kernel.
    #[default]
    Safe,
    /// `Device::create_shader_module_trusted` with selective check
    /// elision. The `u32` is a fingerprint of the
    /// `wgpu::ShaderRuntimeChecks` set, computed by the caller for
    /// cache identity. Reserved for item #7; not yet used.
    #[allow(dead_code)]
    Trusted(u32),
    /// `Device::create_shader_module_passthrough` — bypasses wgpu
    /// validation entirely. Reserved for item #12 (SPIR-V / Metal
    /// passthrough); not yet used.
    #[allow(dead_code)]
    Passthrough,
}

/// Key for shader modules. Distinguishes modules built via the safe
/// `create_shader_module` path from `_trusted` / `_passthrough`
/// variants when the source is identical.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct ShaderKey {
    source: SourceId,
    runtime_check_mode: RuntimeCheckMode,
}

/// Key for bind group layouts.
///
/// Convention: BGL labels in zkgpu-wgpu are unique per structural
/// shape. Two BGLs that share a label but differ in entries would
/// collide here; the convention is enforced socially via code review,
/// not yet by a structural hash. See speed-opportunities doc item #2
/// precondition for the future-hardening note (BGL structural
/// fingerprint as a follow-on if/when call sites multiply).
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct BglKey {
    label: &'static str,
}

/// Pipeline-layout-side identity. The wgpu `PipelineLayoutDescriptor`
/// owns the BGL list AND `immediate_size`; both must be reflected in
/// the pipeline cache key.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct LayoutKey {
    bgl_label: &'static str,
    /// `PipelineLayoutDescriptor::immediate_size`. Zero means no
    /// immediates. Item #3 (immediates) will populate this for the
    /// kernels it touches.
    immediate_size: u32,
}

/// Snapshot of `wgpu::PipelineCompilationOptions` fields that affect
/// cache identity. Defaults match wgpu's defaults so existing callers
/// produce the same cache key shape as before this commit.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct CompilationOptionsKey {
    /// `PipelineCompilationOptions::zero_initialize_workgroup_memory`.
    /// `true` is wgpu's default. Item #2 introduces opt-in to `false`
    /// for the local Stockham + transpose tile kernels.
    zero_initialize_workgroup_memory: bool,
    // override_constants: Vec<(String, OverrideValue)> deliberately
    // absent. Item #9 (planner-selected workgroup size) is the first
    // user; deferred. Add a field here when that lands.
}

impl Default for CompilationOptionsKey {
    fn default() -> Self {
        Self {
            // wgpu's default is `true` (zero workgroup memory at
            // dispatch start). Match that here so any caller that
            // hasn't opted in produces a key indistinguishable from
            // pre-foundation-commit.
            zero_initialize_workgroup_memory: true,
        }
    }
}

/// Compute pipeline cache key.
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct PipelineKey {
    shader: ShaderKey,
    entry_point: &'static str,
    layout: LayoutKey,
    compilation_options: CompilationOptionsKey,
    /// Capability/feature bits that affect codegen for this pipeline.
    /// Reserved for items that change the compiled output based on
    /// device features (e.g. `Features::IMMEDIATES`, `Features::SUBGROUP`).
    /// Today all callers use `0`, which preserves pre-commit identity.
    capability_bits: u64,
}

// ---------------------------------------------------------------------------
// PipelineSpec — for low-level callers (items #2-7 will use this)
// ---------------------------------------------------------------------------

/// Specialization parameters for a compute pipeline.
///
/// Existing callers (the 14+ in-tree pipeline-creation sites) use
/// [`PipelineRegistry::get_or_create_pipeline`], which builds a
/// pipeline with all-defaults — equivalent to passing
/// `PipelineSpec::default()`.
///
/// Items #2 (zero-init), #3 (immediates), #7 (trusted modules) from
/// the speed-opportunities doc are the first callers that will need
/// non-default values. They will add a `_with_spec` variant of the
/// pipeline-creation methods at that time.
// ---------------------------------------------------------------------------
// Design note for item #3 (`Features::IMMEDIATES`) — drift-prevention rule
// ---------------------------------------------------------------------------
//
// When item #3 lands, both `PipelineSpec::immediate_size` (used in the cache
// key below) and `wgpu::PipelineLayoutDescriptor::immediate_size` (used to
// build the actual GPU resource) must be derived from a **single source
// value**, not set independently from two places. Otherwise we get the
// same drift class the foundation commit's reviewer warned about for
// `compilation_options`: the cache could serve a pipeline whose layout was
// compiled with a different `immediate_size` than the spec's key claims.
//
// Recommended shape when item #3 is implemented:
//
//     // Owned by the plan-build code, single source of truth.
//     struct LayoutSpec {
//         immediate_size: u32,
//         param_mode: ParamMode,  // Uniform | Immediate
//     }
//
//     // Construct PipelineLayoutDescriptor.immediate_size and
//     // PipelineSpec.immediate_size both from the same `LayoutSpec` —
//     // never inline a literal `0` or `16` at the call site.
//
// The cache key field below intentionally stays in `PipelineSpec` because
// the registry needs to distinguish keyed pipelines, but the construction
// must thread through one helper. See the speed-opportunities doc item
// #3 implementation surface.
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub(crate) struct PipelineSpec {
    pub(crate) zero_initialize_workgroup_memory: bool,
    /// Reserved for item #3 (immediates). Read by the cache key today;
    /// no caller sets a non-default value yet. **When item #3 lands,
    /// see the drift-prevention design note above this struct.**
    #[allow(dead_code)]
    pub(crate) immediate_size: u32,
    /// Reserved for item #7 (trusted modules). Read by the cache key today;
    /// no caller sets a non-default value yet.
    #[allow(dead_code)]
    pub(crate) runtime_check_mode: RuntimeCheckMode,
    /// Reserved for item #3 (`Features::IMMEDIATES` capability bit). Read
    /// by the cache key today; no caller sets a non-default value yet.
    #[allow(dead_code)]
    pub(crate) capability_bits: u64,
}

impl Default for PipelineSpec {
    fn default() -> Self {
        Self {
            zero_initialize_workgroup_memory: true,
            immediate_size: 0,
            runtime_check_mode: RuntimeCheckMode::Safe,
            capability_bits: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reserved utility — content-hash a slice of bytes using the std
/// default hasher. Used by [`SourceId::Generated`] when item #6
/// (generated WGSL) lands. Kept here so the hashing approach is
/// consistent across cache-key fields.
#[allow(dead_code)]
fn fingerprint_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// PipelineRegistry impl
// ---------------------------------------------------------------------------

impl PipelineRegistry {
    pub(crate) fn new() -> Self {
        Self {
            modules: Mutex::new(HashMap::new()),
            pipelines: Mutex::new(HashMap::new()),
            bgls: Mutex::new(HashMap::new()),
        }
    }

    /// Get or create a shader module from an `include_str!` source.
    ///
    /// Always uses [`RuntimeCheckMode::Safe`] (the wgpu default).
    /// Future trusted/passthrough variants will live behind separate
    /// methods (item #7).
    pub(crate) fn get_or_create_module(
        &self,
        device: &wgpu::Device,
        source: &'static str,
        label: &'static str,
    ) -> Arc<wgpu::ShaderModule> {
        let key = ShaderKey {
            source: SourceId::Static(source.as_ptr() as usize),
            runtime_check_mode: RuntimeCheckMode::Safe,
        };
        let mut modules = self.modules.lock().expect("pipeline registry lock");
        modules
            .entry(key)
            .or_insert_with(|| {
                Arc::new(device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(label),
                    source: wgpu::ShaderSource::Wgsl(source.into()),
                }))
            })
            .clone()
    }

    /// Get or create a bind group layout.
    pub(crate) fn get_or_create_bgl(
        &self,
        device: &wgpu::Device,
        label: &'static str,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> Arc<wgpu::BindGroupLayout> {
        let key = BglKey { label };
        let mut bgls = self.bgls.lock().expect("pipeline registry lock");
        bgls.entry(key)
            .or_insert_with(|| {
                Arc::new(
                    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(label),
                        entries,
                    }),
                )
            })
            .clone()
    }

    /// Get or create a compute pipeline (default spec).
    ///
    /// Convenience wrapper around [`Self::get_or_create_pipeline_with_spec`]
    /// with [`PipelineSpec::default()`]. Behavior unchanged for all 14+
    /// existing callers — the resulting cache key and compilation
    /// options are both derived from the defaulted spec, matching
    /// pre-foundation-commit identity.
    ///
    /// When a `cache` is provided (Vulkan only), the driver can skip
    /// shader compilation on subsequent runs by reusing serialised
    /// microcode.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_create_pipeline(
        &self,
        device: &wgpu::Device,
        source: &'static str,
        entry_point: &'static str,
        bgl_label: &'static str,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        cache: Option<&wgpu::PipelineCache>,
    ) -> Arc<wgpu::ComputePipeline> {
        self.get_or_create_pipeline_with_spec(
            device,
            source,
            entry_point,
            bgl_label,
            layout,
            module,
            cache,
            &PipelineSpec::default(),
        )
    }

    /// Get or create a compute pipeline with an explicit specialization.
    ///
    /// Both the cache key AND the `wgpu::ComputePipelineDescriptor`'s
    /// `compilation_options` are derived from the same `spec`, so the
    /// two cannot drift apart. Two callers passing differently-valued
    /// specs land in distinct cache slots and produce distinctly-
    /// compiled pipelines; two callers passing the same spec collapse
    /// to one slot.
    ///
    /// Used by the speed-opportunities phase (see
    /// `docs/research/zkgpu-wgpu-speed-opportunities.md`):
    ///
    /// * **Item #2** (zero-init opt-in on local Stockham + transpose
    ///   tile kernels) — passes `zero_initialize_workgroup_memory =
    ///   false` for kernels that fully initialize their workgroup
    ///   memory before any read.
    /// * Future items #3, #7 will populate `immediate_size` and
    ///   `runtime_check_mode` respectively.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn get_or_create_pipeline_with_spec(
        &self,
        device: &wgpu::Device,
        source: &'static str,
        entry_point: &'static str,
        bgl_label: &'static str,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        cache: Option<&wgpu::PipelineCache>,
        spec: &PipelineSpec,
    ) -> Arc<wgpu::ComputePipeline> {
        let key = build_pipeline_key(source, entry_point, bgl_label, spec);
        // Derive compilation_options from the same spec the cache key
        // sees. This is the drift-proof property: a pipeline served
        // from the cache for `key` was compiled with options that
        // match `spec`, because the only path to a pipeline at this
        // key is this method.
        let compilation_options = wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: spec.zero_initialize_workgroup_memory,
            ..Default::default()
        };
        let mut pipelines = self.pipelines.lock().expect("pipeline registry lock");
        pipelines
            .entry(key)
            .or_insert_with(|| {
                Arc::new(
                    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(entry_point),
                        layout: Some(layout),
                        module,
                        entry_point: Some(entry_point),
                        compilation_options,
                        cache,
                    }),
                )
            })
            .clone()
    }

    /// Number of cached pipelines (for testing / diagnostics).
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn pipeline_count(&self) -> usize {
        self.pipelines.lock().expect("lock").len()
    }
}

/// Build a [`PipelineKey`] from the source identity + spec. Pure
/// function so unit tests can verify key distinctness without
/// touching a GPU.
fn build_pipeline_key(
    source: &'static str,
    entry_point: &'static str,
    bgl_label: &'static str,
    spec: &PipelineSpec,
) -> PipelineKey {
    PipelineKey {
        shader: ShaderKey {
            source: SourceId::Static(source.as_ptr() as usize),
            runtime_check_mode: spec.runtime_check_mode,
        },
        entry_point,
        layout: LayoutKey {
            bgl_label,
            immediate_size: spec.immediate_size,
        },
        compilation_options: CompilationOptionsKey {
            zero_initialize_workgroup_memory: spec.zero_initialize_workgroup_memory,
        },
        capability_bits: spec.capability_bits,
    }
}

// ---------------------------------------------------------------------------
// Tests — pure key-shape correctness, no GPU required
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    static SRC_A: &str = "// shader A\n";
    static SRC_B: &str = "// shader B\n";

    #[test]
    fn default_spec_matches_pre_foundation_identity() {
        // The defaulted spec should produce the same key for two
        // calls with the same source/entry/bgl. This is the
        // backwards-compat invariant: every existing caller uses the
        // default spec.
        let k1 = build_pipeline_key(SRC_A, "main", "bgl_a", &PipelineSpec::default());
        let k2 = build_pipeline_key(SRC_A, "main", "bgl_a", &PipelineSpec::default());
        assert_eq!(k1, k2);
    }

    #[test]
    fn different_source_produces_different_key() {
        let k1 = build_pipeline_key(SRC_A, "main", "bgl_a", &PipelineSpec::default());
        let k2 = build_pipeline_key(SRC_B, "main", "bgl_a", &PipelineSpec::default());
        assert_ne!(k1, k2);
    }

    #[test]
    fn different_entry_point_produces_different_key() {
        let k1 = build_pipeline_key(SRC_A, "main", "bgl_a", &PipelineSpec::default());
        let k2 = build_pipeline_key(SRC_A, "alt", "bgl_a", &PipelineSpec::default());
        assert_ne!(k1, k2);
    }

    #[test]
    fn different_bgl_label_produces_different_key() {
        let k1 = build_pipeline_key(SRC_A, "main", "bgl_a", &PipelineSpec::default());
        let k2 = build_pipeline_key(SRC_A, "main", "bgl_b", &PipelineSpec::default());
        assert_ne!(k1, k2);
    }

    /// Item #2 precondition: zero-init opt-in must produce a distinct
    /// cache slot from the default. Otherwise zero_initialize_workgroup_memory
    /// = false would silently share a pipeline with the default
    /// (zero_init = true) version, served from cache for whichever
    /// landed first.
    #[test]
    fn zero_init_opt_in_produces_different_key() {
        let default_spec = PipelineSpec::default();
        let opt_in_spec = PipelineSpec {
            zero_initialize_workgroup_memory: false,
            ..PipelineSpec::default()
        };
        let k_default = build_pipeline_key(SRC_A, "main", "bgl_a", &default_spec);
        let k_opt_in = build_pipeline_key(SRC_A, "main", "bgl_a", &opt_in_spec);
        assert_ne!(
            k_default, k_opt_in,
            "zero-init opt-in must produce a distinct cache slot"
        );
    }

    /// Item #3 precondition: a non-zero `immediate_size` must produce
    /// a distinct cache slot. Two pipelines that share a BGL but differ
    /// in immediates have different `wgpu::PipelineLayout` identity.
    #[test]
    fn immediate_size_produces_different_key() {
        let default_spec = PipelineSpec::default();
        let immediate_spec = PipelineSpec {
            immediate_size: 16,
            ..PipelineSpec::default()
        };
        let k_default = build_pipeline_key(SRC_A, "main", "bgl_a", &default_spec);
        let k_immediate = build_pipeline_key(SRC_A, "main", "bgl_a", &immediate_spec);
        assert_ne!(
            k_default, k_immediate,
            "non-zero immediate_size must produce a distinct cache slot"
        );
    }

    /// Item #7 precondition: trusted shader modules must produce a
    /// distinct cache slot from safe modules of the same source.
    #[test]
    fn trusted_module_produces_different_key() {
        let safe_spec = PipelineSpec::default();
        let trusted_spec = PipelineSpec {
            runtime_check_mode: RuntimeCheckMode::Trusted(0xDEAD_BEEF),
            ..PipelineSpec::default()
        };
        let k_safe = build_pipeline_key(SRC_A, "main", "bgl_a", &safe_spec);
        let k_trusted = build_pipeline_key(SRC_A, "main", "bgl_a", &trusted_spec);
        assert_ne!(
            k_safe, k_trusted,
            "trusted module must produce a distinct cache slot"
        );
    }

    /// Two trusted modules with different `ShaderRuntimeChecks`
    /// fingerprints must also produce distinct cache slots.
    #[test]
    fn trusted_module_with_different_checks_produces_different_key() {
        let trusted_a = PipelineSpec {
            runtime_check_mode: RuntimeCheckMode::Trusted(0xAAAA_AAAA),
            ..PipelineSpec::default()
        };
        let trusted_b = PipelineSpec {
            runtime_check_mode: RuntimeCheckMode::Trusted(0xBBBB_BBBB),
            ..PipelineSpec::default()
        };
        let k_a = build_pipeline_key(SRC_A, "main", "bgl_a", &trusted_a);
        let k_b = build_pipeline_key(SRC_A, "main", "bgl_a", &trusted_b);
        assert_ne!(k_a, k_b);
    }

    /// Capability bits must affect the key — two pipelines compiled
    /// for different feature sets (e.g. with `Features::IMMEDIATES`
    /// available vs not) must not share a slot.
    #[test]
    fn capability_bits_produces_different_key() {
        let bits_a = PipelineSpec {
            capability_bits: 0b001,
            ..PipelineSpec::default()
        };
        let bits_b = PipelineSpec {
            capability_bits: 0b010,
            ..PipelineSpec::default()
        };
        let k_a = build_pipeline_key(SRC_A, "main", "bgl_a", &bits_a);
        let k_b = build_pipeline_key(SRC_A, "main", "bgl_a", &bits_b);
        assert_ne!(k_a, k_b);
    }

    #[test]
    fn spec_default_values_match_wgpu_defaults() {
        // The defaulted PipelineSpec must agree with the corresponding
        // fields of `wgpu::PipelineCompilationOptions::default()`. If
        // wgpu's defaults shift in a future version, this test alerts
        // us to update the spec default — without an actual wgpu
        // comparison, the test would just be re-asserting our literal
        // values, which is useless for drift detection (P3 review of
        // the foundation commit).
        let spec = PipelineSpec::default();
        let wgpu_default = wgpu::PipelineCompilationOptions::default();
        assert_eq!(
            spec.zero_initialize_workgroup_memory,
            wgpu_default.zero_initialize_workgroup_memory,
            "PipelineSpec::default().zero_initialize_workgroup_memory \
             must match wgpu::PipelineCompilationOptions::default()"
        );

        // Remaining fields are zkgpu-side conventions, not wgpu-side
        // defaults — keep them as literal checks.
        assert_eq!(spec.immediate_size, 0);
        assert_eq!(spec.runtime_check_mode, RuntimeCheckMode::Safe);
        assert_eq!(spec.capability_bits, 0);
    }

    #[test]
    fn fingerprint_bytes_distinguishes_content() {
        // Reserved-for-future helper; verify it actually distinguishes.
        let h1 = fingerprint_bytes(b"hello");
        let h2 = fingerprint_bytes(b"world");
        assert_ne!(h1, h2);
    }
}
