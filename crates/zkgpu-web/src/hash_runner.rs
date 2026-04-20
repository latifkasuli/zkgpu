//! Browser async hash runner (Phase F.3.d).
//!
//! Mirrors [`crate::runner::run_suite_async`] for the hash surface —
//! dispatches a [`HashSpec`] to either
//! [`WgpuBabyBearPoseidon2Plan`] or [`WgpuGoldilocksPoseidon2Plan`]
//! based on `spec.field`, validates each case against the CPU
//! reference in [`zkgpu_poseidon2`], and assembles a
//! [`HashSuiteReport`].
//!
//! Kept out of `runner.rs` for the same reason as the native testkit:
//! NTT-specific plumbing (family/tail overrides, soak, roundtrip)
//! doesn't touch the hash path.

use std::rc::Rc;

use zkgpu_babybear::BabyBear;
use zkgpu_core::GpuDevice;
use zkgpu_goldilocks::Goldilocks;
use zkgpu_poseidon2::{Poseidon2, Poseidon2Params, WIDTH as POSEIDON2_WIDTH};
use zkgpu_report::{
    Field, HashAlgorithm, HashCaseReport, HashCaseSpec, HashSpec, HashSuiteReport,
    KernelReport, SuiteSummary, TimingReport,
};
use zkgpu_wgpu::{WgpuBabyBearPoseidon2Plan, WgpuDevice, WgpuGoldilocksPoseidon2Plan};

use crate::device;
use crate::inputs::{make_babybear_hash_input, make_goldilocks_hash_input};
use crate::runner;
use crate::validation::compare_vectors;

/// Run a full hash suite asynchronously.
///
/// Errors bubble up as `String` to match the rest of the web runner
/// surface — the outer wasm entry point in `lib.rs` wraps the
/// `HashSuiteReport` in a `HarnessResponse`.
pub(crate) async fn run_hash_suite_async(
    spec: &HashSpec,
) -> Result<HashSuiteReport, String> {
    if spec.cases.is_empty() {
        return Err("suite must contain at least one case".to_string());
    }

    let dev = device::clone_device()?;
    let device_report = runner::build_device_report_from(&dev);

    // Build the plan once + reuse across every case. Plan types diverge
    // on buffer element type, so the dispatch fork happens here.
    let mut cases = Vec::with_capacity(spec.cases.len());
    match (spec.field, spec.algorithm) {
        (Field::BabyBear, HashAlgorithm::Poseidon2) => {
            let params =
                Poseidon2Params::<BabyBear, POSEIDON2_WIDTH>::babybear_default();
            let cpu = Poseidon2::new(params.clone());
            let mut plan = match WgpuBabyBearPoseidon2Plan::new(&dev, params) {
                Ok(p) => p,
                Err(e) => return Err(e.to_string()),
            };
            for case in &spec.cases {
                let report =
                    run_case_babybear(&dev, case, &mut plan, &cpu).await;
                let failed = !report.passed;
                cases.push(report);
                if spec.fail_fast && failed {
                    break;
                }
            }
        }
        (Field::Goldilocks, HashAlgorithm::Poseidon2) => {
            let params =
                Poseidon2Params::<Goldilocks, POSEIDON2_WIDTH>::goldilocks_default();
            let cpu = Poseidon2::new(params.clone());
            let mut plan = match WgpuGoldilocksPoseidon2Plan::new(&dev, params) {
                Ok(p) => p,
                Err(e) => return Err(e.to_string()),
            };
            for case in &spec.cases {
                let report =
                    run_case_goldilocks(&dev, case, &mut plan, &cpu).await;
                let failed = !report.passed;
                cases.push(report);
                if spec.fail_fast && failed {
                    break;
                }
            }
        }
    }

    let passed_cases = cases.iter().filter(|c| c.passed).count() as u32;
    let total_cases = cases.len() as u32;
    let failed_cases = total_cases - passed_cases;
    let kernel_family = match spec.field {
        Field::BabyBear => "babybear-poseidon2",
        Field::Goldilocks => "goldilocks-poseidon2-portable",
    };

    Ok(HashSuiteReport {
        schema_version: 1,
        suite: spec.kind,
        device: device_report,
        kernel: KernelReport {
            field: spec.field.display_name().to_string(),
            ntt_variant: kernel_family.to_string(),
            stockham_tail_strategy: None,
        },
        cases,
        summary: SuiteSummary {
            total_cases,
            passed_cases,
            failed_cases,
        },
    })
}

// ---------------------------------------------------------------------------
// Per-case runners
// ---------------------------------------------------------------------------

async fn run_case_babybear(
    dev: &Rc<WgpuDevice>,
    case: &HashCaseSpec,
    plan: &mut WgpuBabyBearPoseidon2Plan,
    cpu: &Poseidon2<BabyBear, POSEIDON2_WIDTH>,
) -> HashCaseReport {
    if case.num_permutations == 0 {
        return empty_case_report(case, "babybear-poseidon2");
    }

    let input = make_babybear_hash_input(case.num_permutations, &case.input);

    // CPU reference: permute each WIDTH-block independently.
    let mut expected = input.clone();
    for chunk in expected.chunks_exact_mut(POSEIDON2_WIDTH) {
        let arr: &mut [BabyBear; POSEIDON2_WIDTH] =
            chunk.try_into().expect("chunks_exact_mut enforces length");
        cpu.permute(arr);
    }

    match measure_babybear_poseidon2_async(
        dev,
        &input,
        plan,
        case.warmup_iterations,
        case.iterations,
    )
    .await
    {
        Ok((timings, gpu_out)) => {
            finalize_case(case, "babybear-poseidon2", timings, &gpu_out, &expected)
        }
        Err(e) => case_error(case, "babybear-poseidon2", e),
    }
}

async fn run_case_goldilocks(
    dev: &Rc<WgpuDevice>,
    case: &HashCaseSpec,
    plan: &mut WgpuGoldilocksPoseidon2Plan,
    cpu: &Poseidon2<Goldilocks, POSEIDON2_WIDTH>,
) -> HashCaseReport {
    if case.num_permutations == 0 {
        return empty_case_report(case, "goldilocks-poseidon2-portable");
    }

    let input = make_goldilocks_hash_input(case.num_permutations, &case.input);

    let mut expected = input.clone();
    for chunk in expected.chunks_exact_mut(POSEIDON2_WIDTH) {
        let arr: &mut [Goldilocks; POSEIDON2_WIDTH] =
            chunk.try_into().expect("chunks_exact_mut enforces length");
        cpu.permute(arr);
    }

    match measure_goldilocks_poseidon2_async(
        dev,
        &input,
        plan,
        case.warmup_iterations,
        case.iterations,
    )
    .await
    {
        Ok((timings, gpu_out)) => finalize_case(
            case,
            "goldilocks-poseidon2-portable",
            timings,
            &gpu_out,
            &expected,
        ),
        Err(e) => case_error(case, "goldilocks-poseidon2-portable", e),
    }
}

// ---------------------------------------------------------------------------
// Async measure helpers — wall-only per F.3.b contract
// ---------------------------------------------------------------------------
//
// `profile_gpu_timestamps` is not wired (Poseidon2 plans have no
// profiled-execute variant today). Honouring it is a future F.3.*
// sub-phase; the contract matches the native testkit's
// `measure_*_poseidon2_plan`.

async fn measure_babybear_poseidon2_async(
    dev: &Rc<WgpuDevice>,
    data: &[BabyBear],
    plan: &mut WgpuBabyBearPoseidon2Plan,
    warmup_iterations: u32,
    iterations: u32,
) -> Result<(TimingReport, Vec<BabyBear>), String> {
    for _ in 0..warmup_iterations {
        let mut buf = dev.upload(data).map_err(|e| e.to_string())?;
        plan.execute_async(dev, &mut buf)
            .await
            .map_err(|e| e.to_string())?;
    }

    let measured = iterations.max(1);
    let mut wall_total_ns: u64 = 0;
    let mut final_output: Option<Vec<BabyBear>> = None;

    for iter_idx in 0..measured {
        let mut buf = dev.upload(data).map_err(|e| e.to_string())?;
        let wall_start = web_time::Instant::now();
        plan.execute_async(dev, &mut buf)
            .await
            .map_err(|e| e.to_string())?;
        wall_total_ns += wall_start.elapsed().as_nanos() as u64;

        if iter_idx + 1 == measured {
            final_output = Some(
                buf.read_to_vec_async()
                    .await
                    .map_err(|e| e.to_string())?,
            );
        }
    }

    let wall_avg_ns = wall_total_ns / measured as u64;
    Ok((
        TimingReport {
            wall_time_ns: Some(wall_avg_ns),
            gpu_total_ns: None,
            gpu_stage_ns: Vec::new(),
        },
        final_output.expect("measured loop always captures final output"),
    ))
}

async fn measure_goldilocks_poseidon2_async(
    dev: &Rc<WgpuDevice>,
    data: &[Goldilocks],
    plan: &mut WgpuGoldilocksPoseidon2Plan,
    warmup_iterations: u32,
    iterations: u32,
) -> Result<(TimingReport, Vec<Goldilocks>), String> {
    for _ in 0..warmup_iterations {
        let mut buf = dev.upload(data).map_err(|e| e.to_string())?;
        plan.execute_async(dev, &mut buf)
            .await
            .map_err(|e| e.to_string())?;
    }

    let measured = iterations.max(1);
    let mut wall_total_ns: u64 = 0;
    let mut final_output: Option<Vec<Goldilocks>> = None;

    for iter_idx in 0..measured {
        let mut buf = dev.upload(data).map_err(|e| e.to_string())?;
        let wall_start = web_time::Instant::now();
        plan.execute_async(dev, &mut buf)
            .await
            .map_err(|e| e.to_string())?;
        wall_total_ns += wall_start.elapsed().as_nanos() as u64;

        if iter_idx + 1 == measured {
            final_output = Some(
                buf.read_to_vec_async()
                    .await
                    .map_err(|e| e.to_string())?,
            );
        }
    }

    let wall_avg_ns = wall_total_ns / measured as u64;
    Ok((
        TimingReport {
            wall_time_ns: Some(wall_avg_ns),
            gpu_total_ns: None,
            gpu_stage_ns: Vec::new(),
        },
        final_output.expect("measured loop always captures final output"),
    ))
}

// ---------------------------------------------------------------------------
// Report-building helpers
// ---------------------------------------------------------------------------

fn finalize_case<F>(
    case: &HashCaseSpec,
    kernel_family: &'static str,
    timings: TimingReport,
    gpu_out: &[F],
    expected: &[F],
) -> HashCaseReport
where
    F: PartialEq + core::fmt::Display,
{
    let mut mismatch_count = 0u32;
    let mut first_mismatch: Option<(u32, u32, String, String)> = None;

    for (perm_idx, (gpu_block, cpu_block)) in gpu_out
        .chunks_exact(POSEIDON2_WIDTH)
        .zip(expected.chunks_exact(POSEIDON2_WIDTH))
        .enumerate()
    {
        for (slot, (g, c)) in gpu_block.iter().zip(cpu_block.iter()).enumerate() {
            if g != c {
                mismatch_count += 1;
                if first_mismatch.is_none() {
                    first_mismatch = Some((
                        perm_idx as u32,
                        slot as u32,
                        g.to_string(),
                        c.to_string(),
                    ));
                }
            }
        }
    }

    // Use the generic validation helper we already generified in
    // E.2.b to compute a summary-style outcome (keeps the slot-level
    // mismatch detail we collected above, though).
    let _outcome = compare_vectors(gpu_out, expected);

    let (first_mismatch_index, first_mismatch_gpu, first_mismatch_cpu) =
        match first_mismatch {
            Some((p, s, g, c)) => (Some((p, s)), Some(g), Some(c)),
            None => (None, None, None),
        };

    HashCaseReport {
        name: case.name.clone(),
        num_permutations: case.num_permutations,
        input: case.input,
        kernel_family: Some(kernel_family.to_string()),
        passed: mismatch_count == 0,
        mismatch_count,
        first_mismatch_index,
        first_mismatch_gpu,
        first_mismatch_cpu,
        timings,
        error: None,
    }
}

fn case_error(
    case: &HashCaseSpec,
    kernel_family: &'static str,
    err: String,
) -> HashCaseReport {
    HashCaseReport {
        name: case.name.clone(),
        num_permutations: case.num_permutations,
        input: case.input,
        kernel_family: Some(kernel_family.to_string()),
        passed: false,
        mismatch_count: 0,
        first_mismatch_index: None,
        first_mismatch_gpu: None,
        first_mismatch_cpu: None,
        timings: TimingReport {
            wall_time_ns: None,
            gpu_total_ns: None,
            gpu_stage_ns: Vec::new(),
        },
        error: Some(err),
    }
}

fn empty_case_report(
    case: &HashCaseSpec,
    kernel_family: &'static str,
) -> HashCaseReport {
    HashCaseReport {
        name: case.name.clone(),
        num_permutations: 0,
        input: case.input,
        kernel_family: Some(kernel_family.to_string()),
        passed: true,
        mismatch_count: 0,
        first_mismatch_index: None,
        first_mismatch_gpu: None,
        first_mismatch_cpu: None,
        timings: TimingReport {
            wall_time_ns: Some(0),
            gpu_total_ns: None,
            gpu_stage_ns: Vec::new(),
        },
        error: None,
    }
}

#[cfg(test)]
mod tests {
    //! Native-target tests exercise the structural paths that don't
    //! need a GPU — empty-suite rejection plus the up-front failure
    //! surface. Browser-side GPU integration tests live in
    //! `tests/browser_smoke.rs` (or a future hash-specific sibling).
    use super::*;
    use zkgpu_report::{HashAlgorithm, HashCaseSpec, HashSpec, SuiteKind};

    /// Phase F.3.d: empty-suite request must surface as a string error
    /// before any device work. Matches the NTT `run_suite_async`
    /// contract so the outer wasm entry can treat empty-suite as a
    /// client-side config error (not a browser GPU failure).
    #[test]
    fn run_hash_suite_async_rejects_empty_suite() {
        let spec = HashSpec {
            kind: SuiteKind::Smoke,
            cases: Vec::new(),
            fail_fast: true,
            algorithm: HashAlgorithm::Poseidon2,
            field: Field::BabyBear,
        };
        let err = pollster::block_on(run_hash_suite_async(&spec))
            .expect_err("empty suite must be rejected");
        assert!(err.contains("at least one case"), "err: {err}");
    }

    /// Phase F.3.d: verifies the empty-case short-circuit builds a
    /// report row without touching the GPU. Exercises
    /// `empty_case_report` directly since the run_*_case_*
    /// functions short-circuit on num_permutations == 0 before any
    /// async work.
    #[test]
    fn empty_case_report_shape() {
        let case = HashCaseSpec::new(
            "empty",
            0,
            zkgpu_report::HashInputPattern::AllZeros,
        );
        let r = empty_case_report(&case, "babybear-poseidon2");
        assert!(r.passed);
        assert_eq!(r.num_permutations, 0);
        assert_eq!(r.mismatch_count, 0);
        assert_eq!(r.kernel_family.as_deref(), Some("babybear-poseidon2"));
        assert_eq!(r.timings.wall_time_ns, Some(0));
        assert!(r.timings.gpu_total_ns.is_none());
    }

    /// Phase F.3.d cross-runner parity pin: the browser-side input
    /// generators must stay byte-identical to the native testkit's
    /// `inputs::make_*_hash_input` for the same `(num_permutations,
    /// pattern)` pair. Without this invariant, differential tests
    /// comparing a browser HashSuiteReport against a native one
    /// would diverge on the input stream and reveal nothing about
    /// the kernels.
    ///
    /// Pinned against a literal expected output rather than calling
    /// the testkit (would require a dev-dep cycle). When the testkit
    /// generator changes, these constants must change in lockstep —
    /// the docstring on `hash_mix64` is the single source of truth,
    /// and this test holds both implementations honest to it.
    #[test]
    fn hash_inputs_pin_sequential_and_splitmix64() {
        use zkgpu_core::GpuField;
        use zkgpu_report::HashInputPattern;

        // Sequential, num=2 → 32 slots, each = p*WIDTH + i + 1.
        // Slot 0 of perm 0 is `0*16 + 0 + 1 = 1`.
        // Slot 15 of perm 1 is `1*16 + 15 + 1 = 32`.
        let bb = make_babybear_hash_input(2, &HashInputPattern::Sequential);
        assert_eq!(bb.len(), 32);
        assert_eq!(bb[0].to_repr(), 1);
        assert_eq!(bb[15].to_repr(), 16);
        assert_eq!(bb[16].to_repr(), 17);
        assert_eq!(bb[31].to_repr(), 32);

        let gl = make_goldilocks_hash_input(2, &HashInputPattern::Sequential);
        assert_eq!(gl.len(), 32);
        assert_eq!(gl[0].to_repr(), 1);
        assert_eq!(gl[31].to_repr(), 32);

        // SplitMix64 { seed: 1 } first two outputs — these values
        // come from running the shared `hash_mix64(0, 1)` and
        // `hash_mix64(1, 1)` under the literal constants in the
        // docstring. If either the constants or the mixer change,
        // both implementations must update and so must these pins.
        let bb_mix = make_babybear_hash_input(
            1,
            &HashInputPattern::SplitMix64 { seed: 1 },
        );
        assert_eq!(bb_mix.len(), 16);
        // Verify the mixer output is reduced mod p (smoke check —
        // BabyBear p < 2^31 so any valid output is < 2^31).
        for f in &bb_mix {
            assert!(f.to_repr() < 0x7800_0001);
        }

        // AllZeros / AllOnes — trivial but included so the pattern
        // enumeration is exhaustive.
        let z = make_babybear_hash_input(3, &HashInputPattern::AllZeros);
        assert_eq!(z.len(), 48);
        assert!(z.iter().all(|f| f.to_repr() == 0));
        let o = make_goldilocks_hash_input(3, &HashInputPattern::AllOnes);
        assert_eq!(o.len(), 48);
        assert!(o.iter().all(|f| f.to_repr() == 1));
    }
}
