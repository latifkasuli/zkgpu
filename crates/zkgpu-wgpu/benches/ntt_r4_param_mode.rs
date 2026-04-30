//! A/B bench for Gate 2 item #3 — R4 stage params storage vs immediate.
//!
//! Measures `WgpuNttPlan::execute` end-to-end at four log_n sizes
//! (10, 14, 18, 20) under both `R4ParamMode::Storage` (today's
//! default) and `R4ParamMode::Immediate` (the pilot path). Plan-build
//! and upload are pulled out of the timing loop; only the GPU NTT
//! itself is measured.
//!
//! Two log_n regimes:
//!
//! - `log_n=10` and `log_n=14`: the regime where item #3's CPU-side
//!   savings (one fewer bind-group entry per R4 stage, no per-stage
//!   `wgpu::Buffer` allocation) compete with kernel time. A win here
//!   is the encoder-overhead win the speed-opportunities doc
//!   predicted for "small log_n on integrated/mobile/browser".
//!
//! - `log_n=18` and `log_n=20`: the regime where kernel time
//!   dominates. Any win here would suggest the immediate-vs-uniform
//!   read path is faster for the R4 kernel itself, not just the
//!   encoder. Less likely a-priori, but the plonky3 hot path lives
//!   here so even a small delta compounds.
//!
//! `R4ParamMode::Immediate` requires `wgpu::Features::IMMEDIATES`;
//! the bench panics with a clear message if the device doesn't
//! advertise it. M4 Pro (Metal), RTX 4090/5090 (Vulkan), and modern
//! WebGPU all support it.
//!
//! Output groups:
//! - `ntt_r4_storage` — per-stage uniform-buffer params (today)
//! - `ntt_r4_immediate` — per-stage immediates (pilot)
//!
//! Comparison: Criterion writes both groups to `target/criterion/`;
//! eyeballing the median deltas at each log_n gives the signal.
//! Decision criterion (mirrors item #6's gate shape): ≥ 5% win on
//! at least one log_n on at least one host = propagate to R2/local/
//! Poseidon2 in a follow-up. Null/regression on every host = keep
//! the pilot path as documented and move to item #5.

use std::sync::OnceLock;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuDevice, NttDirection, NttPlan};
use zkgpu_wgpu::{R4ParamMode, WgpuDevice, WgpuNttPlan};

static DEVICE: OnceLock<WgpuDevice> = OnceLock::new();

fn device() -> &'static WgpuDevice {
    DEVICE.get_or_init(|| {
        let d = WgpuDevice::new().expect("GPU device required for benchmarks");
        eprintln!("Benchmark device: {}", d.caps());
        if !d.caps().has_immediates {
            panic!(
                "ntt_r4_param_mode bench requires `wgpu::Features::IMMEDIATES`; \
                 the active device does not advertise it. Run on a host with a \
                 wgpu v29 backend that supports immediates (Metal, Vulkan with a \
                 modern driver, DX12, or modern WebGPU)."
            );
        }
        d
    })
}

const LOG_NS: &[u32] = &[10, 14, 18, 20];

fn make_data(log_n: u32) -> Vec<BabyBear> {
    let n = 1usize << log_n;
    (0..n as u32).map(BabyBear::new).collect()
}

fn bench_one_mode(c: &mut Criterion, group_name: &str, mode: R4ParamMode) {
    let dev = device();
    let mut group = c.benchmark_group(group_name);

    for &log_n in LOG_NS {
        let n = 1u64 << log_n;
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(
            BenchmarkId::from_parameter(log_n),
            &log_n,
            |b, &log_n| {
                let mut plan = WgpuNttPlan::new_with_r4_param_mode(
                    dev,
                    log_n,
                    NttDirection::Forward,
                    mode,
                )
                .expect("plan creation failed");
                let data = make_data(log_n);
                // Warm the pipeline + first command-buffer encode so
                // the first `b.iter` call doesn't pay shader compile
                // or first-encode latency.
                let mut warm_buf = dev.upload(&data).expect("upload failed");
                plan.execute(dev, &mut warm_buf).expect("warmup failed");
                b.iter_batched(
                    || dev.upload(&data).expect("upload failed"),
                    |mut buf| {
                        plan.execute(dev, &mut buf).expect("execute failed");
                        buf
                    },
                    criterion::BatchSize::LargeInput,
                );
            },
        );
    }
    group.finish();
}

fn bench_storage(c: &mut Criterion) {
    bench_one_mode(c, "ntt_r4_storage", R4ParamMode::Storage);
}

fn bench_immediate(c: &mut Criterion) {
    bench_one_mode(c, "ntt_r4_immediate", R4ParamMode::Immediate);
}

criterion_group!(benches, bench_storage, bench_immediate);
criterion_main!(benches);
