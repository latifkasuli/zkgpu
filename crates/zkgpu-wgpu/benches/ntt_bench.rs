use std::sync::OnceLock;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use zkgpu_babybear::BabyBear;
use zkgpu_core::{GpuDevice, NttDirection, NttPlan};
use zkgpu_ntt::ntt_cpu_reference;
use zkgpu_wgpu::{WgpuDevice, WgpuNttPlan};

static DEVICE: OnceLock<WgpuDevice> = OnceLock::new();

fn device() -> &'static WgpuDevice {
    DEVICE.get_or_init(|| {
        let d = WgpuDevice::new().expect("GPU device required for benchmarks");
        eprintln!("Benchmark device: {}", d.caps());
        d
    })
}

fn make_data(log_n: u32) -> Vec<BabyBear> {
    let n = 1usize << log_n;
    (0..n as u32).map(BabyBear::new).collect()
}

fn bench_cpu_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_forward");
    for log_n in [10, 14, 18, 20] {
        let n = 1u64 << log_n;
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(log_n), &log_n, |b, &log_n| {
            let data = make_data(log_n);
            b.iter_batched(
                || data.clone(),
                |mut d| ntt_cpu_reference(&mut d, NttDirection::Forward),
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_cpu_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_inverse");
    for log_n in [10, 14, 18, 20] {
        let n = 1u64 << log_n;
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(log_n), &log_n, |b, &log_n| {
            let data = make_data(log_n);
            b.iter_batched(
                || data.clone(),
                |mut d| ntt_cpu_reference(&mut d, NttDirection::Inverse),
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_gpu_forward_e2e(c: &mut Criterion) {
    let dev = device();
    let mut group = c.benchmark_group("gpu_forward_e2e");

    for log_n in [10, 14, 18, 20] {
        let n = 1u64 << log_n;
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(log_n), &log_n, |b, &log_n| {
            let data = make_data(log_n);
            let mut plan =
                WgpuNttPlan::new(dev, log_n, NttDirection::Forward).expect("plan creation failed");
            b.iter_batched(
                || dev.upload(&data).expect("upload failed"),
                |mut buf| {
                    plan.execute(dev, &mut buf).expect("execute failed");
                    buf
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_gpu_inverse_e2e(c: &mut Criterion) {
    let dev = device();
    let mut group = c.benchmark_group("gpu_inverse_e2e");

    for log_n in [10, 14, 18, 20] {
        let n = 1u64 << log_n;
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(log_n), &log_n, |b, &log_n| {
            let data = make_data(log_n);
            let mut plan =
                WgpuNttPlan::new(dev, log_n, NttDirection::Inverse).expect("plan creation failed");
            b.iter_batched(
                || dev.upload(&data).expect("upload failed"),
                |mut buf| {
                    plan.execute(dev, &mut buf).expect("execute failed");
                    buf
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_gpu_forward_kernel(c: &mut Criterion) {
    let dev = device();
    let mut group = c.benchmark_group("gpu_forward_kernel");

    for log_n in [10, 14, 18, 20] {
        let n = 1u64 << log_n;
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(log_n), &log_n, |b, &log_n| {
            let data = make_data(log_n);
            let mut plan =
                WgpuNttPlan::new(dev, log_n, NttDirection::Forward).expect("plan creation failed");
            b.iter_batched(
                || dev.upload(&data).expect("upload failed"),
                |mut buf| {
                    plan.execute_kernels(dev, &mut buf).expect("kernel failed");
                    buf
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

fn bench_gpu_inverse_kernel(c: &mut Criterion) {
    let dev = device();
    let mut group = c.benchmark_group("gpu_inverse_kernel");

    for log_n in [10, 14, 18, 20] {
        let n = 1u64 << log_n;
        group.throughput(Throughput::Elements(n));
        group.bench_with_input(BenchmarkId::from_parameter(log_n), &log_n, |b, &log_n| {
            let data = make_data(log_n);
            let mut plan =
                WgpuNttPlan::new(dev, log_n, NttDirection::Inverse).expect("plan creation failed");
            b.iter_batched(
                || dev.upload(&data).expect("upload failed"),
                |mut buf| {
                    plan.execute_kernels(dev, &mut buf).expect("kernel failed");
                    buf
                },
                criterion::BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_cpu_forward,
    bench_cpu_inverse,
    bench_gpu_forward_e2e,
    bench_gpu_inverse_e2e,
    bench_gpu_forward_kernel,
    bench_gpu_inverse_kernel,
);
criterion_main!(benches);
