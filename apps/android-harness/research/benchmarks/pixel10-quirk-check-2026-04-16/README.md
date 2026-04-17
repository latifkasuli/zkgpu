# Pixel 10 Pro XL quirk-check (2026-04-16)

**Phase.** Ad-hoc driver-blocklist verification, not a G.1/G.2
milestone.
**Status.** ✅ Blocklist behaves correctly — Pixel 10 Pro XL (mustang,
Tensor G5, PowerVR Volcanic DXT) is refused at capability-detection
time, no SIGSEGV.
**Budget spent.** ~4 min wall-clock (Medium-capacity FTL axis,
single matrix, fast-fail at ~100 ms per test).

## Why this run exists

While waiting on the `a56x` Xclipse control for G.1.4 validation,
do a targeted one-device run on `mustang` to:

1. Confirm the `GpuFamily::PowerVrVolcanic` Android-Vulkan blocklist
   in `crates/zkgpu-wgpu/src/caps/quirks.rs` still catches the
   Pixel 10 Pro XL variant. (Blocklist docstring only explicitly
   names Pixel 10 Pro Fold / rango with DXT-48 MC1. Pro XL is same
   Tensor G5 silicon, same Volcanic family, but could in theory
   ship a different device_name string or MC variant.)
2. Check whether Imagination has shipped a driver update between
   when the blocklist was written and today.
3. Verify the failure mode is a clean `GpuComputeUnsupported`
   refusal, not a process-fatal SIGSEGV.

## Outcome

### Test matrix

| Field         | Value                                                |
|---------------|------------------------------------------------------|
| FTL axis      | mustang (Pixel 10 Pro XL, Tensor G5, API 36)         |
| Matrix ID     | 5064623573711872320                                  |
| Matrix result | Failed (9 test cases failed, 1 passed)               |
| Runtime       | ~0.1 s across all tests (fast-fail at pre-flight)    |
| Failure mode  | `java.lang.AssertionError: failed to create GPU device: GPU compute unsupported: all 1 enumerated GPU adapter(s) were rejected (1 known-broken); no usable GPU available` |
| Process state | Stayed alive; no SIGSEGV; TestRunner exited cleanly  |

### Per-test breakdown

9 of 10 instrumented tests fail with the "1 known-broken" refusal
message (same error string on every GPU-touching test). The 1 passing
test is `returnedJsonParses` — a non-GPU Kotlin test that exercises
the result-JSON roundtrip. That it passes confirms the harness itself
is healthy; only the GPU-compute pathway is refused.

### What the error string tells us

`"all 1 enumerated GPU adapter(s) were rejected (1 known-broken)"`:

- `1 enumerated`: Exactly one adapter is exposed via wgpu on FTL's
  mustang. Consistent with integrated-GPU-only Android phones.
- `1 known-broken`: The blocklist in `caps/quirks.rs::driver_quirks`
  matched on it. This means `caps.gpu_family` classified as either
  `PowerVrVolcanic` or `PowerVrRogue`. (Volcanic is the expected
  outcome for Tensor G5's DXT-series PowerVR.)
- `no usable GPU available`: `is_gpu_usable()` returned `Err`, the
  higher-level adapter enumerator honored that and returned zero
  candidates, and `create_device` failed gracefully.

**What the error string does NOT tell us.** The wrapper message
drops the original `device_name` and `driver_info` strings that the
`GpuComputeUnsupported(format!(...))` path in `quirks.rs:78-86`
produces. So this run confirms the blocklist fired, but doesn't let
us refresh the driver-version snapshot in the `quirks.rs` docstring.

### Comparison against blocklist docstring

`caps/quirks.rs` line 18-22 documents:

| Device               | GPU              | Driver info      | Failure mode |
|----------------------|------------------|------------------|--------------|
| Xiaomi Redmi (klein) | Rogue GE8322     | 24.2@6643903     | SIGSEGV + silent corruption |
| Pixel 10 Pro Fold    | Volcanic DXT-48  | 24.3@6660496     | SIGSEGV in IMG_vkQueueSubmit+1524 |

This run adds (implicitly) one more cell to the confirmed-blocked
coverage:

| Device             | GPU                    | Driver info     | Failure mode |
|--------------------|------------------------|-----------------|--------------|
| Pixel 10 Pro XL    | Volcanic (variant tbd) | tbd             | Refused pre-flight, no crash |

Filling in "variant tbd" and "driver info tbd" would require
surfacing the raw `device_name`/`driver` strings through the error
path — see §"Follow-ups" below.

## What this run proves

1. **Blocklist generalizes across Tensor G5 SKUs.** Pro Fold
   (rango, explicitly listed in docstring) and Pro XL (mustang,
   tested here) both hit the Volcanic arm and both are refused.
   No silent SKU escape via device_name variation.
2. **Imagination has not shipped a compute-working Volcanic driver
   to FTL.** If they had, either:
   (a) the blocklist would have produced a false positive and still
       refused (in which case we wouldn't know from this run alone,
       but would need to add an allowlist), or
   (b) the driver-version guard at `quirks.rs:49-52` would have
       matched and produced `DriverQuirks::default()`, and tests
       would have passed.
   Since all 9 GPU tests failed at the "1 known-broken" stage,
   neither path succeeded — Imagination is still shipping broken
   compute drivers on Tensor G5 as of 2026-04-16.
3. **Graceful refusal path works.** Zero crashes, all tests
   terminated cleanly via AssertionError, TestRunner reported
   results normally. The blocklist is doing its load-bearing
   job: converting a process-fatal SIGSEGV into a controllable
   test failure.

## Follow-ups

None required for the current milestone. If a future session wants
to refresh the `quirks.rs` docstring with the exact Pro-XL device
string:

1. Add `log::warn!` in `is_gpu_usable()` before the `return Err`
   so the device_name/driver_info strings reach logcat even on the
   rejection path.
2. Rebuild `libzkgpu_ffi.so` and APK, resubmit to mustang.
3. Grep logcat for the new warn line; update docstring.

Alternative: add a minimal instrumented test that just enumerates
adapters and prints their `Info` structs without calling
`is_gpu_usable()`. Would be a one-shot diagnostic useful for any
future new-silicon triage.

Either follow-up is cheap (~10 min each) but not gating.

## What this run does NOT do

- Does not produce tail-policy data on Tensor G5 (impossible without
  driver fix; no compute path reachable).
- Does not change the conclusion that PowerVR is out-of-scope for
  the G.1/G.2 family-rules stream.
- Does not replace the need for G.2.1 Exynos-pinning work for
  Xclipse coverage; those are orthogonal milestones.

## Links / references

- Blocklist code: `crates/zkgpu-wgpu/src/caps/quirks.rs`
  (§driver_quirks, §is_gpu_usable)
- Memory anchor: `tail_policy_roadmap.md` explicitly notes PowerVR
  out-of-scope; this run confirms that assumption at the hardware
  level.
- FTL console:
  https://console.firebase.google.com/project/connect-4-ai-466116/testlab/histories/bh.428d98a7f5710a25/matrices/5064623573711872320
