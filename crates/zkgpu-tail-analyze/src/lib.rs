//! Logcat parser + threshold recommender for the Stockham tail A/B.
//!
//! PR 1 of the tail-strategy refactor added forced-A/B instrumented tests to
//! `apps/android-harness`:
//!
//!   * `crossoverStockhamLocalTail`  — emits `CROSSOVER_STOCKHAM_LOCAL_TAIL`
//!   * `crossoverStockhamGlobalTail` — emits `CROSSOVER_STOCKHAM_GLOBAL_TAIL`
//!
//! Each test logs one line per case with the shape
//!
//! ```text
//! ZkgpuHarnessTest: <TAG> <name>: family=<f> tail=<t> reason=<r> \
//!     stride_bytes=<s> wall=<w>ms gpu=<g>ms
//! ```
//!
//! where `<name>` is `forward_log18` / `inverse_log22` / etc. PR 2 consumes
//! those logcats, pairs (Local, Global) at each `log_n`, and applies the
//! research/stockham-local-fused-rewrite.md §4 decision bar to recommend a
//! per-device threshold:
//!
//!   * `never` (GlobalOnlyR4) wins by ≥20% at log_n ≥ 21 on both devices
//!     → unconditional threshold flip.
//!   * Wins narrowly (5–20%) → per-device threshold.
//!   * Loses or negligible → no change; gather deeper signal first.
//!
//! Pure data; no GPU calls. Tests exercise the Xclipse 540 scaling table
//! from the research doc directly.

use std::collections::BTreeMap;

/// Which forced-A/B variant this logcat line came from. Derived from the
/// per-test `tagPrefix` in `ZkgpuInstrumentedTest.kt`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TailVariant {
    /// `CROSSOVER_STOCKHAM_LOCAL_TAIL` — `StockhamTailOverride::Local`.
    Local,
    /// `CROSSOVER_STOCKHAM_GLOBAL_TAIL` — `StockhamTailOverride::Global`.
    Global,
}

impl TailVariant {
    pub fn tag(self) -> &'static str {
        match self {
            TailVariant::Local => "CROSSOVER_STOCKHAM_LOCAL_TAIL",
            TailVariant::Global => "CROSSOVER_STOCKHAM_GLOBAL_TAIL",
        }
    }

    fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "CROSSOVER_STOCKHAM_LOCAL_TAIL" => Some(TailVariant::Local),
            "CROSSOVER_STOCKHAM_GLOBAL_TAIL" => Some(TailVariant::Global),
            _ => None,
        }
    }
}

/// Forward / inverse direction extracted from the case name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Direction {
    Forward,
    Inverse,
}

/// One parsed `CROSSOVER_STOCKHAM_*_TAIL` line.
#[derive(Debug, Clone, PartialEq)]
pub struct CaseTiming {
    pub variant: TailVariant,
    pub log_n: u32,
    pub direction: Direction,
    pub family: String,
    pub tail: String,
    pub reason: String,
    /// `stride_bytes=none` parses to `None`.
    pub stride_bytes: Option<u64>,
    pub wall_ms: f64,
    pub gpu_ms: f64,
}

/// Parse a whole logcat blob (as written by FTL to `logcat.txt`). Any line
/// that doesn't match the expected shape is silently skipped — logcats mix
/// output from unrelated processes.
pub fn parse_logcat(text: &str) -> Vec<CaseTiming> {
    text.lines().filter_map(parse_line).collect()
}

/// Parse one line. `None` if it's not a `CROSSOVER_STOCKHAM_*_TAIL` record.
///
/// Accepts both raw lines (`CROSSOVER_STOCKHAM_LOCAL_TAIL …`) and full FTL
/// logcat lines (`04-15 10:00:00.000 1 1 I ZkgpuHarnessTest: …`).
pub fn parse_line(line: &str) -> Option<CaseTiming> {
    // Find the tag anchor, then slice from there — lets us ignore logcat's
    // timestamp/pid/tid/level prefix without a full parser.
    let (tag, rest) = find_tag_split(line)?;
    let variant = TailVariant::from_tag(tag)?;

    // rest starts with "<name>: family=... tail=... reason=... stride_bytes=... wall=...ms gpu=...ms"
    let (name, body) = rest.split_once(':')?;
    let name = name.trim();
    let body = body.trim();

    let (direction, log_n) = parse_case_name(name)?;

    let family = extract_kv(body, "family=")?.to_string();
    let tail = extract_kv(body, "tail=")?.to_string();
    let reason = extract_kv(body, "reason=")?.to_string();
    let stride_raw = extract_kv(body, "stride_bytes=")?;
    let stride_bytes = if stride_raw == "none" {
        None
    } else {
        stride_raw.parse::<u64>().ok()
    };
    let wall_ms = parse_ms(extract_kv(body, "wall=")?)?;
    let gpu_ms = parse_ms(extract_kv(body, "gpu=")?)?;

    Some(CaseTiming {
        variant,
        log_n,
        direction,
        family,
        tail,
        reason,
        stride_bytes,
        wall_ms,
        gpu_ms,
    })
}

fn find_tag_split(line: &str) -> Option<(&str, &str)> {
    for tag in [
        TailVariant::Local.tag(),
        TailVariant::Global.tag(),
    ] {
        if let Some(idx) = line.find(tag) {
            let after = &line[idx + tag.len()..];
            let after = after.trim_start();
            return Some((tag, after));
        }
    }
    None
}

fn parse_case_name(name: &str) -> Option<(Direction, u32)> {
    // `forward_log18` / `inverse_log22`
    let (dir, rest) = name.split_once('_')?;
    let direction = match dir {
        "forward" => Direction::Forward,
        "inverse" => Direction::Inverse,
        _ => return None,
    };
    let log_n = rest.strip_prefix("log")?.parse::<u32>().ok()?;
    Some((direction, log_n))
}

/// Extract the whitespace-delimited value that follows `key=`, preserving
/// quoted strings up to the matching quote. `None` if the key is absent.
fn extract_kv<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let idx = body.find(key)?;
    let after = &body[idx + key.len()..];
    // Values never contain whitespace in our format; trailing 'ms' is left
    // on timing values and stripped by `parse_ms`.
    let end = after
        .find(|c: char| c.is_whitespace())
        .unwrap_or(after.len());
    Some(&after[..end])
}

fn parse_ms(raw: &str) -> Option<f64> {
    raw.strip_suffix("ms").and_then(|s| s.parse::<f64>().ok())
}

// ---------------------------------------------------------------------------
// Pairing + recommendation
// ---------------------------------------------------------------------------

/// Per-(log_n, direction) (local, global) pair. Either side may be absent
/// if the matching test didn't run.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedCase {
    pub log_n: u32,
    pub direction: Direction,
    pub local_gpu_ms: Option<f64>,
    pub global_gpu_ms: Option<f64>,
}

impl PairedCase {
    /// Relative GPU-time delta: `(local - global) / local`. Positive means
    /// Global is faster (win for `GlobalOnlyR4`). `None` if either side is
    /// missing or `local_gpu_ms` is zero.
    pub fn global_win_ratio(&self) -> Option<f64> {
        let (l, g) = (self.local_gpu_ms?, self.global_gpu_ms?);
        if l <= 0.0 {
            return None;
        }
        Some((l - g) / l)
    }
}

/// Pair `Local`/`Global` timings by `(log_n, direction)`. The output is
/// stably ordered by `(log_n, direction)` so the CLI table is deterministic.
pub fn pair_cases(cases: &[CaseTiming]) -> Vec<PairedCase> {
    let mut map: BTreeMap<(u32, Direction), PairedCase> = BTreeMap::new();
    for c in cases {
        let slot = map.entry((c.log_n, c.direction)).or_insert(PairedCase {
            log_n: c.log_n,
            direction: c.direction,
            local_gpu_ms: None,
            global_gpu_ms: None,
        });
        match c.variant {
            TailVariant::Local => slot.local_gpu_ms = Some(c.gpu_ms),
            TailVariant::Global => slot.global_gpu_ms = Some(c.gpu_ms),
        }
    }
    map.into_values().collect()
}

/// Verdict from applying the §4 decision bar of
/// `research/stockham-local-fused-rewrite.md` to one device's paired cases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Recommendation {
    /// Global wins by ≥20% at the threshold log_n on a majority of paired
    /// cases — apply the flip unconditionally.
    Unconditional { threshold_log_n: u32 },
    /// Global wins by 5–20% — flip per-device at the suggested threshold.
    PerDevice { threshold_log_n: u32 },
    /// No clear win; keep the existing heuristic and gather deeper signal
    /// (shmem counters, stride sweep).
    NoChange,
    /// Not enough paired data to decide (e.g. only one side of the A/B ran,
    /// or fewer than two log_n buckets).
    InsufficientData,
}

/// Decision-bar thresholds, in relative GPU-time delta units.
pub const UNCONDITIONAL_WIN: f64 = 0.20;
pub const PER_DEVICE_WIN: f64 = 0.05;

/// Apply the research-doc decision bar to a device's paired cases.
///
/// `candidate_log_ns` controls which sizes are considered the "threshold
/// regime"; the research doc uses `[21, 22]`. The recommended threshold is
/// the smallest candidate where the average `global_win_ratio` across
/// directions clears the bar.
pub fn recommend(paired: &[PairedCase], candidate_log_ns: &[u32]) -> Recommendation {
    // Group wins by log_n, averaging forward + inverse.
    let mut by_log_n: BTreeMap<u32, Vec<f64>> = BTreeMap::new();
    for p in paired {
        if let Some(ratio) = p.global_win_ratio() {
            by_log_n.entry(p.log_n).or_default().push(ratio);
        }
    }
    if by_log_n.len() < 2 {
        return Recommendation::InsufficientData;
    }

    for &log_n in candidate_log_ns {
        let Some(ratios) = by_log_n.get(&log_n) else {
            continue;
        };
        if ratios.is_empty() {
            continue;
        }
        let avg = ratios.iter().sum::<f64>() / ratios.len() as f64;
        if avg >= UNCONDITIONAL_WIN {
            return Recommendation::Unconditional {
                threshold_log_n: log_n,
            };
        }
        if avg >= PER_DEVICE_WIN {
            return Recommendation::PerDevice {
                threshold_log_n: log_n,
            };
        }
    }
    Recommendation::NoChange
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_line(tag: &str, name: &str, tail: &str, stride: &str, gpu_ms: f64) -> String {
        // Full FTL logcat shape; parser must tolerate the prefix.
        format!(
            "04-15 10:00:00.000 12345 12345 I ZkgpuHarnessTest: {tag} {name}: \
             family=stockham tail={tail} reason=Forced{} \
             stride_bytes={stride} wall={:.2}ms gpu={:.2}ms",
            if tail == "LocalFusedR4" { "Local" } else { "Global" },
            gpu_ms + 2.0,
            gpu_ms,
        )
    }

    #[test]
    fn parse_local_tail_line() {
        let line = mk_line(
            "CROSSOVER_STOCKHAM_LOCAL_TAIL",
            "forward_log18",
            "LocalFusedR4",
            "8",
            6.34,
        );
        let t = parse_line(&line).expect("parses");
        assert_eq!(t.variant, TailVariant::Local);
        assert_eq!(t.log_n, 18);
        assert_eq!(t.direction, Direction::Forward);
        assert_eq!(t.tail, "LocalFusedR4");
        assert_eq!(t.reason, "ForcedLocal");
        assert_eq!(t.stride_bytes, Some(8));
        assert!((t.gpu_ms - 6.34).abs() < 1e-9);
    }

    #[test]
    fn parse_global_tail_line_with_no_stride() {
        let line = mk_line(
            "CROSSOVER_STOCKHAM_GLOBAL_TAIL",
            "inverse_log22",
            "GlobalOnlyR4",
            "none",
            48.30,
        );
        let t = parse_line(&line).expect("parses");
        assert_eq!(t.variant, TailVariant::Global);
        assert_eq!(t.log_n, 22);
        assert_eq!(t.direction, Direction::Inverse);
        assert_eq!(t.tail, "GlobalOnlyR4");
        assert_eq!(t.stride_bytes, None);
        assert!((t.gpu_ms - 48.30).abs() < 1e-9);
    }

    #[test]
    fn non_tail_lines_are_ignored() {
        let blob = "\
04-15 10:00:00.000 1 1 I ZkgpuHarnessTest: BENCH forward_log10: wall=1.22ms gpu=0.32ms
04-15 10:00:00.001 1 1 I ZkgpuHarnessTest: CROSSOVER forward_log18: wall=10.82ms gpu=8.87ms
04-15 10:00:00.002 1 1 I random_other_tag: noise
";
        assert!(parse_logcat(blob).is_empty());
    }

    #[test]
    fn parse_mixed_blob_returns_only_tail_lines() {
        let local = mk_line(
            "CROSSOVER_STOCKHAM_LOCAL_TAIL",
            "forward_log18",
            "LocalFusedR4",
            "8",
            6.34,
        );
        let global = mk_line(
            "CROSSOVER_STOCKHAM_GLOBAL_TAIL",
            "forward_log18",
            "GlobalOnlyR4",
            "none",
            5.10,
        );
        let noise =
            "04-15 10:00:00.003 1 1 I ZkgpuHarnessTest: CROSSOVER forward_log18: wall=10.82ms gpu=8.87ms";
        let blob = format!("{local}\n{noise}\n{global}\n");

        let parsed = parse_logcat(&blob);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].variant, TailVariant::Local);
        assert_eq!(parsed[1].variant, TailVariant::Global);
    }

    #[test]
    fn pair_cases_matches_by_log_n_and_direction() {
        let cases = vec![
            parse_line(&mk_line(
                "CROSSOVER_STOCKHAM_LOCAL_TAIL",
                "forward_log18",
                "LocalFusedR4",
                "8",
                10.0,
            ))
            .unwrap(),
            parse_line(&mk_line(
                "CROSSOVER_STOCKHAM_GLOBAL_TAIL",
                "forward_log18",
                "GlobalOnlyR4",
                "none",
                8.0,
            ))
            .unwrap(),
            parse_line(&mk_line(
                "CROSSOVER_STOCKHAM_LOCAL_TAIL",
                "inverse_log22",
                "LocalFusedR4",
                "8",
                50.0,
            ))
            .unwrap(),
        ];

        let paired = pair_cases(&cases);
        assert_eq!(paired.len(), 2);
        assert_eq!(paired[0].log_n, 18);
        assert_eq!(paired[0].local_gpu_ms, Some(10.0));
        assert_eq!(paired[0].global_gpu_ms, Some(8.0));
        assert!((paired[0].global_win_ratio().unwrap() - 0.2).abs() < 1e-9);

        // Unpaired inverse_log22 — only local side present.
        assert_eq!(paired[1].log_n, 22);
        assert_eq!(paired[1].local_gpu_ms, Some(50.0));
        assert_eq!(paired[1].global_gpu_ms, None);
        assert!(paired[1].global_win_ratio().is_none());
    }

    // -- Decision-bar tests, based on research/stockham-local-fused-rewrite.md §4 --

    fn synth_pair(log_n: u32, direction: Direction, local: f64, global: f64) -> PairedCase {
        PairedCase {
            log_n,
            direction,
            local_gpu_ms: Some(local),
            global_gpu_ms: Some(global),
        }
    }

    #[test]
    fn unconditional_flip_when_global_wins_by_30pct_at_log22() {
        // Research-doc Xclipse 540 scaling table collapse shape: local fused
        // hits 19.64 ns/elem at log22, global stays near 4.77 ns/elem.
        let paired = vec![
            synth_pair(18, Direction::Forward, 4.77, 4.77),
            synth_pair(20, Direction::Forward, 10.0, 8.5),
            synth_pair(22, Direction::Forward, 19.64, 13.0), // 33% win
            synth_pair(22, Direction::Inverse, 20.1, 13.8),  // 31% win
        ];
        let rec = recommend(&paired, &[21, 22]);
        assert_eq!(
            rec,
            Recommendation::Unconditional { threshold_log_n: 22 }
        );
    }

    #[test]
    fn per_device_when_narrow_win_at_log22() {
        let paired = vec![
            synth_pair(18, Direction::Forward, 5.0, 5.0),
            synth_pair(20, Direction::Forward, 10.0, 9.6),
            // 8% average win — inside the 5–20% per-device band.
            synth_pair(22, Direction::Forward, 20.0, 18.4),
            synth_pair(22, Direction::Inverse, 21.0, 19.3),
        ];
        let rec = recommend(&paired, &[21, 22]);
        assert_eq!(rec, Recommendation::PerDevice { threshold_log_n: 22 });
    }

    #[test]
    fn no_change_when_global_loses() {
        let paired = vec![
            synth_pair(18, Direction::Forward, 5.0, 5.5),
            synth_pair(22, Direction::Forward, 20.0, 21.0),
            synth_pair(22, Direction::Inverse, 21.0, 22.0),
        ];
        let rec = recommend(&paired, &[21, 22]);
        assert_eq!(rec, Recommendation::NoChange);
    }

    #[test]
    fn earliest_candidate_wins_even_if_later_also_clears_bar() {
        let paired = vec![
            synth_pair(19, Direction::Forward, 5.0, 5.0),
            synth_pair(21, Direction::Forward, 10.0, 8.0), // 20% → unconditional
            synth_pair(22, Direction::Forward, 20.0, 14.0), // 30%, but 21 came first
        ];
        let rec = recommend(&paired, &[21, 22]);
        assert_eq!(
            rec,
            Recommendation::Unconditional { threshold_log_n: 21 }
        );
    }

    #[test]
    fn insufficient_data_with_fewer_than_two_buckets() {
        let paired = vec![synth_pair(22, Direction::Forward, 20.0, 14.0)];
        let rec = recommend(&paired, &[21, 22]);
        assert_eq!(rec, Recommendation::InsufficientData);
    }

    #[test]
    fn insufficient_data_when_only_local_ran() {
        // Global side missing → no ratio computable.
        let paired = vec![
            PairedCase {
                log_n: 18,
                direction: Direction::Forward,
                local_gpu_ms: Some(5.0),
                global_gpu_ms: None,
            },
            PairedCase {
                log_n: 22,
                direction: Direction::Forward,
                local_gpu_ms: Some(20.0),
                global_gpu_ms: None,
            },
        ];
        let rec = recommend(&paired, &[21, 22]);
        assert_eq!(rec, Recommendation::InsufficientData);
    }
}
