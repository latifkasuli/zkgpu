//! CLI entry point for the Stockham tail A/B analyzer.
//!
//! Usage:
//!
//! ```text
//! zkgpu-tail-analyze <benchmark-dir>
//! ```
//!
//! `<benchmark-dir>` is expected to match the FTL artifact layout:
//!
//! ```text
//! <benchmark-dir>/
//!   <device-a>/logcat.txt
//!   <device-b>/logcat.txt
//!   ...
//! ```
//!
//! Writes a human-readable table + per-device recommendation to stdout,
//! and a machine-readable `tail_ab_report.json` next to the input dir.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::Serialize;

use zkgpu_tail_analyze::{
    pair_cases, parse_logcat, recommend, CaseTiming, Direction, PairedCase, Recommendation,
};

/// Default candidate thresholds per the research doc §4 decision bar.
const DEFAULT_CANDIDATES: [u32; 2] = [21, 22];

#[derive(Serialize)]
struct Report<'a> {
    benchmark_dir: &'a str,
    devices: Vec<DeviceReport>,
}

#[derive(Serialize)]
struct DeviceReport {
    device: String,
    paired_cases: Vec<PairedCaseOut>,
    recommendation: String,
    recommended_threshold_log_n: Option<u32>,
    /// Populated only for `WindowedFlip` verdicts: inclusive `log_n` range
    /// where Global wins by ≥ UNCONDITIONAL_WIN on average.
    #[serde(skip_serializing_if = "Option::is_none")]
    recommended_window: Option<WindowOut>,
}

#[derive(Serialize)]
struct WindowOut {
    start_log_n: u32,
    end_log_n: u32,
    avg_win: f64,
}

#[derive(Serialize)]
struct PairedCaseOut {
    log_n: u32,
    direction: &'static str,
    local_gpu_ms: Option<f64>,
    global_gpu_ms: Option<f64>,
    global_win_ratio: Option<f64>,
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let Some(dir) = args.first() else {
        eprintln!("usage: zkgpu-tail-analyze <benchmark-dir>");
        return ExitCode::from(2);
    };
    let root = Path::new(dir);
    if !root.is_dir() {
        eprintln!("error: {} is not a directory", root.display());
        return ExitCode::from(2);
    }

    let device_dirs = match list_device_dirs(root) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    if device_dirs.is_empty() {
        eprintln!(
            "error: no device subdirectories with logcat.txt found under {}",
            root.display()
        );
        return ExitCode::from(2);
    }

    let mut devices: Vec<DeviceReport> = Vec::new();
    for dev in &device_dirs {
        let logcat_path = dev.join("logcat.txt");
        let text = match fs::read_to_string(&logcat_path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("warn: skipping {}: {e}", logcat_path.display());
                continue;
            }
        };

        let cases: Vec<CaseTiming> = parse_logcat(&text);
        let paired: Vec<PairedCase> = pair_cases(&cases);
        let rec = recommend(&paired, &DEFAULT_CANDIDATES);

        let device_name = dev
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        print_device_table(&device_name, &cases, &paired, &rec);

        devices.push(DeviceReport {
            device: device_name,
            paired_cases: paired
                .iter()
                .map(|p| PairedCaseOut {
                    log_n: p.log_n,
                    direction: match p.direction {
                        Direction::Forward => "Forward",
                        Direction::Inverse => "Inverse",
                    },
                    local_gpu_ms: p.local_gpu_ms,
                    global_gpu_ms: p.global_gpu_ms,
                    global_win_ratio: p.global_win_ratio(),
                })
                .collect(),
            recommendation: recommendation_label(&rec).to_string(),
            recommended_threshold_log_n: match rec {
                Recommendation::Unconditional { threshold_log_n }
                | Recommendation::PerDevice { threshold_log_n } => Some(threshold_log_n),
                _ => None,
            },
            recommended_window: match rec {
                Recommendation::WindowedFlip {
                    start_log_n,
                    end_log_n,
                    avg_win,
                } => Some(WindowOut {
                    start_log_n,
                    end_log_n,
                    avg_win,
                }),
                _ => None,
            },
        });
    }

    // Emit machine-readable JSON next to the input dir, so it can be
    // committed alongside the logcats for future diffing.
    let report = Report {
        benchmark_dir: dir,
        devices,
    };
    let json_path = root.join("tail_ab_report.json");
    match serde_json::to_string_pretty(&report) {
        Ok(s) => {
            if let Err(e) = fs::write(&json_path, s) {
                eprintln!("warn: could not write {}: {e}", json_path.display());
            } else {
                println!("\nreport: {}", json_path.display());
            }
        }
        Err(e) => eprintln!("warn: could not serialize report: {e}"),
    }

    ExitCode::SUCCESS
}

fn list_device_dirs(root: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && path.join("logcat.txt").is_file() {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

fn print_device_table(
    device: &str,
    cases: &[CaseTiming],
    paired: &[PairedCase],
    rec: &Recommendation,
) {
    println!("## {device}");
    println!("  parsed tail-A/B lines: {}", cases.len());
    if paired.is_empty() {
        println!("  (no paired cases — did both forced tests run?)");
        return;
    }

    // Split by direction for a readable table.
    let mut by_dir: BTreeMap<&'static str, Vec<&PairedCase>> = BTreeMap::new();
    for p in paired {
        let key = match p.direction {
            Direction::Forward => "forward",
            Direction::Inverse => "inverse",
        };
        by_dir.entry(key).or_default().push(p);
    }

    for (dir, rows) in &by_dir {
        println!();
        println!(
            "  | log_n | local_ms | global_ms | global_win |  verdict  ({dir})"
        );
        println!(
            "  |-------|----------|-----------|------------|-----------"
        );
        for row in rows {
            let local = row
                .local_gpu_ms
                .map(|v| format!("{v:>8.2}"))
                .unwrap_or_else(|| "        ".to_string());
            let global = row
                .global_gpu_ms
                .map(|v| format!("{v:>9.2}"))
                .unwrap_or_else(|| "         ".to_string());
            let ratio = row
                .global_win_ratio()
                .map(|r| format!("{:>+9.1}%", r * 100.0))
                .unwrap_or_else(|| "     n/a".to_string());
            let verdict = row
                .global_win_ratio()
                .map(classify_win)
                .unwrap_or("incomplete");
            println!(
                "  | {:>5} | {} | {} | {} |  {}",
                row.log_n, local, global, ratio, verdict
            );
        }
    }

    println!();
    println!("  recommendation: {}", recommendation_label(rec));
}

fn classify_win(ratio: f64) -> &'static str {
    if ratio >= zkgpu_tail_analyze::UNCONDITIONAL_WIN {
        "global-big"
    } else if ratio >= zkgpu_tail_analyze::PER_DEVICE_WIN {
        "global-narrow"
    } else if ratio > -zkgpu_tail_analyze::PER_DEVICE_WIN {
        "neutral"
    } else {
        "local-wins"
    }
}

fn recommendation_label(rec: &Recommendation) -> String {
    match rec {
        Recommendation::Unconditional { threshold_log_n } => {
            format!("UNCONDITIONAL @ log{threshold_log_n} (global ≥20% win)")
        }
        Recommendation::PerDevice { threshold_log_n } => {
            format!("PER-DEVICE @ log{threshold_log_n} (global 5–20% win)")
        }
        Recommendation::WindowedFlip {
            start_log_n,
            end_log_n,
            avg_win,
        } => {
            if start_log_n == end_log_n {
                format!(
                    "WINDOWED-FLIP @ log{start_log_n} ({:.1}% avg win; does not extend to max log_n)",
                    avg_win * 100.0,
                )
            } else {
                format!(
                    "WINDOWED-FLIP @ log{start_log_n}..=log{end_log_n} ({:.1}% avg win)",
                    avg_win * 100.0,
                )
            }
        }
        Recommendation::NoChange => "NO-CHANGE (local still wins)".to_string(),
        Recommendation::InsufficientData => "INSUFFICIENT-DATA".to_string(),
    }
}
