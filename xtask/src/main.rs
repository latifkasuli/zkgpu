use std::path::Path;
use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("shaders") => compile_shaders(),
        Some("help") | None => print_usage(),
        Some(cmd) => {
            eprintln!("Unknown command: {cmd}");
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("Usage: cargo xtask <command>");
    eprintln!();
    eprintln!("Commands:");
    eprintln!("  shaders   Compile GLSL compute shaders to SPIR-V");
    eprintln!("  help      Show this help message");
}

/// Compile all GLSL compute shaders (.comp) to SPIR-V (.spv).
///
/// Requires `glslangValidator` on PATH (install via system package
/// manager or the Vulkan SDK).
///
/// Targets Vulkan 1.1 for subgroup operations support.
fn compile_shaders() {
    let kernels_dir = Path::new("crates/zkgpu-wgpu/src/kernels/native");

    let glsl_files = [
        "babybear_stockham_local_subgroup.comp",
    ];

    // Verify glslangValidator is available
    let validator = find_glslang();

    let mut failed = false;
    for file in &glsl_files {
        let input = kernels_dir.join(file);
        let output = input.with_extension("spv");

        println!("Compiling {} -> {}", input.display(), output.display());

        let status = Command::new(&validator)
            .args([
                "-V",
                "--target-env", "vulkan1.1",
                "-o", output.to_str().expect("valid UTF-8 path"),
                input.to_str().expect("valid UTF-8 path"),
            ])
            .status()
            .unwrap_or_else(|e| {
                eprintln!("Failed to run glslangValidator: {e}");
                std::process::exit(1);
            });

        if !status.success() {
            eprintln!("FAILED: {}", input.display());
            failed = true;
        }
    }

    if failed {
        eprintln!();
        eprintln!("Some shaders failed to compile.");
        std::process::exit(1);
    }

    println!();
    println!("All shaders compiled successfully.");
}

/// Find `glslangValidator` on PATH or at known locations.
fn find_glslang() -> String {
    // Try PATH first
    if Command::new("glslangValidator")
        .arg("--version")
        .output()
        .is_ok()
    {
        return "glslangValidator".to_string();
    }

    // Homebrew on macOS (Apple Silicon)
    let homebrew = "/opt/homebrew/bin/glslangValidator";
    if Path::new(homebrew).exists() {
        return homebrew.to_string();
    }

    // Homebrew on macOS (Intel)
    let homebrew_intel = "/usr/local/bin/glslangValidator";
    if Path::new(homebrew_intel).exists() {
        return homebrew_intel.to_string();
    }

    eprintln!("Error: glslangValidator not found.");
    eprintln!();
    eprintln!("Install it via one of:");
    eprintln!("  macOS:   brew install glslang");
    eprintln!("  Ubuntu:  apt install glslang-tools");
    eprintln!("  Arch:    pacman -S glslang");
    eprintln!("  Or install the Vulkan SDK: https://vulkan.lunarg.com/sdk/home");
    std::process::exit(1);
}
