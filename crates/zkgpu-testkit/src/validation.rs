use serde::{Deserialize, Serialize};
use zkgpu_babybear::BabyBear;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationOutcome {
    pub passed: bool,
    pub mismatch_count: u32,
    pub first_mismatch_index: Option<u32>,
    pub first_mismatch_gpu: Option<String>,
    pub first_mismatch_cpu: Option<String>,
}

pub fn compare_vectors(gpu: &[BabyBear], cpu: &[BabyBear]) -> ValidationOutcome {
    let mut mismatch_count = 0u32;
    let mut first = None;

    for (idx, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
        if g != c {
            mismatch_count += 1;
            if first.is_none() {
                first = Some((idx as u32, g.to_string(), c.to_string()));
            }
        }
    }

    let (first_mismatch_index, first_mismatch_gpu, first_mismatch_cpu) = match first {
        Some((idx, gpu, cpu)) => (Some(idx), Some(gpu), Some(cpu)),
        None => (None, None, None),
    };

    ValidationOutcome {
        passed: mismatch_count == 0,
        mismatch_count,
        first_mismatch_index,
        first_mismatch_gpu,
        first_mismatch_cpu,
    }
}
