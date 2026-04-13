use zkgpu_core::{GpuDevice, GpuField, NttDirection, NttPlan, ZkGpuError};

/// CPU reference NTT implementation for testing and golden vector generation.
///
/// Textbook Cooley-Tukey radix-2 DIT. Not optimized — exists only
/// to validate GPU results against.
pub fn ntt_cpu_reference<F: GpuField>(data: &mut [F], direction: NttDirection) {
    let n = data.len();
    assert!(n.is_power_of_two(), "NTT size must be a power of two");
    let log_n = n.trailing_zeros();

    let omega = F::root_of_unity(log_n);
    let omega = match direction {
        NttDirection::Forward => omega,
        NttDirection::Inverse => omega.inverse().expect("root of unity must be invertible"),
    };

    // Bit-reversal permutation
    for i in 0..n {
        let j = bit_reverse(i as u32, log_n) as usize;
        if i < j {
            data.swap(i, j);
        }
    }

    // Cooley-Tukey butterfly stages
    for s in 0..log_n {
        let m = 1usize << (s + 1);
        let half_m = m >> 1;
        let step_root = omega.pow(n as u64 / m as u64);

        let mut k = 0;
        while k < n {
            let mut w = F::ONE;
            for j in 0..half_m {
                let u = data[k + j];
                let t = w.mul(data[k + j + half_m]);
                data[k + j] = u.add(t);
                data[k + j + half_m] = u.sub(t);
                w = w.mul(step_root);
            }
            k += m;
        }
    }

    // For inverse NTT, divide by n
    if direction == NttDirection::Inverse {
        let n_field = F::from_repr(bytemuck::cast(n as u32));
        let n_inv = n_field
            .inverse()
            .expect("n must be invertible in the field");
        for elem in data.iter_mut() {
            *elem = elem.mul(n_inv);
        }
    }
}

/// Plan and execute an NTT on the GPU.
pub fn gpu_ntt<F, D, P>(device: &D, buf: &mut D::Buffer<F>, plan: &mut P) -> Result<(), ZkGpuError>
where
    F: GpuField,
    D: GpuDevice,
    P: NttPlan<F, D>,
{
    plan.execute(device, buf)
}

fn bit_reverse(x: u32, bits: u32) -> u32 {
    x.reverse_bits() >> (32 - bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reverse_smoke() {
        assert_eq!(bit_reverse(0b000, 3), 0b000);
        assert_eq!(bit_reverse(0b001, 3), 0b100);
        assert_eq!(bit_reverse(0b010, 3), 0b010);
        assert_eq!(bit_reverse(0b011, 3), 0b110);
        assert_eq!(bit_reverse(0b100, 3), 0b001);
    }
}
