use std::fmt;

#[derive(Debug)]
pub enum ZkGpuError {
    DeviceNotFound,
    DeviceLost(String),
    ShaderCompilation(String),
    GpuValidation(String),
    BufferSize { requested: u64, limit: u64 },
    InvalidNttSize(String),
    /// GPU device exists but cannot execute compute shaders correctly.
    /// Typically caused by broken Vulkan drivers on budget hardware.
    GpuComputeUnsupported(String),
    BackendError(Box<dyn std::error::Error + Send + Sync>),
}

impl fmt::Display for ZkGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeviceNotFound => write!(f, "no compatible GPU device found"),
            Self::DeviceLost(msg) => write!(f, "GPU device lost: {msg}"),
            Self::ShaderCompilation(msg) => write!(f, "shader compilation failed: {msg}"),
            Self::GpuValidation(msg) => write!(f, "GPU validation error: {msg}"),
            Self::BufferSize { requested, limit } => {
                write!(f, "buffer size {requested} exceeds device limit {limit}")
            }
            Self::InvalidNttSize(msg) => write!(f, "invalid NTT size: {msg}"),
            Self::GpuComputeUnsupported(msg) => {
                write!(f, "GPU compute unsupported: {msg}")
            }
            Self::BackendError(e) => write!(f, "backend error: {e}"),
        }
    }
}

impl std::error::Error for ZkGpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::BackendError(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}
