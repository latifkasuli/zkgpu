//! Minimal async utilities for bridging wgpu's callback APIs to Rust futures.
//!
//! On native, `device.poll(Wait)` fires callbacks synchronously so the futures
//! resolve immediately. On WebGPU (browser), the event loop drives callbacks
//! and the futures wake the async runtime when the callback fires.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

use zkgpu_core::ZkGpuError;

// ---------------------------------------------------------------------------
// CallbackFuture — single-shot future driven by a callback
// ---------------------------------------------------------------------------

/// A single-shot future that resolves when a wgpu callback sends a value.
///
/// Created in pairs: `CallbackFuture::new()` returns `(sender, future)`.
/// Pass the sender into a wgpu callback closure, then `.await` the future.
pub(crate) struct CallbackFuture<T: Send + 'static> {
    state: Arc<Mutex<CallbackState<T>>>,
}

struct CallbackState<T> {
    value: Option<T>,
    waker: Option<Waker>,
}

/// Sender half of a `CallbackFuture`. Passed into wgpu callback closures.
pub(crate) struct CallbackSender<T: Send + 'static> {
    state: Arc<Mutex<CallbackState<T>>>,
}

impl<T: Send + 'static> CallbackFuture<T> {
    /// Create a new callback future and the corresponding sender.
    pub fn new() -> (CallbackSender<T>, Self) {
        let state = Arc::new(Mutex::new(CallbackState {
            value: None,
            waker: None,
        }));
        (
            CallbackSender {
                state: state.clone(),
            },
            Self { state },
        )
    }
}

impl<T: Send + 'static> CallbackSender<T> {
    /// Complete the associated future with the given value.
    ///
    /// If the future has already been polled, the stored waker is woken
    /// to notify the async runtime.
    pub fn send(self, value: T) {
        let mut guard = self.state.lock().expect("callback future poisoned");
        guard.value = Some(value);
        if let Some(waker) = guard.waker.take() {
            waker.wake();
        }
    }
}

impl<T: Send + 'static> Future for CallbackFuture<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        let mut guard = self.state.lock().expect("callback future poisoned");
        if let Some(value) = guard.value.take() {
            Poll::Ready(value)
        } else {
            guard.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

// ---------------------------------------------------------------------------
// Buffer mapping helper
// ---------------------------------------------------------------------------

/// Asynchronously map a buffer slice for reading.
///
/// On native, calls `device.poll(Wait)` to fire the callback immediately.
/// On WebGPU, yields to the browser event loop.
pub(crate) async fn map_buffer_read(
    device: &wgpu::Device,
    slice: wgpu::BufferSlice<'_>,
) -> Result<(), ZkGpuError> {
    let (sender, future) = CallbackFuture::new();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result);
    });

    #[cfg(not(target_arch = "wasm32"))]
    device.poll(wgpu::Maintain::Wait);

    // Suppress unused variable warning on wasm where poll is not called.
    #[cfg(target_arch = "wasm32")]
    let _ = device;

    future
        .await
        .map_err(|e| ZkGpuError::BackendError(Box::new(e)))
}

// ---------------------------------------------------------------------------
// GPU submission completion helper
// ---------------------------------------------------------------------------

/// Wait for all previously submitted GPU work to complete.
///
/// On native, uses `queue.on_submitted_work_done()` + `device.poll(Wait)`
/// to block until the GPU finishes.
///
/// On WebGPU (browser), this is a no-op — `on_submitted_work_done` is
/// unimplemented in wgpu's WebGPU backend (as of v24). Browser GPU work
/// completes implicitly before `map_async` callbacks fire, so the
/// subsequent buffer readback already provides the synchronization
/// barrier we need.
pub(crate) async fn wait_for_submission(device: &wgpu::Device, queue: &wgpu::Queue) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let (sender, future) = CallbackFuture::new();
        queue.on_submitted_work_done(move || {
            sender.send(());
        });
        device.poll(wgpu::Maintain::Wait);
        future.await;
    }

    // On wasm, submission completion is driven by the browser event loop.
    // Buffer mapping (map_async) already waits for prior GPU work, so
    // no explicit wait is needed here.
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (device, queue);
    }
}
