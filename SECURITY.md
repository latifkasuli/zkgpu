# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in zkgpu, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, email: **security@latifkasuli.dev**

You should receive a response within 48 hours. We will work with you to understand the issue and coordinate a fix before any public disclosure.

## Scope

This project provides GPU-accelerated cryptographic primitives. Correctness is critical:

- Field arithmetic must produce identical results to CPU reference implementations
- NTT outputs must match golden test vectors exactly
- Buffer handling must not leak sensitive data across invocations

## Known Considerations

- GPU compute shaders do not guarantee constant-time execution. Do not use this library for operations where timing side channels are a concern (e.g., private key operations).
- WGSL shader compilation and execution are handled by the GPU driver. We cannot guarantee driver-level security.
- This project is pre-1.0 and has not been audited. Use at your own risk in production.
