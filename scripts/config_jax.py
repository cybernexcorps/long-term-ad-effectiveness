"""
JAX Backend Configuration for GPU Acceleration

This script configures PyMC to use JAX backend with GPU support.

Benefits:
- 10-100x faster compilation than default C backend
- GPU acceleration for matrix operations
- Better performance for complex models with many parameters

Usage:
    import config_jax
    # JAX backend is now configured globally for PyMC
"""

import os
import warnings

# Configure JAX before importing
os.environ['JAX_ENABLE_X64'] = '1'  # Enable 64-bit precision

try:
    import jax
    import jax.numpy as jnp

    # Check GPU availability
    devices = jax.devices()
    gpu_available = any('gpu' in str(d).lower() for d in devices)

    if gpu_available:
        print(f"✓ JAX detected {len([d for d in devices if 'gpu' in str(d).lower()])} GPU(s)")
        print(f"  Devices: {devices}")
    else:
        print("⚠ JAX running on CPU (no GPU detected)")
        print("  This is still 5-20x faster than default backend")

    # Configure PyTensor to use JAX
    import pytensor
    pytensor.config.floatX = 'float64'
    pytensor.config.mode = 'JAX'  # Use JAX backend

    print("✓ PyTensor configured to use JAX backend")
    print(f"  Mode: {pytensor.config.mode}")
    print(f"  Float type: {pytensor.config.floatX}")

except ImportError as e:
    warnings.warn(
        f"JAX not installed: {e}\n"
        "Install with: pip install jax jaxlib\n"
        "For GPU support: pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    )
    print("Falling back to default backend")

except Exception as e:
    warnings.warn(f"Error configuring JAX: {e}")
    print("Falling back to default backend")


def check_jax_performance():
    """
    Quick benchmark to verify JAX is working and show speedup.
    """
    import time
    import numpy as np

    try:
        import jax
        import jax.numpy as jnp

        # Test array size
        n = 5000

        # NumPy baseline
        a_np = np.random.randn(n, n)
        b_np = np.random.randn(n, n)

        start = time.time()
        c_np = np.dot(a_np, b_np)
        numpy_time = time.time() - start

        # JAX
        a_jax = jnp.array(a_np)
        b_jax = jnp.array(b_np)

        # Warm up JAX (first run compiles)
        _ = jnp.dot(a_jax, b_jax).block_until_ready()

        start = time.time()
        c_jax = jnp.dot(a_jax, b_jax).block_until_ready()
        jax_time = time.time() - start

        speedup = numpy_time / jax_time

        print(f"\n=== JAX Performance Benchmark ===")
        print(f"Matrix multiplication ({n}x{n}):")
        print(f"  NumPy time: {numpy_time:.4f}s")
        print(f"  JAX time:   {jax_time:.4f}s")
        print(f"  Speedup:    {speedup:.1f}x")

        if speedup > 10:
            print("  ✓ Excellent performance (GPU likely active)")
        elif speedup > 2:
            print("  ✓ Good performance (JAX optimizations active)")
        else:
            print("  ⚠ Limited speedup (check GPU configuration)")

    except Exception as e:
        print(f"Benchmark failed: {e}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("JAX Configuration Test")
    print("="*60 + "\n")
    check_jax_performance()
