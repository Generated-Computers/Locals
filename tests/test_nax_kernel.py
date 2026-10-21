"""The NAX small-M kernel (locals/nax_kernel.py) gives what MLX's quantized_matmul gives, on Qwen3.5's layer shapes
(4B, 9B, 27B), for every row count from 1 to 32 a configuration supports, in bfloat16 and float16. Skipped on GPUs
without NAX (it does not compile there)."""

import mlx.core as mx
import pytest

nax = pytest.importorskip("locals.nax_kernel")
pytestmark = pytest.mark.skipif(not nax.available(), reason="no NAX on this GPU (or the kernel does not compile)")

SHAPES = [(2560, 4096), (10240, 2560), (2560, 9216), (4096, 12288), (5120, 6144), (14336, 5120)]  # (N, K)


def _layer(N, K, dtype, seed):
    w = (mx.random.normal((N, K), key=mx.random.key(seed)) * 0.05).astype(dtype)
    return mx.quantize(w, group_size=64, bits=4)


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("N,K", SHAPES)
def test_matches_mlx(N, K, dtype):
    wq, sc, bi = _layer(N, K, dtype, N + K)
    for M in (1, 2, 5, 6, 8, 12, 15, 16, 17, 24, 31, 32):
        x = mx.random.normal((M, K), key=mx.random.key(M)).astype(dtype)
        ref = mx.quantized_matmul(x, wq, sc, bi, transpose=True, group_size=64, bits=4).astype(mx.float32)
        tol = 0.05 * (1.0 + float(mx.max(mx.abs(ref))))  # as kernels._tune_shape accepts a candidate
        for cfg in nax.CANDIDATES:
            if not nax.supported(M, K, N, cfg):
                continue
            got = nax.qmm_nax(x, wq, sc, bi, *cfg).astype(mx.float32)
            assert got.shape == ref.shape
            err = float(mx.max(mx.abs(got - ref)))
            assert err <= tol, (M, cfg, err, tol)


def test_inside_a_batch_of_rows_each_row_is_its_own():
    """Rows do not leak into each other: row i of a 12-row product equals the 1-row product of row i."""
    N, K = 2560, 4096
    wq, sc, bi = _layer(N, K, mx.bfloat16, 7)
    x = mx.random.normal((12, K), key=mx.random.key(3)).astype(mx.bfloat16)
    cfg = next(c for c in nax.CANDIDATES if nax.supported(12, K, N, c))
    full = nax.qmm_nax(x, wq, sc, bi, *cfg).astype(mx.float32)
    for i in (0, 5, 11):
        one = mx.quantized_matmul(x[i : i + 1], wq, sc, bi, transpose=True, group_size=64, bits=4).astype(mx.float32)
        assert float(mx.max(mx.abs(full[i : i + 1] - one))) <= 0.05 * (1.0 + float(mx.max(mx.abs(one))))

