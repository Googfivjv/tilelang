"""BF16 math lowers through FP32 before restoring the result dtype."""

import pytest
import tilelang
import tilelang.ascend.language as T


@pytest.mark.parametrize("name", ["exp", "exp2", "log", "sqrt", "tanh", "erf", "sin", "floor"])
def test_bfloat16_math_intrinsic_lowering(name):
    @T.prim_func
    def main(A: T.Tensor((64,), T.bfloat16), B: T.Tensor((64,), T.bfloat16)):
        with T.Kernel(1), T.SimtVF(threads=64):
            for i in T.Parallel(64):
                B[i] = getattr(T, name)(A[i])

    with tilelang.tvm.target.Target("ascend"):
        source = tilelang.lower(main, target="ascend").kernel_source
    assert name + "f(" in source
