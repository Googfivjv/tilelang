"""Cross-lane merging writes the compact result and preserves the rest."""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
import tilelang.testing


def make_kernel(name, dtype):
    lanes = 2048 // T.dtype(dtype).bits

    @T.prim_func
    def main(A: T.Tensor((lanes,), dtype), Old: T.Tensor((lanes,), dtype), Out: T.Tensor((lanes,), dtype)):
        with T.Kernel(1):
            a = T.alloc_shared((lanes,), dtype)
            old = T.alloc_shared((lanes,), dtype)
            out = T.alloc_shared((lanes,), dtype)
            T.copy(A, a)
            T.copy(Old, old)
            with T.SimdVF():
                x = T.simd.vld(a[0])
                dst = T.simd.alloc_local(1, dtype)
                dst[0] = T.simd.vld(old[0])
                dst[0] = getattr(T.simd, name)(x, mode="MODE_MERGING")
                T.simd.vsts(out[0], dst[0])
            T.copy(out, Out)

    return main


@pytest.mark.parametrize("name", ["vcpadd", "vcgadd", "vcgmax", "vcgmin"])
def test_cross_lane_merging_uses_software_adapter(name):
    source = tilelang.lower(make_kernel(name, "float32"), target="ascend").kernel_source
    assert f"simd_inst::{name}(*" in source


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("name", ["vcpadd", "vcgadd", "vcgmax", "vcgmin"])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_cross_lane_merging_preserves_nonresult_lanes(name, dtype):
    lanes = 2048 // T.dtype(dtype).bits
    tdtype = getattr(torch, dtype)
    a = torch.arange(lanes, dtype=tdtype)
    old = torch.full((lanes,), -123, dtype=tdtype)
    expected = old.clone()
    if name == "vcpadd":
        expected[: lanes // 2] = a.reshape(-1, 2).sum(dim=1)
    else:
        groups = a.reshape(8, -1)
        if name == "vcgadd":
            expected[:8] = groups.sum(dim=1)
        elif name == "vcgmax":
            expected[:8] = groups.max(dim=1).values
        else:
            expected[:8] = groups.min(dim=1).values
    kernel = tilelang.compile(make_kernel(name, dtype), target="ascend", out_idx=-1)
    actual = kernel(a.npu(), old.npu()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
