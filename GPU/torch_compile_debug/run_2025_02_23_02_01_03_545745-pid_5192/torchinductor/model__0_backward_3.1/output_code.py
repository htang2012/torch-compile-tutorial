# AOT ID: ['0_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_root/x6/cx6pbax3lcolti5ajr4uopo5fvgozqfm3phdbfdgzisw6mv4ahf6.py
# Topologically Sorted Source Nodes: [sin, cos], Original ATen: [aten.sin, aten.neg, aten.mul, aten.sum, aten.cos, aten.add]
# Source node to ATen node mapping:
#   cos => cos
#   sin => sin
# Graph fragment:
#   %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%addmm,), kwargs = {})
#   %neg : [num_users=1] = call_function[target=torch.ops.aten.neg.default](args = (%sin,), kwargs = {})
#   %mul_1 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, %neg), kwargs = {})
#   %sum_1 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_1, [0], True), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%tangents_1, 2), kwargs = {})
#   %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%addmm,), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_2, %cos), kwargs = {})
#   %sum_2 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_3, [0], True), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view, %view_1), kwargs = {})
triton_red_fused_add_cos_mul_neg_sin_sum_0 = async_compile.triton('triton_red_fused_add_cos_mul_neg_sin_sum_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints=[1, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'rnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cos_mul_neg_sin_sum_0', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': 'CC5CD2B4144D54CA80B78EEF6F54FBFC888FC6846BD6BCF5764202597E9C3FA5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cos_mul_neg_sin_sum_0(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 2000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl_math.sin(tmp1)
        tmp3 = -tmp2
        tmp4 = tmp0 * tmp3
        tmp5 = 2.0
        tmp6 = tmp0 * tmp5
        tmp7 = tl_math.cos(tmp1)
        tmp8 = tmp6 * tmp7
        tmp9 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
        tmp12 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp4, rmask)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp8, rmask)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tmp15 = tmp10 + tmp13
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp15, None)
''', device_str='cuda')


# kernel path: /tmp/torchinductor_root/yd/cydnqz7aolhbpuk5nbvsa43kywhl4kfdsbecl5qi2a4eawjksd3k.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
# Source node to ATen node mapping:
# Graph fragment:
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_4, %permute_7), kwargs = {})
triton_poi_fused_add_1 = async_compile.triton('triton_poi_fused_add_1', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108, warp_size=32), 'constants': {'xnumel': 1}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_1', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'CC5CD2B4144D54CA80B78EEF6F54FBFC888FC6846BD6BCF5764202597E9C3FA5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_1(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    tmp0 = tl.load(in_out_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.load(in_ptr0 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp4 = tmp1 + tmp3
    tl.store(in_out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp4, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_3, addmm, tangents_1 = args
    args.clear()
    assert_size_stride(primals_3, (2000, 1), (1, 1))
    assert_size_stride(addmm, (2000, 1), (1, 1))
    assert_size_stride(tangents_1, (2000, 1), (1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((2000, 1), (1, 1), torch.float32)
        buf3 = empty_strided_cuda((2000, 1), (1, 1), torch.float32)
        buf2 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        buf6 = reinterpret_tensor(buf2, (1, ), (1, ), 0); del buf2  # reuse
        # Topologically Sorted Source Nodes: [sin, cos], Original ATen: [aten.sin, aten.neg, aten.mul, aten.sum, aten.cos, aten.add]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_cos_mul_neg_sin_sum_0.run(buf6, tangents_1, addmm, buf0, buf3, 1, 2000, grid=grid(1), stream=stream0)
        del addmm
        del tangents_1
        buf1 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (1, 2000), (0, 1), 0), primals_3, out=buf1)
        del buf0
        buf4 = empty_strided_cuda((1, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (1, 2000), (0, 1), 0), primals_3, out=buf4)
        del buf3
        del primals_3
        buf7 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.add]
        triton_poi_fused_add_1.run(buf7, buf4, 1, grid=grid(1), stream=stream0)
    return (buf7, buf6, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_3 = rand_strided((2000, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    addmm = rand_strided((2000, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    tangents_1 = rand_strided((2000, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_3, addmm, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
