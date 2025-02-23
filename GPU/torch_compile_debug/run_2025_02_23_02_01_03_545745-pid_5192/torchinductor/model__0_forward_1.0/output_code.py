# AOT ID: ['0_forward']
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


# kernel path: /tmp/torchinductor_root/wb/cwbykqxttc3couj6vqtnfppei4lui62sdubucyupqil3747qfz7m.py
# Topologically Sorted Source Nodes: [sin, mul, cos, add], Original ATen: [aten.sin, aten.mul, aten.cos, aten.add]
# Source node to ATen node mapping:
#   add => add
#   cos => cos
#   mul => mul
#   sin => sin
# Graph fragment:
#   %sin : [num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%addmm,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sin, 2), kwargs = {})
#   %cos : [num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%addmm,), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %cos), kwargs = {})
triton_poi_fused_add_cos_mul_sin_0 = async_compile.triton('triton_poi_fused_add_cos_mul_sin_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=80, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=108, warp_size=32), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cos_mul_sin_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'CC5CD2B4144D54CA80B78EEF6F54FBFC888FC6846BD6BCF5764202597E9C3FA5', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cos_mul_sin_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tl_math.sin(tmp0)
    tmp2 = 2.0
    tmp3 = tmp1 * tmp2
    tmp4 = tl_math.cos(tmp0)
    tmp5 = tmp3 + tmp4
    tl.store(out_ptr0 + (x0), tmp5, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3 = args
    args.clear()
    assert_size_stride(primals_1, (1, 1), (1, 1))
    assert_size_stride(primals_2, (1, ), (1, ))
    assert_size_stride(primals_3, (2000, 1), (1, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((2000, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_2, primals_3, primals_1, alpha=1, beta=1, out=buf1)
        del primals_1
        del primals_2
        buf2 = empty_strided_cuda((2000, 1), (1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [sin, mul, cos, add], Original ATen: [aten.sin, aten.mul, aten.cos, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cos_mul_sin_0.run(buf1, buf2, 2000, grid=grid(2000), stream=stream0)
    return (buf2, primals_3, buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((2000, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
