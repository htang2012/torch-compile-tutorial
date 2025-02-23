
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config






isolate_fails_code_str = None



# torch version: 2.6.0a0+df5bbc09d1.nv24.12
# torch cuda version: 12.6
# torch git version: Unknown


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2024 NVIDIA Corporation 
# Built on Tue_Oct_29_23:50:19_PDT_2024 
# Cuda compilation tools, release 12.6, V12.6.85 
# Build cuda_12.6.r12.6/compiler.35059454_0 

# GPU Hardware Info: 
# NVIDIA A100-SXM4-80GB : 8 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, primals_3, addmm, tangents_1):
        sin = torch.ops.aten.sin.default(addmm)
        neg = torch.ops.aten.neg.default(sin);  sin = None
        mul_1 = torch.ops.aten.mul.Tensor(tangents_1, neg);  neg = None
        permute_2 = torch.ops.aten.permute.default(mul_1, [1, 0])
        mm = torch.ops.aten.mm.default(permute_2, primals_3);  permute_2 = None
        permute_3 = torch.ops.aten.permute.default(mm, [1, 0]);  mm = None
        sum_1 = torch.ops.aten.sum.dim_IntList(mul_1, [0], True);  mul_1 = None
        view = torch.ops.aten.view.default(sum_1, [1]);  sum_1 = None
        permute_4 = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mul_2 = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None
        cos = torch.ops.aten.cos.default(addmm);  addmm = None
        mul_3 = torch.ops.aten.mul.Tensor(mul_2, cos);  mul_2 = cos = None
        permute_5 = torch.ops.aten.permute.default(mul_3, [1, 0])
        mm_1 = torch.ops.aten.mm.default(permute_5, primals_3);  permute_5 = primals_3 = None
        permute_6 = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_3, [0], True);  mul_3 = None
        view_1 = torch.ops.aten.view.default(sum_2, [1]);  sum_2 = None
        add_1 = torch.ops.aten.add.Tensor(view, view_1);  view = view_1 = None
        permute_7 = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        add_2 = torch.ops.aten.add.Tensor(permute_4, permute_7);  permute_4 = permute_7 = None
        return (add_2, add_1, None)
        
def load_args(reader):
    buf0 = reader.storage(None, 8000, device=device(type='cuda', index=0))
    reader.tensor(buf0, (2000, 1), is_leaf=True)  # primals_3
    buf1 = reader.storage(None, 8000, device=device(type='cuda', index=0))
    reader.tensor(buf1, (2000, 1), is_leaf=True)  # addmm
    buf2 = reader.storage(None, 8000, device=device(type='cuda', index=0))
    reader.tensor(buf2, (2000, 1), is_leaf=True)  # tangents_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='real', check_str=None)
        # mod(*args)