class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "f32[2000, 1]", addmm: "f32[2000, 1]", tangents_1: "f32[2000, 1]"):
         # File: /workspaces/main_torch_compile.py:74 in forward, code: return 2*torch.sin(self.linear(x)) + torch.cos(self.linear(x))
        sin: "f32[2000, 1]" = torch.ops.aten.sin.default(addmm)
        neg: "f32[2000, 1]" = torch.ops.aten.neg.default(sin);  sin = None
        mul_1: "f32[2000, 1]" = torch.ops.aten.mul.Tensor(tangents_1, neg);  neg = None
        permute_2: "f32[1, 2000]" = torch.ops.aten.permute.default(mul_1, [1, 0])
        mm: "f32[1, 1]" = torch.ops.aten.mm.default(permute_2, primals_3);  permute_2 = None
        permute_3: "f32[1, 1]" = torch.ops.aten.permute.default(mm, [1, 0]);  mm = None
        sum_1: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1, [0], True);  mul_1 = None
        view: "f32[1]" = torch.ops.aten.view.default(sum_1, [1]);  sum_1 = None
        permute_4: "f32[1, 1]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        mul_2: "f32[2000, 1]" = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None
        cos: "f32[2000, 1]" = torch.ops.aten.cos.default(addmm);  addmm = None
        mul_3: "f32[2000, 1]" = torch.ops.aten.mul.Tensor(mul_2, cos);  mul_2 = cos = None
        permute_5: "f32[1, 2000]" = torch.ops.aten.permute.default(mul_3, [1, 0])
        mm_1: "f32[1, 1]" = torch.ops.aten.mm.default(permute_5, primals_3);  permute_5 = primals_3 = None
        permute_6: "f32[1, 1]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
        sum_2: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(mul_3, [0], True);  mul_3 = None
        view_1: "f32[1]" = torch.ops.aten.view.default(sum_2, [1]);  sum_2 = None
        
         # File: /workspaces/main_torch_compile.py:74 in forward, code: return 2*torch.sin(self.linear(x)) + torch.cos(self.linear(x))
        add_1: "f32[1]" = torch.ops.aten.add.Tensor(view, view_1);  view = view_1 = None
        
         # File: /workspaces/main_torch_compile.py:74 in forward, code: return 2*torch.sin(self.linear(x)) + torch.cos(self.linear(x))
        permute_7: "f32[1, 1]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        
         # File: /workspaces/main_torch_compile.py:74 in forward, code: return 2*torch.sin(self.linear(x)) + torch.cos(self.linear(x))
        add_2: "f32[1, 1]" = torch.ops.aten.add.Tensor(permute_4, permute_7);  permute_4 = permute_7 = None
        return (add_2, add_1, None)
        