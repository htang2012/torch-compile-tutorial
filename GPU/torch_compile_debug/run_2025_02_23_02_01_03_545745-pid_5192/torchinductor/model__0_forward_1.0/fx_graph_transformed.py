class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[1, 1]", primals_2: "f32[1]", primals_3: "f32[2000, 1]"):
         # File: /workspaces/main_torch_compile.py:74 in forward, code: return 2*torch.sin(self.linear(x)) + torch.cos(self.linear(x))
        permute: "f32[1, 1]" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        addmm: "f32[2000, 1]" = torch.ops.aten.addmm.default(primals_2, primals_3, permute);  primals_2 = permute = None
        sin: "f32[2000, 1]" = torch.ops.aten.sin.default(addmm)
        mul: "f32[2000, 1]" = torch.ops.aten.mul.Tensor(sin, 2);  sin = None
        cos: "f32[2000, 1]" = torch.ops.aten.cos.default(addmm)
        add: "f32[2000, 1]" = torch.ops.aten.add.Tensor(mul, cos);  mul = cos = None
        return (add, primals_3, addmm)
        