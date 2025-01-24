import argparse
import math
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import torch.optim as optim
from functorch.compile import make_boxed_func
from torch._functorch.aot_autograd import aot_module_simplified
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch._decomp import core_aten_decompositions
import habana_frameworks.torch.core as htcore
import argparse

torch._dynamo.reset()

decompositions = core_aten_decompositions()
decompositions.update(
    torch._decomp.get_decompositions([
        torch.ops.aten.sin,
        torch.ops.aten.cos,
        torch.ops.aten.add,
        torch.ops.aten.sub,
        torch.ops.aten.mul,
        torch.ops.aten.sum,
        torch.ops.aten.mean,
        torch.ops.aten.pow.Tensor_Scalar,
        torch.ops.aten.convolution,
        torch.ops.aten.conv2d,
        torch.ops.aten.relu,
        torch.ops.aten.linear,
        torch.ops.aten.max_pool2d,
        torch.ops.aten.log_softmax,
        torch.ops.aten.amax
    ])
)


def inspect_backend(gm, sample_inputs,decompositions): 
    # Forward compiler capture
    def fw(gm, sample_inputs):
        gm.print_readable()
        g = FxGraphDrawer(gm, 'fn')
        with open("forward_aot.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())
        return make_boxed_func(gm.forward)
    
    # Backward compiler capture
    def bw(gm, sample_inputs):
        gm.print_readable()
        g = FxGraphDrawer(gm, 'fn')
        with open("backward_aot.svg", "wb") as file:
            file.write(g.get_dot_graph().create_svg())
        return make_boxed_func(gm.forward)
    
    # Call AOTAutograd
    gm_forward = aot_module_simplified(gm,sample_inputs,
                                       fw_compiler=fw,
                                       bw_compiler=bw,
                                       decompositions=decompositions,
                                       )

    return gm_forward


# Create a simple model 
class TrigModel(nn.Module):
    def __init__(self):
        super(TrigModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return 2*torch.sin(self.linear(x)) + torch.cos(self.linear(x))


        
def main():
    
    # Set environment variables
    os.environ['PT_HPU_LAZY_MODE'] = "0"
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    os.environ["TORCH_LOGS"] = "+inductor,dynamo"
    parser = argparse.ArgumentParser(description='Train a TrigModel with custom parameters.')
    parser.add_argument('--torch_compile_type', type=str, choices=['inductor', 'atenIR', 'primsIR', 'hpubackend'], default='inductor', help='Backend type for torch compile.')
    parser.add_argument('--device', type=str, default='hpu', help='Device to run the training on')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    args = parser.parse_args()
    
    if (args.device in ['cpu', 'gpu']) and args.torch_compile_type == 'hpubackend':
        raise ValueError("The device 'cpu' or 'gpu' and torch_compile_type 'hpubackend' are mutually exclusive.")

    device = torch.device(args.device)
    model = TrigModel().to(args.device)
    
   
    # Define the backend mapping
    backend_mapping = {
        'inductor': 'inductor',
        'hpubackend': "hpu_backend",
        'atenIR': lambda gm, sample_inputs: inspect_backend(gm, sample_inputs, decompositions),
        'primsIR': lambda gm, sample_inputs: inspect_backend(gm, sample_inputs, core_aten_decompositions())
    }

    backend_callable = backend_mapping[args.torch_compile_type]
    model = torch.compile(model, backend=backend_callable)
        
    loss_fn = nn.MSELoss()

    # Create some sample data
    x = torch.linspace(-math.pi, math.pi, 2000).unsqueeze(1)
    y = 2 * torch.sin(x) + torch.cos(x) + 0.01 * torch.randn(x.size())

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(1000):
        # Forward pass
        y_pred = model(x.to(args.device))
        loss = loss_fn(y_pred, y.to(args.device))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    main()
