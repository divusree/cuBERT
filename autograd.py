
import triton
import torch
import kernels
from typing import Callable, Tuple, Union

def validate_gradients(torch_op, triton_op, input_shapes, dtype=torch.float32, atol=1e-3, rtol=1e-2, n_checks = 5):
    # Generate fresh inputs for each test
    for _ in range(n_checks):

        inputs = [
            torch.randn(shape, dtype=dtype, device='cuda', requires_grad=True)
            for shape in input_shapes
        ]
        
        # Synthetic gradient (match output shape)
        with torch.no_grad():
            dummy_out = torch_op(*inputs)
            dO = torch.randn_like(dummy_out)
        
        # PyTorch reference
        torch_out = torch_op(*inputs)
        torch_out.backward(dO, retain_graph=True)
        torch_grads = [t.grad.clone() for t in inputs]
        
        # Reset gradients before Triton pass
        for t in inputs:
            t.grad.zero_()
        
        # Triton pass
        triton_out = triton_op(*inputs)
        triton_out.backward(dO)
        triton_grads = [t.grad.clone() for t in inputs]
        
        # Validation
        forward_ok = torch.allclose(triton_out, torch_out, atol=atol, rtol=rtol)
        backward_ok = all(
            torch.allclose(t_grad, p_grad, atol=atol, rtol=rtol)
            for t_grad, p_grad in zip(triton_grads, torch_grads)
            if t_grad is not None and p_grad is not None
        )
        
        if not forward_ok or not backward_ok:
            print("\nValidation failed:")
            print(f"Forward diff: {(triton_out - torch_out).abs().max().item():.3e}")
            for i, (t_g, p_g) in enumerate(zip(triton_grads, torch_grads)):
                if t_g is not None and p_g is not None:
                    print(f"Grad {i} max diff: {(t_g - p_g).abs().max().item():.3e}")
        assert forward_ok, "Forward pass mismatch"
        assert backward_ok, "Backward pass mismatch"        
    print(f"✓ All {n_checks} gradient checks passed")

class TritonLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, b = None):
        ctx.save_for_backward(h, W, b)
        # print("forward")
        # print("h.shape", h.shape)
        # print("W.shape", W.shape)
        # print("b.shape", b.shape)             
        output = kernels.matmul(h, W)    
        # print("output.shape", output.shape)        
        if b is not None:
            output = output + b.unsqueeze(1)
        return output

    @staticmethod
    def backward(ctx, dO):
        h, W, b = ctx.saved_tensors
        # print("dO.shape", dO.shape)
        # print("h.shape", h.shape)
        # print("W.shape", W.shape)
        # print("b.shape", b.shape)
        dh = kernels.matmul(dO, W, ytrans = True)
        dW = kernels.matmul(h, dO, xtrans = True)
        dW = dW.reshape(W.shape)   
        # print("dh.shape", dh.shape)
        # print("dW.shape", dW.shape)

        db = dO.sum(dim=1) if b is not None else None
        # print("db.shape", db.shape)

        return dh, dW, db
    
class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.b = torch.nn.Parameter(torch.empty(out_features)) if bias else None
        torch.nn.init.kaiming_uniform_(self.W)

    def forward(self, h):
        return TritonLinearFn.apply(h, self.W, self.b)
        