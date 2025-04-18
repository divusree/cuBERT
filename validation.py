
import torch
from autograd import TritonLinearFn, TritonLayerNormFn, TritonFlashAttnFn
import math

class TorchOps:
    def torch_matmul(self, x, weight, b = None):
        # x.shape = (B, M, K), weight.shape = (B, K, N)  
        # also works with x.shape = (B, H, M, K), weight.shape = (B, H, K, N)
        output = torch.matmul(x, weight)  # Batch matmul  
        if b is not None:  
            output += b.unsqueeze(-2)  # Add bias (B, H, 1, N)  
        return output      
    def layer_norm(self, x, gamma, beta):
        norm =  torch.nn.LayerNorm(x.shape[-1], device = x.device, elementwise_affine = False)
        # Activate module
        op = gamma * norm(x) + beta  
        return op
    def flash_attention(self, hq, hk, hv, attn_mask = None):
        qk_scale = 1 / math.sqrt(hq.shape[-1])
        O = hq @ hk.transpose(-1, -2) # [batch_size, n_heads, seq_len, seq_len]
        O = O * qk_scale

        if attn_mask is not None:
            O = O.masked_fill(attn_mask == 0, float('-inf'))

        # matmul: [batch_size, n_heads, seq_len, seq_len] @  [batch_size, n_heads, seq_len, head_dim]
        # attn_scores = [batch_size, n_heads, seq_len, head_dim]
        O  =  torch.softmax(O, dim = -1)  @ hv 
        return O
    
class Validator:
    def __init__(self):
        self.torch_operators = TorchOps()
    def validate_gradients(self, torch_op, triton_op, input_shapes, dtype=torch.float32, atol=1e-3, rtol=1e-2, n_checks = 5):
        # Generate fresh inputs for each test
        for _ in range(n_checks):

            inputs = [
                torch.randn(shape, dtype=dtype, device='cuda', requires_grad=True)
                for shape in input_shapes
            ]
            
            # Synthetic gradient (match output shape)
            # with torch.no_grad():
            #     dummy_out = torch_op(*inputs)
            #     dO = torch.randn_like(dummy_out, device = 'cuda')
            
            # PyTorch reference
            torch_out = torch_op(*inputs)
            dO = torch.randn_like(torch_out, device = 'cuda', requires_grad = False)
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
                print(f"\nValidation failed for {triton_op}:")
                print(f"Forward diff: {(triton_out - torch_out).abs().max().item():.3e}")
                for i, (t_g, p_g) in enumerate(zip(triton_grads, torch_grads)):
                    if t_g is not None and p_g is not None:
                        print(f"Grad {i} max diff: {(t_g - p_g).abs().max().item():.3e}")
            assert forward_ok, f"Forward pass mismatch for {triton_op }"
            assert backward_ok, f"Backward pass mismatch for {triton_op }"        
        print(f"✓ All {n_checks} gradient checks passed for {triton_op }")

    def check_matmul(self):
        B, H, M, K, N = 8, 7, 15, 12, 13  
        # check for large block sizes as well
        x_shape = (B, M, K)      
        weight_shape = (B, K, N)     
        bias_shape = (B, N)     

        print("Validating matmul3D")
        self.validate_gradients(
            torch_op= self.torch_operators.torch_matmul,
            triton_op= TritonLinearFn.apply,
            input_shapes=[x_shape, weight_shape, bias_shape],  # Shapes for x, weight, bias
            dtype=torch.float32,
            atol=1e-3,
            rtol=1e-2
        )        
        print("matmul3D passed all checks")
        print("Validating matmul4D")

        x_shape = (B, H, M, K)      
        weight_shape = (B, H, K, N)     
        bias_shape = (B, H, N)     

        self.validate_gradients(
            torch_op= self.torch_operators.torch_matmul,
            triton_op= TritonLinearFn.apply,
            input_shapes=[x_shape, weight_shape, bias_shape],  # Shapes for x, weight, bias
            dtype=torch.float32,
            atol=1e-3,
            rtol=1e-2
        )        
        print("matmul4D passed all checks")
    def check_layer_norm(self):
        B, M, N = 8, 32, 1024

        self.validate_gradients(
            torch_op= self.torch_operators.layer_norm,
            triton_op= TritonLayerNormFn.apply,
            input_shapes=[(B, M, N), (N), (N)],  # Shapes for x, weight, bias
            dtype=torch.float32,
            atol=1e-3,
            rtol=1e-2
        )              
    def check_flash_attention(self):
        B, H, M, N = 4, 4, 32, 32
        self.validate_gradients(
            torch_op= self.torch_operators.flash_attention,
            triton_op= TritonFlashAttnFn.apply,
            input_shapes=[(B, H, M, N), (B, H, M, N), (B, H, M, N)],  # Shapes for x, weight, bias
            dtype=torch.float32,
            atol=1e-3,
            rtol=1e-2
        )           
        