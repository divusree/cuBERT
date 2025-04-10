
import triton
import torch
import kernels
from typing import Callable, Tuple, Union
          
class TritonLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, W, b = None):
        ctx.save_for_backward(h, W, b)  
        output = kernels.matmul(h, W)    
        if b is not None:
            output += b.unsqueeze(-2)
        return output

    @staticmethod
    def backward(ctx, dO):
        h, W, b = ctx.saved_tensors
        dh = kernels.matmul(dO, W, ytrans = True)
        dW = kernels.matmul(h, dO, xtrans = True)
        dW = dW.reshape(W.shape)   
        # sum over seq dim except feature dims
        db = dO.sum(dim=-2) if b is not None else None
        return dh, dW, db
    
class TritonLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.W = torch.nn.Parameter(torch.empty(out_features, in_features))
        self.b = torch.nn.Parameter(torch.empty(out_features)) if bias else None
        torch.nn.init.kaiming_uniform_(self.W)

    def forward(self, h):
        return TritonLinearFn.apply(h, self.W, self.b)
        
class TritonLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h, gamma, beta, norm_eps = 1e-6):
        o, mean, var = kernels.layer_norm(h, norm_eps) 
        output = gamma * o + beta    
        ctx.save_for_backward(gamma, var, o) 
        return output

    @staticmethod
    def backward(ctx, dO):
        gamma, var, o = ctx.saved_tensors

        rsqrt = torch.rsqrt(var + 1e-6) 
        term1 =  gamma * dO                               
        term2 = (term1 * o).sum(dim=-1, keepdim=True)     
        term3 = term1.sum(dim=-1, keepdim=True)          
        dh = (term1 - (term3 + o * term2) / dO.shape[-1]) * rsqrt 

        dg = (dO * o).sum(dim = -2)
        db = dO.sum(dim = -2)
        return dh, dg, db
    
class TritonLayerNorm(torch.nn.Module):
    def __init__(self, n_dims, norm_eps = 1e-6, device = 'cuda'):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(n_dims, device = device))
        self.beta = torch.nn.Parameter(torch.zeros(n_dims, device = device))
        self.norm_eps = norm_eps
    def forward(self, h):
        return TritonLayerNormFn.apply(h, self.gamma, self.beta, self.norm_eps)
                