
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
    def __init__(self, in_features, out_features, bias = None):
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
    def __init__(self, args):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(args.n_dims, device = args.device))
        self.beta = torch.nn.Parameter(torch.zeros(args.n_dims, device = args.device))
        self.norm_eps = args.norm_eps
    def forward(self, h):
        return TritonLayerNormFn.apply(h, self.gamma, self.beta, self.norm_eps)
                

class TritonFlashAttnFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hq, hk, hv, attn_mask = None):
        # hq.shape = [batch_size, seq_len, n_heads * head_dim]
        O, L = kernels.fa_fwd(hq, hk, hv, attn_mask)
        ctx.save_for_backward(O, L, hq, hk, hv)
        return O

    @staticmethod
    def backward(ctx, dO):
        O, L, hq, hk, hv = ctx.saved_tensors
        dQ, dK, dV = kernels.fa_bwd(hq, hk, hv, O, dO, L)
        return dQ, dK, dV
      
class TritonSelfAttn(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.proj_dim = self.n_heads * self.head_dim         
        self.wq = TritonLinear(args.dim, self.proj_dim, bias = False)
        self.wk = TritonLinear(args.dim, self.proj_dim, bias = False)
        self.wv = TritonLinear(args.dim, self.proj_dim, bias = False)
        self.wo = TritonLinear(self.proj_dim, args.dim, bias = False)
        self.dropout = torch.nn.Dropout(args.dropout_rate)        
    def forward(self, h, attn_mask = None):
        batch_size, seq_len, dim = h.shape
        hq = self.wq(h)
        hk = self.wk(h)
        hv = self.wv(h)        
        hq = hq.view(batch_size, self.n_heads, seq_len, self.head_dim)
        hk = hk.view(batch_size, self.n_heads, seq_len, self.head_dim)
        hv = hv.view(batch_size, self.n_heads, seq_len, self.head_dim)

        # how to apply mask?
        O = TritonFlashAttnFn.apply(hq, hk, hv, attn_mask)
        
        O = self.dropout(O)
        
        # attn_scores = [batch_size, seq_len, head_dim * n_heads = dim]
        O  = O.view(batch_size, seq_len, self.head_dim * self.n_heads)
        O  = O.contiguous() #.type_as(hq)

        # attn_scores.shape = [batch_size, seq_len, dim]
        O = self.wo(O)
        return O