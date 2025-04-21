import triton
import triton.language as tl
import torch
import math
torch.set_printoptions(profile="full")

@triton.jit
def batched_addition_kernel(
        x_ptr, #ptr to first idx of x, just like in cpp
        y_ptr,
        output_ptr,
        B, H, M, N, 
        stride_xb,stride_xh, stride_xm, stride_xn,
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        GROUP_SIZE_M: tl.constexpr 
        ):
    pid_batch = tl.program_id(axis=0) 
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    pid_b = pid_batch // H
    pid_h = pid_batch % H

    offset_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset =  pid_b * stride_xb + pid_h * stride_xh + offset_xm[:, None] * stride_xm + offset_xn[None, :] * stride_xn
    mask = (offset_xm[:, None] < M) & (offset_xn[None, :] < N)
    x = tl.load(x_ptr + offset, mask = mask, other = 0.0)
    y = tl.load(y_ptr + offset, mask = mask, other = 0.0)
    tl.store(output_ptr + offset, x + y, mask = mask)

def add4D(x:torch.Tensor, y:torch.Tensor):
    # preallocate output
    assert x.shape == y.shape
    output = torch.empty_like(x, device = x.device, dtype = x.dtype)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    B, H, M, N = output.shape

    grid = lambda meta: (B*H, triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N'])) 
    batched_addition_kernel[grid](x, y, output, 
                                  B, H, M, N, 
                                  *x.stride(),
                                  BLOCK_SIZE_M = 32, BLOCK_SIZE_N = 32,
                                  GROUP_SIZE_M = 8)
    return output    

@triton.jit
def addition3D_kernel(
        x_ptr, #ptr to first idx of x, just like in cpp
        y_ptr,
        output_ptr,
        B, M, N, 
        stride_xb,stride_xm, stride_xn,
        BLOCK_SIZE_M: tl.constexpr, 
        BLOCK_SIZE_N: tl.constexpr, 
        GROUP_SIZE_M: tl.constexpr 
        ):
    pid_b = tl.program_id(axis=0) 
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    offset_xm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_xn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset =  pid_b * stride_xb + offset_xm[:, None] * stride_xm + offset_xn[None, :] * stride_xn
    mask = (offset_xm[:, None] < M) & (offset_xn[None, :] < N)
    x = tl.load(x_ptr + offset, mask = mask, other = 0.0)
    y = tl.load(y_ptr + offset, mask = mask, other = 0.0)
    tl.store(output_ptr + offset, x + y, mask = mask)
def add3D(x:torch.Tensor, y:torch.Tensor):
    # preallocate output
    assert x.shape == y.shape
    output = torch.empty_like(x, device = x.device, dtype = x.dtype)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    B, M, N = output.shape

    grid = lambda meta: (B, triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N'])) 
    addition3D_kernel[grid](x, y, output, 
                                  B, M, N, 
                                  *x.stride(),
                                  BLOCK_SIZE_M = 32, BLOCK_SIZE_N = 32,
                                  GROUP_SIZE_M = 8)
    return output   

@triton.jit
def matmul4D_kernel(
                # my pointers
                x_ptr,y_ptr,output_ptr,
                # matrix dims
                B, H, M, K, N,
                # strides for each matrix
                stride_xb, stride_xh, stride_xm, stride_xk,
                stride_yb, stride_yh, stride_yk, stride_yn,
                stride_ob, stride_oh, stride_om, stride_on,
                BLOCK_SIZE_M: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr, 
                BLOCK_SIZE_K: tl.constexpr,
                GROUP_SIZE_M: tl.constexpr):

    pid_batch = tl.program_id(axis=0) 
    pid_m = tl.program_id(axis=1)
    # pid_n = tl.program_id(axis=2)

    pid_b = pid_batch // H
    pid_h = pid_batch % H

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_m // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_m % num_pid_in_group) % group_size_m)
    pid_n = (pid_m % num_pid_in_group) // group_size_m

    # strides for the M, K and N dim. 
    # A[b, i, j] = b * stride_Ab  + i *stride_Am + j * stride_An
    # &A[b, m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + A.stride(0) + (m : m+BLOCK_SIZE_M)*A.stride(1) + (k : k+BLOCK_SIZE_K)*A.stride(2);
    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offset_k =  tl.arange(0, BLOCK_SIZE_K)
    offset_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))

    x_ptr += pid_b * stride_xb + pid_h * stride_xh + offset_xm[:,None] * stride_xm +  offset_k[None,:]*stride_xk
    y_ptr += pid_b * stride_yb + pid_h * stride_yh + offset_k[:,None] * stride_yk + offset_yn[None,:] *stride_yn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptr, mask= offset_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        y = tl.load(y_ptr, mask=offset_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)       
        accumulator += tl.dot(x, y, input_precision = 'ieee')
        
        # advance to next k block 
        x_ptr += BLOCK_SIZE_K *  stride_xk
        y_ptr += BLOCK_SIZE_K *  stride_yk
        
    offset_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_offset = pid_b * stride_ob + pid_h * stride_oh + offset_om[:,None] *stride_om + offset_on[None, :] *stride_on
    output_mask =  (offset_om[:,None] < M) & ( offset_on[None, :] < N)
    tl.store(output_ptr + output_offset, accumulator, mask = output_mask)

def matmul4D(x: torch.Tensor, y: torch.Tensor, xtrans = False, ytrans = False):
    # Check dimensions
    B, H, M, K = x.shape
    B1, H1, L, N = y.shape
    assert (B == B1) and (H == H1)
    # Allocate output
    output = torch.empty((B, H, M, N), device='cuda', dtype=y.dtype)
    BLOCK_SIZE_M = 64 if M >= 64 else triton.next_power_of_2(M)
    BLOCK_SIZE_N = 64 if N >= 64 else triton.next_power_of_2(N)
    BLOCK_SIZE_K = 32  # Keep K block size smaller for register pressure
    
    stride_xb,stride_xh, stride_xm, stride_xk = x.stride()
    stride_yb, stride_yh, stride_yk, stride_yn = y.stride()
    if xtrans:
        assert M == L 
        stride_xm, stride_xk = stride_xk, stride_xm
        output = torch.empty((B, H, K, N), device = x.device, dtype = x.dtype)
        M,K = K, M # check this
    elif ytrans:
        assert N == K
        stride_yk, stride_yn = stride_yn, stride_yk
        output = torch.empty((B, H, M, L), device = x.device, dtype = x.dtype)
        N, L = L, N # check this
    elif (not xtrans) and (not ytrans):
        assert K == L  
    else:
        raise NotImplementedError("x.T @ y.T is not implemented")
    grid = lambda _: (B * H, triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N))
    matmul4D_kernel[grid](
        x, y, output,
        B, H, M, K, N,
        stride_xb, stride_xh, stride_xm, stride_xk,
        stride_yb, stride_yh, stride_yk, stride_yn,
        *output.stride(),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K, GROUP_SIZE_M=8
    )
    return output

@triton.jit
def matmul3D_kernel(
                # my pointers
                x_ptr,y_ptr,output_ptr,
                # matrix dims
                B, M, K, N,
                # strides for each matrix
                stride_xb, stride_xm, stride_xk,
                stride_yb, stride_yk, stride_yn,
                stride_ob, stride_om, stride_on,
                # BLOCK_SIZE for all dims - for simplicity
                BLOCK_SIZE_B: tl.constexpr, 
                BLOCK_SIZE_M: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr, 
                BLOCK_SIZE_K: tl.constexpr, 
                # GROUP_SIZE for all dims - for simplicity
                GROUP_SIZE_M: tl.constexpr):

    pid_b = tl.program_id(axis=0) 
    pid_m = tl.program_id(axis=1)
    pid = pid_m

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # strides for the M, K and N dim. 
    # A[b, i, j] = b * stride_Ab  + i *stride_Am + j * stride_An
    # &A[b, m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + A.stride(0) + (m : m+BLOCK_SIZE_M)*A.stride(1) + (k : k+BLOCK_SIZE_K)*A.stride(2);
    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M 
    offset_k =  tl.arange(0, BLOCK_SIZE_K)
    offset_yn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N

    x_ptr += pid_b * stride_xb  + offset_xm[:,None] * stride_xm +  offset_k[None,:]*stride_xk
    y_ptr += pid_b * stride_yb  + offset_k[:,None] * stride_yk + offset_yn[None,:] *stride_yn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptr, mask= offset_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        y = tl.load(y_ptr, mask=offset_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)       
        accumulator += tl.dot(x, y, input_precision = 'ieee')
        
        # advance to next k block 
        x_ptr += BLOCK_SIZE_K *  stride_xk
        y_ptr += BLOCK_SIZE_K *  stride_yk
        
    offset_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    output_offset = pid_b * stride_ob  + offset_om[:,None] *stride_om + offset_on[None, :] *stride_on
    output_mask =  (offset_om[:,None] < M) & ( offset_on[None, :] < N)
    tl.store(output_ptr + output_offset, accumulator, mask = output_mask)
    
def matmul3D(x: torch.Tensor, y: torch.Tensor, xtrans = False, ytrans = False):
    B, M , K = x.shape
    B1, L, N = y.shape
    assert (B == B1)
    stride_xb, stride_xm, stride_xk = x.stride()
    stride_yb, stride_yk, stride_yn = y.stride()
    output = torch.empty((B, M, N), device = x.device, dtype = x.dtype)
    if xtrans:
        assert M == L 
        stride_xm, stride_xk = stride_xk, stride_xm
        output = torch.empty((B, K, N), device = x.device, dtype = x.dtype)
        M, K = K, M # check this
    elif ytrans:
        assert N == K
        stride_yk, stride_yn = stride_yn, stride_yk
        output = torch.empty((B, M, L), device = x.device, dtype = x.dtype)
        N, L = L, N # check this
    elif (not xtrans) and (not ytrans):
        assert K == L  
    else:
        raise NotImplementedError("x.T @ y.T is not implemented")
    assert x.is_cuda and y.is_cuda and output.is_cuda
    BLOCK_SIZE_M = 64 # min(max(16, M), 64)
    BLOCK_SIZE_N = 64 # min(max(16, N), 64)
    BLOCK_SIZE_K = 32  # Keep K block size smaller for register pressure
    grid = lambda meta: (B, triton.cdiv(M, meta['BLOCK_SIZE_M'])*triton.cdiv(N, meta['BLOCK_SIZE_N']))
    matmul3D_kernel[grid](x, y, output, 
                        B, M, K, N, 
                        stride_xb, stride_xm, stride_xk,
                        stride_yb, stride_yk, stride_yn,
                        *output.stride(),
                        BLOCK_SIZE_B = 1, BLOCK_SIZE_M = BLOCK_SIZE_M, BLOCK_SIZE_N = BLOCK_SIZE_N, BLOCK_SIZE_K = BLOCK_SIZE_K, GROUP_SIZE_M = 8
                        )
    return output
def matmul(x,y, xtrans = False, ytrans = False):
    if (len(x.shape) == 3) and (len(y.shape) == 3):
        return matmul3D(x, y, xtrans, ytrans)
    elif (len(x.shape) == 4) and (len(y.shape) == 4):
        return matmul4D(x, y, xtrans, ytrans)
    elif (len(x.shape) == 2) and (len(y.shape) == 2):
        return matmul3D(x.unsqueeze(0), y.unsqueeze(0), xtrans, ytrans).squeeze(0)
    elif (len(x.shape) == 3) and (len(y.shape) == 2):
        return matmul3D(x, y.unsqueeze(0), xtrans, ytrans).squeeze(0)
    elif (len(x.shape) == 2) and (len(y.shape) == 3):
        return matmul3D(x.unsqueeze(0), y, xtrans, ytrans).squeeze(0)
    else:
        raise NotImplementedError(f"x.shape and y.shape incompatible {x.shape, y.shape}")        


@triton.jit
def fa_kernel(q_ptr, k_ptr, v_ptr, output_ptr, L_ptr, mask_ptr, qk_scale,
                B: tl.constexpr, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                stride_qb, stride_qh, stride_qm, stride_qn, # all of the matrices have the same stride
                stride_ob, stride_oh, stride_om, stride_on, 
                stride_Lb, stride_Lh, stride_Lm, stride_Ln, 
                BLOCK_SIZE_M: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr,
                GROUP_SIZE_M: tl.constexpr 
            ):
    pid_batch = tl.program_id(axis=0) 
    pid_row = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)

    pid_b = pid_batch // H
    pid_h = pid_batch % H

    offset_row_block = (pid_row * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_col = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % BLOCK_SIZE_N
    
    offset = pid_b * stride_qb + pid_h * stride_qh + offset_row_block[:,None] * stride_qm + offset_col[None,:] *stride_qn
    q_ptr += offset
    mask = (offset_row_block[:,None] < M) & (offset_col[None, :] < N)
    Q = tl.load(q_ptr, mask = mask, other = 0)
    k_ptr += offset 
    v_ptr += offset
    m = tl.full((BLOCK_SIZE_M, 1), value = -torch.inf, dtype = tl.float32) # axis = 1 gives me row wise values
    l = tl.zeros((BLOCK_SIZE_M, 1), dtype = tl.float32)    
    O = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)

    for j in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        K = tl.load(k_ptr, mask = (offset_row_block[:,None] < M - j *BLOCK_SIZE_M), other = 0) # has to be transposed - check if .T is fine or if stride does the trick
        V = tl.load(v_ptr, mask = (offset_row_block[:,None] < M - j *BLOCK_SIZE_M), other = 0) 

        S = tl.dot(Q,K.trans(1,0), input_precision = 'ieee') * qk_scale
        # mask_val = tl.load(mask_ptr + mask_offset)
        # output_val = tl.where(mask_val != 0, fill_value, x_val)
        prev_m = m
        m = tl.maximum(tl.max(S, axis = -1, keep_dims = True), prev_m)
        P = tl.exp(S - m)
        corr = tl.exp(prev_m - m)
        l = corr * l + tl.sum(P, axis = -1, keep_dims = True) 
        O = (corr) * O + tl.dot(P, V, input_precision = 'ieee')
        k_ptr += BLOCK_SIZE_M * stride_qm
        v_ptr += BLOCK_SIZE_M * stride_qm

    O = (1/l) * O
    L = m + tl.log(l)
    output_offset =  pid_b * stride_ob + pid_h * stride_oh + offset_row_block[:,None] * stride_om + offset_col[None,:] *stride_on
    tl.store(output_ptr + output_offset, O) #, mask = (offset_row_block[:,None] < M) & ( offset_col[None,:] < N))
    L_offset =  pid_b * stride_Lb + pid_h * stride_Lh + offset_row_block[:,None] * stride_Lm #+ offset_col[None,:] *stride_Ln
    tl.store(L_ptr + L_offset, L)


def fa_fwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: torch.Tensor):
    B, H, M, N = Q.shape
    output = torch.zeros_like(Q, device = 'cuda', dtype = torch.float32)
    L = torch.zeros((B, H, M, 1), device = 'cuda', dtype = torch.float32)
    qk_scale = 1/math.sqrt(N)
    assert Q.is_cuda and K.is_cuda and V.is_cuda 
    grid = lambda meta: (B*H, triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    fa_kernel[grid](Q, K, V, output, L, attn_mask, qk_scale,
                    B, H, M, N,   
                    *Q.stride(), 
                    *output.stride(), 
                    *L.stride(), 
                    BLOCK_SIZE_M = max(16, triton.next_power_of_2(M)),
                    BLOCK_SIZE_N = max(16, triton.next_power_of_2(N)),
                    GROUP_SIZE_M = 8)   
  
    return output, L


@triton.jit
def fa_bwd_kernel(q_ptr, k_ptr, v_ptr,
                        o_ptr, dO_ptr, L_ptr, qk_scale,
                        dQ_ptr, dK_ptr, dV_ptr, D_ptr, 
                        B: tl.constexpr , H: tl.constexpr, M: tl.constexpr , N: tl.constexpr, 
                        stride_qb, stride_qh, stride_qm, stride_qn, # all of the matrices have the same stride
                        stride_lb, stride_lh, stride_lm, stride_ln, 
                        BLOCK_SIZE_M: tl.constexpr, 
                        BLOCK_SIZE_N: tl.constexpr 
                        ):


    pid_batch = tl.program_id(axis=0) 
    pid_row = tl.program_id(axis = 1)
    pid_col = 0

    pid_b = pid_batch // H
    pid_h = pid_batch % H

    offset_kv_row_block = (pid_row * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) #% M
    offset_kv_col_block = (pid_col * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) #% BLOCK_SIZE_N # check the modulo

    offset =  pid_b * stride_qb + pid_h * stride_qh + offset_kv_row_block[:,None] * stride_qm + offset_kv_col_block[None,:] *stride_qn
    v_ptr += offset
    k_ptr += offset
    kv_mask = (offset_kv_row_block[:,None] < M) & (offset_kv_col_block[None, :] < N)
    K_j = tl.load(k_ptr, mask = kv_mask, other = 0)
    V_j = tl.load(v_ptr, mask = kv_mask, other = 0)

    offset_row_block = offset_kv_col_block # tl.arange(0, BLOCK_SIZE_M) #% M
    offset_col_block = offset_kv_col_block #tl.arange(0, BLOCK_SIZE_N) #% BLOCK_SIZE_N # check the modulo

    q_ptr += pid_b * stride_qb + pid_h * stride_qh + offset_row_block[:,None] * stride_qm + offset_col_block[None,:] *stride_qn
    o_ptr += pid_b * stride_qb + pid_h * stride_qh + offset_row_block[:,None] * stride_qm + offset_col_block[None,:] *stride_qn
    dO_ptr += pid_b * stride_qb + pid_h * stride_qh + offset_row_block[:,None] * stride_qm + offset_col_block[None,:] *stride_qn
    dQ_ptr += pid_b * stride_qb + pid_h * stride_qh + offset_row_block[:,None] * stride_qm + offset_col_block[None,:] *stride_qn
    q_mask  = (offset_row_block[:,None] < M) & (offset_col_block[None, :] < N)

    L_ptr += pid_b * stride_lb + pid_h * stride_lh + offset_row_block[:,None] * stride_lm 
    D_ptr += pid_b * stride_lb + pid_h * stride_lh + offset_row_block[:,None] * stride_lm 
    L_mask =(offset_row_block[:,None] < M)  

    dV_j = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)
    dK_j = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype = tl.float32)

    for i in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        Q_i = tl.load(q_ptr, mask = q_mask, other = 0) # check mask
        dO_i = tl.load(dO_ptr, mask = q_mask, other = 0) # check mask
        dQ_i = tl.load(dQ_ptr, mask = q_mask, other = 0) #.to(tl.float32) # check mask
        L_i = tl.load(L_ptr, mask = L_mask, other = 0) 
        D_i = tl.load(D_ptr, mask = L_mask, other = 0) 

        # back pass
        S_i = tl.dot(Q_i, K_j.trans(1,0), input_precision = "ieee") * qk_scale
        P_i = tl.exp(S_i - L_i)
        dV_j += tl.dot(P_i.trans(1,0), dO_i, input_precision = "ieee")      
        dP_i = tl.dot(dO_i, V_j.trans(1,0), input_precision = "ieee")        
        dS_i = P_i * (dP_i - D_i)
        dQ_i += tl.dot(dS_i, K_j, input_precision = "ieee") # write back to HBM
        tl.store(dQ_ptr, dQ_i, mask = q_mask)
        dK_j += tl.dot(dS_i.trans(1,0), Q_i, input_precision = "ieee")

        # CHECK THIS
        q_ptr += BLOCK_SIZE_M * stride_qm
        dO_ptr += BLOCK_SIZE_M * stride_qm
        dQ_ptr += BLOCK_SIZE_M * stride_qm
        L_ptr += BLOCK_SIZE_M * stride_lm
        D_ptr += BLOCK_SIZE_M * stride_lm

    tl.store(dK_ptr + offset, dK_j, mask = kv_mask)
    tl.store(dV_ptr + offset, dV_j, mask = kv_mask)

def fa_bwd(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                 O: torch.Tensor, dO: torch.Tensor,
                 L: torch.Tensor):
    """
    inputs - dO from torch - upstream gradient 

    outputs - dQ, dK, dV - downstream gradient. helper vars - dS, dP
    """

    assert Q.is_cuda and K.is_cuda and V.is_cuda and O.is_cuda and dO.is_cuda and L.is_cuda

    B, H, M, N = Q.shape
    qk_scale = 1/ math.sqrt(N)

    dQ = torch.zeros_like(Q, device = 'cuda', dtype = torch.float32)
    dK = torch.zeros_like(K, device = 'cuda', dtype = torch.float32)
    dV = torch.zeros_like(V, device = 'cuda', dtype = torch.float32)

    D = O * dO # optimize?
    D = D.sum(dim = -1).reshape(B, H, M, 1)
    grid = lambda meta: (B*H, triton.cdiv(M, meta['BLOCK_SIZE_M'])) 
    fa_bwd_kernel[grid]( 
                        Q, K, V, 
                        O, dO, L, qk_scale,
                        dQ, dK, dV, D,
                        B, H, M, N,
                        *Q.stride(), 
                        *L.stride(), 
                        BLOCK_SIZE_M = 16,
                        BLOCK_SIZE_N = max(16,triton.next_power_of_2(N)))   
    return dQ, dK, dV


@triton.jit
def softmax_kernel(
    x_ptr,
    output_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    stride_xb, stride_xh, stride_xm, stride_xn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_batch = tl.program_id(axis=0) 
    pid_m = tl.program_id(axis=1)

    pid_b = pid_batch // H
    pid_h = pid_batch % H

    offset_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offset_xn = tl.arange(0, BLOCK_SIZE_N) % BLOCK_SIZE_N
    offset =  pid_b * stride_xb + pid_h * stride_xh + offset_xm[:, None] * stride_xm + offset_xn[None, :] * stride_xn
    x_ptr += offset
    mask = (offset_xm[:, None] < M) & (offset_xn[None, :] < N)

    # notation is from flash attrntion 2 paper
    s = tl.load(x_ptr, mask=mask, other = -float('inf'))
    m = tl.max(s, axis=-1, keep_dims=True)
    p = tl.exp(s - m)
    l = tl.sum(p, axis=-1, keep_dims=True)  # correction factor = 0

    tl.store(output_ptr + offset, p / l, mask=mask)

def softmax4D(x: torch.Tensor):
    B, H, M, N = x.shape
    output = torch.zeros((B, H, M, N), device = 'cuda', dtype = torch.float32)
    assert x.is_cuda
    grid = lambda meta: (B*H, triton.cdiv(M, meta['BLOCK_SIZE_M']))
    softmax_kernel[grid](x, output,
                         B, H, M, N,     
                    *x.stride(), 
                    BLOCK_SIZE_M = 32,
                    BLOCK_SIZE_N = triton.next_power_of_2(N))   
    return output
    

@triton.jit
def batched_layer_norm_kernel(x_ptr,output_ptr, eps,
                B, M, N: tl.constexpr,
                stride_xb,  stride_xm, stride_xn,
                mean_ptr, var_ptr,
                stride_mb,  stride_mm, stride_mn,

                BLOCK_SIZE_M: tl.constexpr,
                BLOCK_SIZE_N: tl.constexpr,
                ):

    pid_b = tl.program_id(axis=0) 
    pid_m = tl.program_id(axis=1)

    offset = pid_b * stride_xb + (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:,None] * stride_xm +  tl.arange(0, BLOCK_SIZE_N)[None,:]*stride_xn
    x_ptr += offset
    mean = tl.zeros((BLOCK_SIZE_M, 1), dtype = tl.float32 )
    for n in range (0, tl.cdiv(N, BLOCK_SIZE_N)):
        offset_col = n * BLOCK_SIZE_N * stride_xn
        x = tl.load(x_ptr + offset_col , mask = offset_col < N, other = 0.0)   
        mean += tl.sum(x, axis = -1, keep_dims = True) 
    mean = tl.sum(mean, axis = -1, keep_dims = True) /N
    tl.store(mean_ptr + pid_b * stride_mb +  (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:,None] * stride_mm , mean)

    variance = tl.zeros((BLOCK_SIZE_M, 1), dtype = tl.float32 )
    for n in range (0, tl.cdiv(N, BLOCK_SIZE_N)):
        offset_col = n * BLOCK_SIZE_N * stride_xn
        x = tl.load(x_ptr + offset_col , mask = offset_col < N, other = 0.0)    
        variance += tl.sum((x- mean)*(x- mean), axis = -1, keep_dims = True)

    variance = tl.sum(variance, axis = -1, keep_dims = True) /N
    rstd = 1/tl.sqrt(variance + eps)
    tl.store(var_ptr + pid_b * stride_mb +  (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))[:,None] * stride_mm, variance)

    output_ptr += offset
    for n in range (0, tl.cdiv(N, BLOCK_SIZE_N)):
        offset_col = n * BLOCK_SIZE_N * stride_xn
        x = tl.load(x_ptr + offset_col , mask = offset_col < N, other = 0.0)   
        output = (x - mean) * rstd        
        tl.store(output_ptr  + offset_col ,  output, mask = offset_col < N)

def layer_norm(x: torch.Tensor, eps = 1e-6):
    B, M , N = x.shape
    output = torch.zeros_like(x,  device = x.device, dtype = x.dtype)
    assert x.is_cuda and output.is_cuda
    
    mean = torch.zeros((B, M, 1), device = x.device, dtype = x.dtype)
    var = torch.zeros((B, M, 1), device = x.device, dtype = x.dtype)
    grid = lambda meta: (B, triton.cdiv(M, meta['BLOCK_SIZE_M']))
    batched_layer_norm_kernel[grid](x, output, eps,
                    B, M , N,
                    *x.stride(),
                    mean, var, *mean.stride(),
                    BLOCK_SIZE_M = 32, 
                    BLOCK_SIZE_N = 32
                    )
    return output, mean, var

