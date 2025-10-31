import math
import torch
import triton
import triton.language as tl
from einops import einsum


B_q = 16
B_k = 16


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kq, stride_kd,
    stride_vb, stride_vq, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kq, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vq, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float16)

    # Initialize Oi, li, mi
    Oij = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    lij = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    mij = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)

    if is_causal:
        q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)

    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # Load Kj Vj from global memory
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float16)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float16)

        # Compute tile of pre-softmax attention scores
        Sij = tl.dot(Qi, tl.trans(Kj), out_dtype=tl.float32) * scale
        if is_causal:
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = (q_idx[:, None] >= k_idx[None, :]) & (q_idx[:, None] < N_QUERIES) & (k_idx[None, :] < N_KEYS)
            Sij = tl.where(mask, Sij, float('-inf'))

        # Compute mij = max(mi, rowmax(Sij))
        mij_old = mij
        mij = tl.maximum(mij, tl.max(Sij, axis=1))

        # Compute P = exp(Sij - mij)
        Pij = tl.exp(Sij - tl.broadcast_to(mij[:, None], Sij.shape))

        # Compute lij = exp(mij_old - mij) + rowsum(Pij)
        rescale = tl.exp(mij_old - mij)
        lij = rescale * lij + tl.sum(Pij, axis=1)

        # Compute Oij = diag(exp(mij_old - mij)) x Oij + Pij x Vj
        Oij = tl.dot(
            Pij.to(Vj.dtype), Vj, 
            acc=tl.broadcast_to(rescale[:, None], Oij.shape) * Oij)

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # Compute Oi = diag(li)^-1 x Oi
    Oi = (Oij / tl.broadcast_to(lij[:, None], Oij.shape)).to(dtype=O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, Oi, boundary_check=(0, 1))

    # Compute Li = mij + log(lij)
    Li = (mij + tl.log(lij)).to(dtype=L_block_ptr.type.element_ty)
    tl.store(L_block_ptr, Li, boundary_check=(0,))


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size = Q.shape[0]
        N_q, N_k = Q.shape[1], K.shape[1]
        d = Q.shape[2]

        O = torch.empty((batch_size, N_q, d), device=Q.device, dtype=Q.dtype)
        L = torch.empty((batch_size, N_q,), device=Q.device, dtype=Q.dtype)

        flash_fwd_kernel[(triton.cdiv(N_q, B_q), batch_size)](
                          Q, K, V,
                          O, L,
                          Q.stride(0), Q.stride(1), Q.stride(2),
                          K.stride(0), K.stride(1), K.stride(2),
                          V.stride(0), V.stride(1), V.stride(2),
                          O.stride(0), O.stride(1), O.stride(2),
                          L.stride(0), L.stride(1),
                          N_q, N_k,
                          1.0 / math.sqrt(d),
                          d,
                          B_q,
                          B_k,
                          is_causal,
                        )
        
        ctx.is_causal = is_causal
        ctx.save_for_backward(L)
        return O


    @staticmethod
    def backward(ctx):
        raise NotImplementedError
