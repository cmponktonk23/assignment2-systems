import math
import torch
from einops import einsum


B_q = 16
B_k = 16


def flash_fwd_kernel(Q, K, V):
    N_q, N_k = Q.shape[0], K.shape[0]
    T_q, T_k = N_q // B_q, N_k // B_k
    d = Q.shape[-1]
    sqrt_d = math.sqrt(d)

    O = torch.empty((N_q, d), device=Q.device, dtype=Q.dtype)
    L = torch.empty((N_q,), device=Q.device, dtype=Q.dtype)

    for i in range(T_q):
        # Load Qi from global memory
        q_start = i * B_q
        q_end = q_start + B_q
        Qi = Q[q_start:q_end]

        # Initialize Oi, li, mi
        Oij = torch.zeros((B_q, d), device=Q.device, dtype=Q.dtype)
        lij = torch.zeros((B_q,), device=Q.device, dtype=Q.dtype)
        mij = torch.full((B_q,), float('-inf'), device=Q.device, dtype=Q.dtype)

        for j in range(T_k):
            # Load Kj Vj from global memory
            k_start = j * B_k
            k_end = k_start + B_k
            Kj, Vj = K[k_start:k_end], V[k_start:k_end]

            # Compute tile of pre-softmax attention scores
            Sij = einsum(Qi, Kj, "Bq d, Bk d -> Bq Bk") / sqrt_d

            # Compute mij = max(mi, rowmax(Sij))
            mij_old = mij 
            mij = torch.max(mij, Sij.max(dim=-1).values)

            # Compute P = exp(Sij - mij)
            Pij = (Sij - mij[:, None]).exp()

            # Compute lij = exp(mij_old - mij) + rowsum(Pij)
            scale = (mij_old - mij).exp()
            lij = scale * lij + Pij.sum(dim=-1)

            # Compute Oij = diag(exp(mij_old - mij)) x Oij + Pij x Vj
            Oij = scale[:, None] * Oij + einsum(Pij, Vj, "Bq Bk, Bk d -> Bq d")

        # Compute Oi = diag(li)^-1 x Oi
        Oi = Oij / lij[:, None]
        O[q_start:q_end] = Oi

        # Compute Li = mij + log(lij)
        Li = mij + lij.log()
        L[q_start:q_end] = Li

    return O, L


def flash_bwd_kernel(Q, K, V, O, dO, L):
    N_q, N_k = Q.shape[0], K.shape[0]
    T_q, T_k = N_q // B_q, N_k // B_k
    d = Q.shape[-1]
    sqrt_d = math.sqrt(d)

    dQ = torch.zeros_like(Q, device=Q.device, dtype=Q.dtype)
    dK = torch.zeros_like(K, device=K.device, dtype=K.dtype)
    dV = torch.zeros_like(V, device=V.device, dtype=V.dtype)

    for i in range(T_q):
        # Load Qi from global memory
        q_start = i * B_q
        q_end = q_start + B_q
        Qi = Q[q_start:q_end]

        # Load Oi
        Oi = O[q_start:q_end]

        # Load dOi
        dOi = dO[q_start:q_end]

        # Load Li
        Li = L[q_start:q_end]

        # Initialize Oi, li, mi
        dQi = torch.zeros((B_q, d), device=Q.device, dtype=Q.dtype)

        for j in range(T_k): 
            # Load Kj Vj from global memory
            k_start = j * B_k
            k_end = k_start + B_k
            Kj, Vj = K[k_start:k_end], V[k_start:k_end]

            # Compute tile of pre-softmax attention scores
            Sij = einsum(Qi, Kj, "Bq d, Bk d -> Bq Bk") / sqrt_d

            Pij = (Sij - Li[:, None]).exp()

            dVi = einsum(Pij, dOi, "Bq Bk, Bq d -> Bk d")
            dV[k_start:k_end] += dVi

            dPij = einsum(dOi, Vj, "Bq d, Bk d -> Bq Bk")

            Di = (Oi * dOi).sum(dim=-1)
            dSij = Pij * (dPij - Di[:, None])

            dQi += einsum(dSij, Kj, "Bq Bk, Bk d -> Bq d") / sqrt_d
            dKi = einsum(dSij, Qi, "Bq Bk, Bq d -> Bk d") / sqrt_d
            dK[k_start:k_end] += dKi
        
        dQ[q_start:q_end] = dQi

    return dQ, dK, dV


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        batch_size = Q.shape[0]
        O = []
        L = []

        for batch in range(batch_size):
            O_batch, L_batch = flash_fwd_kernel(Q[batch], K[batch], V[batch])
            O.append(O_batch)
            L.append(L_batch)
        
        O = torch.stack(O, dim=0)
        L = torch.stack(L, dim=0)
        ctx.save_for_backward(Q, K, V, O, L)

        return O


    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        batch_size = Q.shape[0]
        dQ = []
        dK = []
        dV = []

        for batch in range(batch_size):
            dQ_batch, dK_batch, dV_batch = flash_bwd_kernel(Q[batch], K[batch], V[batch], O[batch], dO[batch], L[batch])
            dQ.append(dQ_batch)
            dK.append(dK_batch)
            dV.append(dV_batch)

        dQ = torch.stack(dQ, dim=0)
        dK = torch.stack(dK, dim=0)
        dV = torch.stack(dV, dim=0)

        return dQ, dK, dV, None
