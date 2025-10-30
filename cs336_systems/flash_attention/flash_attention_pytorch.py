import math
import torch
from einops import einsum


B_q = 16
B_k = 16


def flash_fwd_kernel(Q, K, V):
        N_q, N_k = Q.shape[0], K.shape[0]
        T_q, T_k = N_q // B_q, N_k // B_k
        d = Q.shape[-1]

        O = []
        L = []

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
                Sij = einsum(Qi, Kj, "Bq d, Bk d -> Bq Bk") / math.sqrt(d)

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
            O.append(Oi)

            # Compute Li = mij + log(lij)
            Li = mij + lij.log()
            L.append(Li)
        
        O = torch.cat(O, dim=0)
        L = torch.cat(L, dim=0)

        return O, L


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
        ctx.save_for_backward(L)

        return O


    @staticmethod
    def backward(ctx):
        raise NotImplementedError
