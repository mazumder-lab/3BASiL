import copy
import gc
import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners import lora

logger = logging.getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def sync_time():
    """Get synchronized time for accurate CUDA timing."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def randomized_svd(
    A: torch.Tensor,
    num_ranks: int = 64,
    num_oversampling: int = 5,
):
    if A.ndim != 2:
        raise ValueError(f"Expected 2D Matrix, but got {A.ndim}.")

    U, S, V = torch.svd_lowrank(A, num_ranks + num_oversampling)
    # https://pytorch.org/docs/stable/_modules/torch/_lowrank.html#svd_lowrank
    VT = V.mH

    S_sqrt = torch.sqrt(S)
    L1 = U * S_sqrt.unsqueeze(dim=0)
    L2 = VT * S_sqrt.unsqueeze(dim=1)
    L1k = L1[:, :num_ranks]
    L2k = L2[:num_ranks, :]
    return L2k, L1k


def rank_regression(XTX, D_inv, V, W_old, W_S, rank):
    # Minimize ||X W_old - X (W_S + AB)||_F^2 w.r.t. A and B
    W_diff = W_old - W_S
    XTXW = XTX @ W_diff.t()
    tmp = D_inv * V.t() @ XTXW
    A, B = randomized_svd(tmp, num_ranks=rank, num_oversampling=5)
    B = V @ (D_inv * B)
    A, B = B.t(), A.t()
    return A, B


def cg_batch(A, B, A_supp, M_bmm=None, X0=None, rtol=1e-3, atol=0., maxiter=None, verbose=False):
    """Solves a batch of PD matrix linear systems using the preconditioned CG algorithm.

    This function solves matrix linear systems of the form

        A X = B,  

    where A is a n x n positive definite matrix and B is a n x m matrix,
    and X is the n x m matrix representing the solution for the ith system.

    Args:
        A: The positive definite matrix A.
        B: A n x m matrix representing the right hand sides.
        A_supp: Support mask for sparsity constraint.
        M_bmm: (optional) A callable that performs a batch matrix multiply of the preconditioning
        matrices M and a n x m matrix. (default=identity matrix)
        X0: (optional) Initial guess for X, defaults to M_bmm(B). (default=None)
        rtol: (optional) Relative tolerance for norm of residual. (default=1e-3)
        atol: (optional) Absolute tolerance for norm of residual. (default=0)
        maxiter: (optional) Maximum number of iterations to perform. (default=5*n)
        verbose: (optional) Whether or not to use logger for status messages. (default=False)
    """
    n, m = B.shape
    if M_bmm is None:
        M_bmm = lambda x: x
    if X0 is None:
        X0 = M_bmm(B)
    if maxiter is None:
        maxiter = 5 * n
    assert B.shape == (n, m)
    assert X0.shape == (n, m)
    assert rtol > 0 or atol > 0
    assert isinstance(maxiter, int)
    X_k = X0
    R_k = B - A @ X_k
    R_k = R_k * A_supp
    Z_k = M_bmm(R_k)
    P_k = torch.zeros_like(Z_k)
    P_k1 = P_k
    R_k1 = R_k
    R_k2 = R_k
    X_k1 = X0
    Z_k1 = Z_k
    Z_k2 = Z_k
    B_norm = torch.norm(B, dim=1)
    stopping_matrix = torch.max(rtol*B_norm, atol*torch.ones_like(B_norm))
    if verbose:
        logger.debug("Starting CG batch solve")
    optimal = False
    start = time.perf_counter()
    for k in range(1, maxiter + 1):
        Z_k = M_bmm(R_k)
        if k == 1:
            P_k = Z_k
            R_k1 = R_k
            X_k1 = X_k
            Z_k1 = Z_k
        else:
            R_k2 = R_k1
            Z_k2 = Z_k1
            P_k1 = P_k
            R_k1 = R_k
            Z_k1 = Z_k
            X_k1 = X_k
            denominator = (R_k2 * Z_k2).sum(0)
            denominator[denominator == 0] = 1e-8
            beta = (R_k1 * Z_k1).sum(0) / denominator
            P_k = Z_k1 + beta.unsqueeze(0) * P_k1
        denominator = (P_k * (A@P_k)).sum(0)
        denominator[denominator == 0] = 1e-8
        alpha = (R_k1 * Z_k1).sum(0) / denominator
        X_k = X_k1 + alpha.unsqueeze(0) * P_k
        R_k = R_k1 - alpha.unsqueeze(0) * (A@P_k)
        R_k = R_k * A_supp
        residual_norm = torch.norm(A@X_k - B, dim=1)
        if verbose:
            logger.debug("CG iter %d, max_rel_residual=%.4e", k, torch.max(residual_norm/B_norm))
        if (residual_norm <= stopping_matrix).all():
            optimal = True
            break
    end = time.perf_counter()
    if verbose:
        if optimal:
            logger.debug("CG terminated in %d steps (optimal). Took %.3f ms.", k, (end - start) * 1000)
        else:
            logger.debug("CG terminated in %d steps (reached maxiter). Took %.3f ms.", k, (end - start) * 1000)
    return X_k

def sparsegpt_prune(H, Hinv, W_diff, layer, device, prunen=2, prunem=4, blocksize=128, sparsity=None):
    logger.info("Starting SparseGPT-Prune")
    logger.info(f"prunen={prunen}, prunem={prunem}, sparsity={sparsity}")
    logger.debug(f"W_diff shape={tuple(W_diff.shape)}")
    prunen = prunem - prunen
    
    W = W_diff.clone().float()
    rows = W.shape[0]
    columns = W.shape[1]

    tick = sync_time()
    
    dead = torch.diag(H) == 0
    W[:, dead] = 0
    Losses = torch.zeros(rows, device=device)

    mask = None

    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]

        if prunen == 0: 
            if mask is not None:
                mask1 = mask[:, i1:i2]
            else:
                tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                mask1 = tmp <= thresh
        else:
            mask1 = torch.zeros_like(W1) == 1

        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]

            if prunen != 0 and i % prunem == 0:
                tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

            q = w.clone()
            q[mask1[:, i]] = 0

            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2

            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1

        W[:, i1:i2] = Q1
        Losses += torch.sum(Losses1, 1) / 2

        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        
    torch.cuda.synchronize()
    logger.info('sparsegpt_prune time %.2f', (sync_time() - tick))
    logger.debug('sparsegpt_prune error %s', torch.sum(Losses).item())
        
    return W.reshape(layer.weight.shape).to(layer.weight.data.dtype)

def alps_prune(XtX, X_norm, L, Q,  W_diff, layer, dev, nm_n = 2, nm_m = 4, sp = 0.0, rho=0.1, max_iter = 100, update_iter = 3, switch_iter = 30):
    W = W_diff.clone().float().to(dev)
    YtX = torch.zeros_like(W)
    YtX = torch.matmul(W * X_norm, XtX).to(dev)
    admm_st = sync_time()
    XTX_inv = torch.zeros_like(XtX).float().to(dev)
    B = (W * X_norm.to(dev)).t().clone()
    W = None
    B_orig = B.clone()
    V = torch.zeros_like(B)
    D = torch.zeros_like(B)
    D_suppp = torch.zeros_like(B)
    D_supp = torch.zeros_like(B)
    totp, num_cout = B.shape
    XTX_inv = (Q @ ((1/(L+(rho))) * Q).T).float().to(dev)
    init_rho = False
    fix_supp = False
    D_fix = torch.zeros_like(D)
    Res0 = YtX.T
    Res0 = torch.sum(B_orig * Res0)
    Res0 = torch.sum(Res0)
    params = B.shape[0]*B.shape[1]
    if nm_n == 0:
        k_spar = int(np.round((1-sp)*params))
        D = B.clone().reshape(-1)
        _, loss_idx = torch.topk(-D**2,totp * num_cout - k_spar)
        D[loss_idx] = 0    
        D_suppp = (D == 0).to(torch.float)
        D = D.reshape(totp, num_cout)
    else:
        new_dim = totp * num_cout / nm_m
        new_dim = int(new_dim)
        k_spar = totp * num_cout * nm_n/nm_m
        D = B.clone().t().reshape((new_dim, nm_m))
        _, loss_idx = torch.topk(-D**2,nm_m - nm_n, dim = 1)
        D = D.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to(dev),dim=1,index=loss_idx)   
        D_suppp = (D == 0).to(torch.float)
        D = D.reshape(num_cout, totp).t()
    D_init = D.clone()
    errorp = 1
    for i_admm in range(max_iter):
        B = XTX_inv @ (YtX.T-V+rho*D)
        if fix_supp:
            D = ((V + rho * B) / rho) * D_fix
        elif nm_n == 0:
            D = ((V + rho * B) / rho).reshape(-1)
            _, loss_idx = torch.topk(-D**2,totp * num_cout - k_spar)
            D[loss_idx] = 0    
            D = D.reshape(totp, num_cout)   
        else:
            D = ((V + rho * B) / rho).t().reshape((new_dim, nm_m))
            _, loss_idx = torch.topk(-D**2,nm_m - nm_n, dim = 1)
            D = D.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to(dev),dim=1,index=loss_idx) 
            D_supp = (D == 0).to(torch.float)  
            D = D.reshape(num_cout, totp).t()  
        V = V + rho * (B - D)
        if (i_admm+1) % update_iter == 0:
            if nm_n == 0:
                D_supp = (D.reshape(-1) == 0).to(torch.float)
            supp_change = torch.sum((D_supp-D_suppp)**2)
            
            if not fix_supp:
                if supp_change / k_spar > 0.1:
                    init_rho = True
                    rho *= 1.3
                elif supp_change / k_spar > 0.005:
                    init_rho = True
                    rho *= 1.2
                elif supp_change > 0.5:
                    if init_rho:
                        rho *= 1.1
                    else:
                        rho /= 5
                        B = B_orig.clone().to(dev)
                        D = D_init.clone().to(dev)
                        V = torch.zeros_like(B).to(dev)     
                else:
                    if init_rho:
                        break
                    else:
                        rho /= 5
            D_suppp = (D_supp).clone()
            if rho > 1e6:
                rho = 1e6
            XTX_inv = (Q @ ((1/(L+(rho))) * Q).T).float().to(dev)
            if nm_n == 0:
                Btest = B.reshape(-1)
                _, loss_idx = torch.topk(-Btest**2,totp * num_cout - k_spar)
                Btest[loss_idx] = 0    
                Btest = Btest.reshape(totp, num_cout)
            else:
                Btest = B.t().reshape((new_dim, nm_m))
                _, loss_idx = torch.topk(-Btest**2,nm_m - nm_n, dim = 1)
                Btest = Btest.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to(dev),dim=1,index=loss_idx)  
                Btest = Btest.reshape(num_cout, totp).t()
            Resc = torch.matmul(XtX.to(dev),Btest) - YtX.T
            Resc = torch.diag(torch.matmul((Btest-B_orig.to(dev)).t(), Resc))
            errorc = torch.sum(Resc).to("cpu")/Res0
            errorc = errorc.item()
            logger.info("alps_prune iter %d, rel_error %.6f, support_change %.6f, rho %.6f", i_admm, errorc / errorp, supp_change / k_spar, rho)
            if i_admm >= switch_iter and supp_change / k_spar < 0.0003:
                break
    if nm_n == 0:
        B = B.reshape(-1)
        _, loss_idx = torch.topk(-B**2,totp * num_cout - k_spar)
        B[loss_idx] = 0    
        B = B.reshape(totp, num_cout)
    else:
        B = B.t().reshape((new_dim, nm_m))
        _, loss_idx = torch.topk(-B**2,nm_m - nm_n, dim = 1)
        B = B.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to(dev),dim=1,index=loss_idx)  
        B = B.reshape(num_cout, totp).t()
    V = None
    D = None
    Res = torch.matmul(XtX,B ) - YtX.T
    Res = torch.diag(torch.matmul((B  -B_orig).t(), Res))
    error = torch.sum(Res)/Res0
    error = error.item()
    logger.info("alps_prune before backsolve, error %.6f", error)
    admm_time = sync_time() - admm_st
    back_st = sync_time()
    B = cg_batch((XtX).to(dev), YtX.T, 
                    (B != 0).to(torch.float), M_bmm=None, X0=B, rtol=1e-4, atol=0., maxiter=10, verbose=True)
    back_time = sync_time() - back_st
    Res = torch.matmul(XtX,B ) - YtX.T
    Res = torch.diag(torch.matmul((B  -B_orig).t(), Res))
    error = torch.sum(Res)/Res0
    error = error.item()
    torch.cuda.synchronize()
    logger.info("alps_prune iters=%d", i_admm)
    logger.info("alps_prune final error %.6f", error)
    logger.info("alps_prune time admm: %.3f backsolve: %.3f", admm_time, back_time)    
    return (B.t() / X_norm.to(dev)).reshape(layer.weight.shape).to(layer.weight.data.dtype)



def threebasil(XtX, X_norm, L, Q, W_old, lora_A, lora_B, lora_XTX, lora_D_inv, lora_V, layer, dev, nm_n = 2, nm_m = 4, sp = 0.0, rho=0.1, max_iter = 2000, update_iter = 10, switch_iter = 100, rank=64):
    YtX = torch.zeros_like(W_old)
    YtX = torch.matmul(W_old * X_norm, XtX).to(dev)
    XTX_inv = torch.zeros_like(XtX).float().to(dev)
    B = (W_old * X_norm.to(dev)).t().clone()
    B_orig = B.clone()
    D = torch.zeros_like(B)
    D_suppp = torch.zeros_like(B)
    D_supp = torch.zeros_like(B)
    V = torch.zeros_like(B)
    totp, num_cout = B.shape
    XTX_inv = (Q @ ((1/(L+(rho))) * Q).T).float().to(dev)
    init_rho = False
    fix_supp = False
    D_fix = torch.zeros_like(D)
    Res0 = YtX.T
    Res0 = torch.sum(B_orig * Res0)
    Res0 = torch.sum(Res0)
    params = B.shape[0]*B.shape[1]
    if nm_n == 0:
        k_spar = int(np.round((1-sp)*params))
        D = B.clone().reshape(-1)
        _, loss_idx = torch.topk(-D**2,totp * num_cout - k_spar)
        D[loss_idx] = 0    
        D_suppp = (D == 0).to(torch.float)
        D = D.reshape(totp, num_cout)
    else:
        new_dim = totp * num_cout / nm_m
        new_dim = int(new_dim)
        k_spar = totp * num_cout * nm_n/nm_m
        D = B.clone().t().reshape((new_dim, nm_m))
        _, loss_idx = torch.topk(-D**2,nm_m - nm_n, dim = 1)
        D = D.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to(dev),dim=1,index=loss_idx)   
        D_suppp = (D == 0).to(torch.float)
        D = D.reshape(num_cout, totp).t()
    D_init = D.clone()
    T, K = lora_A, lora_B
    for i_admm in range(max_iter):
        B = XTX_inv @ (torch.matmul((W_old - K @ T) * X_norm, XtX).to(dev).T-V+rho*D)
        if fix_supp:
            D = ((V + rho * B) / rho) * D_fix
        elif nm_n == 0:
            D = ((V + rho * B) / rho).reshape(-1)
            _, loss_idx = torch.topk(-D**2,totp * num_cout - k_spar)
            D[loss_idx] = 0    
            D = D.reshape(totp, num_cout)   
        else:
            D = ((V + rho * B) / rho).t().reshape((new_dim, nm_m))
            _, loss_idx = torch.topk(-D**2,nm_m - nm_n, dim = 1)
            D = D.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to(dev),dim=1,index=loss_idx) 
            D_supp = (D == 0).to(torch.float)  
            D = D.reshape(num_cout, totp).t()  
        V = V + rho * (B - D)
        T, K = rank_regression(lora_XTX, lora_D_inv, lora_V, W_old, B.t() / X_norm, rank=rank)
        if (i_admm+1) % update_iter == 0:
            if nm_n == 0:
                D_supp = (D.reshape(-1) == 0).to(torch.float)
            supp_change = torch.sum((D_supp-D_suppp)**2)
            
            if not fix_supp:
                if supp_change / k_spar > 0.1:
                    init_rho = True
                    rho *= 1.1
                elif supp_change / k_spar > 0.005:
                    init_rho = True
                    rho *= 1.05
                elif supp_change > 0.5:
                    if init_rho:
                        rho *= 1.02
                    else:
                        rho /= 5
                        B = B_orig.clone().to(dev)
                        D = D_init.clone().to(dev)
                        V = torch.zeros_like(B).to(dev)     
                else:
                    if init_rho:
                        break
                    else:
                        rho /= 5
            D_suppp = (D_supp).clone()
            if rho > 1e6:
                rho = 1e6
            XTX_inv = (Q @ ((1/(L+(rho))) * Q).T).float().to(dev)
            M_diff = W_old - D.t() / X_norm - K @ T
            diff_norm = torch.norm(M_diff, p="fro")
            true_loss = 1/2 * torch.trace(M_diff @ lora_XTX @ M_diff.t())
            logger.info("3basil iter %d, df_norm=%.6f, true_loss=%.6f, support_change=%.6f, rho=%.6f", i_admm, diff_norm, true_loss, supp_change / k_spar, rho)
            if i_admm >= switch_iter and supp_change / k_spar < 0.0003:
                break
    if nm_n == 0:
        B = B.reshape(-1)
        _, loss_idx = torch.topk(-B**2,totp * num_cout - k_spar)
        B[loss_idx] = 0    
        B = B.reshape(totp, num_cout)
    else:
        B = B.t().reshape((new_dim, nm_m))
        _, loss_idx = torch.topk(-B**2,nm_m - nm_n, dim = 1)
        B = B.scatter(src=torch.zeros((new_dim,nm_m-nm_n)).to(dev),dim=1,index=loss_idx)  
        B = B.reshape(num_cout, totp).t()
    V = None
    D = None
    logger.info("3basil iters=%d", i_admm)
    M_diff = W_old - B.t() / X_norm - K @ T
    diff_norm = torch.norm(M_diff, p="fro")
    true_loss = 1/2 * torch.trace(M_diff @ lora_XTX @ M_diff.t())
    logger.info("3basil final df_norm=%.6f, true_loss=%.6f", diff_norm, true_loss)
    return (B.t() / X_norm.to(dev)).reshape(layer.weight.shape).to(layer.weight.data.dtype), T, K


class LLM_Compressor:
    """Compressing a Model in (S+LR) configuration. 
    Class takes as input a layer from lora_model. It has weight A and B components randomly initialized.
    Three major steps are to be done.
        - Calculate XTX in add_batch
        - Minimization of ||X(W - (S + LR))||_F^2 w.r.t. S and LR s.t. S is Sparse and LR is Low-Rank."""
    def __init__(self, layer, name):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        W = None
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        
        self.scaler_row = torch.zeros((self.columns), device="cpu")

    def add_batch(self, inp, out, blocksize=1024):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, lora.LoraLayer):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1])) #4096x4096
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = inp.to(dtype=self.H.dtype)
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
        
        self.scaler_row += torch.norm(inp.clone().cpu(), p=2, dim=1) ** 2  / self.nsamples
        
        
    def threebasil(self, prunen=0, prunem=0, sparsity=0.0, percdamp=0.01, hess_diag=False, hess_percdamp=0.05):
        start_hess_inv = sync_time()
        module = self.layer
        W_old = module.base_layer.weight.data.clone()
        scale_sqrt = math.sqrt(module.scaling[module.active_adapter[0]])
        A = module.lora_A.default.weight.clone() * scale_sqrt
        B = module.lora_B.default.weight.clone() * scale_sqrt     
        logger.info("Starting 3basil")
        logger.debug("W_old shape=%s, A=%s, B=%s", tuple(W_old.shape), tuple(A.shape), tuple(B.shape))
        H = self.H
        dead = torch.diag(H) == 0
        logger.debug("dead_indices=%s", torch.where(dead == True))
        H[dead, dead] = 1
        rows = W_old.shape[0]
        columns = W_old.shape[1]
        if not hess_diag:
            damp = percdamp * torch.mean(torch.diag(H))
        else:
            damp = hess_percdamp * torch.diag(H) + percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=H.device)
        H[diag, diag] += damp
        XtX = H.clone()
        
        V, S, _ = torch.linalg.svd(XtX)
        D = torch.sqrt(S)
        D_inv = (1/D).unsqueeze(1)

        X_norm = torch.diag(XtX).sqrt() + 1e-9
        scaled_XtX = XtX / X_norm
        scaled_XtX = (scaled_XtX.T / X_norm).T
        L, Q = torch.linalg.eigh(scaled_XtX.double())
        logger.info("Time to calculate H eigendecomp: %.3f", sync_time() - start_hess_inv)
        
        W_S, A, B = threebasil(scaled_XtX, X_norm, L, Q, W_old, A, B, XtX, D_inv, V, self.layer, self.dev, nm_n=prunen, nm_m=prunem, sp=sparsity, rho=0.1, rank=A.shape[0])

        M_diff = W_old - W_S - (B @ A)
        diff_norm = torch.norm(M_diff, p="fro")
        true_loss = 1/2 * torch.trace(M_diff @ self.H @ M_diff.t())
        logger.info("3basil done df_norm=%.6f, true_loss=%.6f", diff_norm, true_loss)
        module.base_layer.weight.copy_(W_S)
        module.lora_A.default.weight.copy_(A / scale_sqrt)
        module.lora_B.default.weight.copy_(B / scale_sqrt)        
        
        
    @torch.no_grad()
    def eora_sparsegpt(self, prunen=0, prunem=0, sparsity=0.0, percdamp=0.01, hess_diag=False, hess_percdamp=0.0):
        module = self.layer
        W_old = module.base_layer.weight.data.clone()
        scale_sqrt = math.sqrt(module.scaling[module.active_adapter[0]])
        A = module.lora_A.default.weight.clone() * scale_sqrt
        B = module.lora_B.default.weight.clone() * scale_sqrt
        logger.info("Starting eora_sparsegpt")
        logger.debug("W_old shape=%s, A=%s, B=%s", tuple(W_old.shape), tuple(A.shape), tuple(B.shape))
        H = self.H
        dead = torch.diag(H) == 0
        logger.debug("dead_indices=%s", torch.where(dead == True))
        H[dead, dead] = 1
        rows = W_old.shape[0]
        columns = W_old.shape[1]
        if not hess_diag:
            damp = percdamp * torch.mean(torch.diag(H))
        else:
            damp = hess_percdamp * torch.diag(H) + percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=H.device)
        H[diag, diag] += damp
        XtX = H.clone()
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        U, S, Vt = torch.linalg.svd(XtX)
        V = U
        D = torch.sqrt(S)
        D_inv = (1/D).unsqueeze(1)
        rank = B.shape[1]
        W_S = sparsegpt_prune(self.H, Hinv, W_old - B @ A, self.layer, self.dev, prunen=prunen, prunem=prunem, sparsity=sparsity)
        A, B = rank_regression(XtX, D_inv, V, W_old, W_S, rank)
        M_diff = W_old - (W_S + B @ A)
        diff_norm = torch.norm(M_diff, p="fro")
        true_loss = 1/2 * torch.trace(M_diff @ self.H @ M_diff.t())
        logger.info("eora_sparsegpt df_norm=%.6f, true_loss=%.6f", diff_norm, true_loss)
        module.base_layer.weight.copy_(W_S)
        module.lora_A.default.weight.copy_(A / scale_sqrt)
        module.lora_B.default.weight.copy_(B / scale_sqrt)
        
    @torch.no_grad()
    def eora_alps(self, prunen=0, prunem=0, sparsity=0.0, percdamp=0.01, hess_diag=False, hess_percdamp=0.0):
        start_hess_inv = sync_time()
        module = self.layer
        W_old = module.base_layer.weight.data.clone()
        scale_sqrt = math.sqrt(module.scaling[module.active_adapter[0]])
        A = module.lora_A.default.weight.clone() * scale_sqrt
        B = module.lora_B.default.weight.clone() * scale_sqrt     
        logger.info("Starting eora_alps")
        logger.debug("W_old shape=%s, A=%s, B=%s", tuple(W_old.shape), tuple(A.shape), tuple(B.shape))
        H = self.H
        dead = torch.diag(H) == 0
        logger.debug("dead_indices=%s", torch.where(dead == True))
        H[dead, dead] = 1
        rows = W_old.shape[0]
        columns = W_old.shape[1]
        if not hess_diag:
            damp = percdamp * torch.mean(torch.diag(H))
        else:
            damp = hess_percdamp * torch.diag(H) + percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=H.device)
        H[diag, diag] += damp
        XtX = H.clone()

        V, S, _ = torch.linalg.svd(XtX)
        D = torch.sqrt(S)
        D_inv = (1/D).unsqueeze(1)

        X_norm = torch.diag(XtX).sqrt() + 1e-9
        scaled_XtX = XtX / X_norm
        scaled_XtX = (scaled_XtX.T / X_norm).T
        L, Q = torch.linalg.eigh(scaled_XtX.double())
        logger.info("Time to calculate H eigendecomp: %.3f", sync_time() - start_hess_inv)
        rank = B.shape[1]
        W_S = alps_prune(scaled_XtX, X_norm, L, Q,  W_old - B @ A, self.layer, self.dev, nm_n=prunen, nm_m=prunem, sp=sparsity, rho=0.1)
        A, B = rank_regression(XtX, D_inv, V, W_old, W_S, rank)
        M_diff = W_old - W_S - (B @ A)
        diff_norm = torch.norm(M_diff, p="fro")
        true_loss = 1/2 * torch.trace(M_diff @ self.H @ M_diff.t())
        logger.info("eora_alps df_norm=%.6f, true_loss=%.6f", diff_norm, true_loss)
        module.base_layer.weight.copy_(W_S)
        module.lora_A.default.weight.copy_(A / scale_sqrt)
        module.lora_B.default.weight.copy_(B / scale_sqrt)

    @torch.no_grad()
    def hassle_free_sparsegpt(self, prunen=0, prunem=0, sparsity=0.0, percdamp=0.01, hess_diag=False, hess_percdamp=0.0, n_iters=80):
        module = self.layer
        W_old = module.base_layer.weight.data.clone()
        scale_sqrt = math.sqrt(module.scaling[module.active_adapter[0]])
        A = module.lora_A.default.weight.clone() * scale_sqrt
        B = module.lora_B.default.weight.clone() * scale_sqrt
        print("Starting SparseGPT-GD")
        print("W_old shape", W_old.shape)
        print("A shape", A.shape)
        print("B shape", B.shape)
        H = self.H
        dead = torch.diag(H) == 0
        print("dead:", torch.where(dead == True))
        H[dead, dead] = 1
        rows = W_old.shape[0]
        columns = W_old.shape[1]
        if not hess_diag:
            damp = percdamp * torch.mean(torch.diag(H))
        else:
            damp = hess_percdamp * torch.diag(H) + percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=H.device)
        H[diag, diag] += damp
        XtX = H.clone()
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        U, S, Vt = torch.linalg.svd(XtX)
        V = U
        D = torch.sqrt(S)
        D_inv = (1/D).unsqueeze(1)
        rank = B.shape[1]
        for it in range(n_iters):
            W_S = sparsegpt_prune(self.H, Hinv, W_old - B @ A, self.layer, self.dev, prunen=prunen, prunem=prunem, sparsity=sparsity)
            A, B = rank_regression(XtX, D_inv, V, W_old, W_S, rank)
            M_diff = W_old - (W_S + B @ A)
            diff_norm = torch.norm(M_diff, p="fro")
            true_loss = 1/2 * torch.trace(M_diff @ self.H @ M_diff.t())
            print(f"For iteration {it}, the data-free distance ||W_old - (W_2_4 + B @ A)||_F^2 = {diff_norm}")
            print(f"For iteration {it}, the true loss ||X(W_old - (W_2_4 + B @ A))||_F^2 = {true_loss}")
        module.base_layer.weight.copy_(W_S)
        module.lora_A.default.weight.copy_(A / scale_sqrt)
        module.lora_B.default.weight.copy_(B / scale_sqrt)

    @torch.no_grad()
    def hassle_free_alps(self, prunen=0, prunem=0, sparsity=0.0, percdamp=0.01, hess_diag=False, hess_percdamp=0.0, n_iters=80):
        start_hess_inv = sync_time()
        module = self.layer
        W_old = module.base_layer.weight.data.clone()
        scale_sqrt = math.sqrt(module.scaling[module.active_adapter[0]])
        A = module.lora_A.default.weight.clone() * scale_sqrt
        B = module.lora_B.default.weight.clone() * scale_sqrt     
        print("Starting SparseGPT-GD")
        print("W_old shape", W_old.shape)
        print("A shape", A.shape)
        print("B shape", B.shape)
        H = self.H
        dead = torch.diag(H) == 0
        print("dead:", torch.where(dead == True))
        H[dead, dead] = 1
        rows = W_old.shape[0]
        columns = W_old.shape[1]
        if not hess_diag:
            damp = percdamp * torch.mean(torch.diag(H))
        else:
            damp = hess_percdamp * torch.diag(H) + percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=H.device)
        H[diag, diag] += damp
        XtX = H.clone()

        U, S, Vt = torch.linalg.svd(XtX)
        V = U
        D = torch.sqrt(S)
        D_inv = (1/D).unsqueeze(1)

        X_norm = torch.diag(XtX).sqrt() + 1e-9
        scaled_XtX = XtX / X_norm
        scaled_XtX = (scaled_XtX.T / X_norm).T
        L, Q = torch.linalg.eigh(scaled_XtX.double())
        print(f"Time to calculate Hinv: {sync_time() - start_hess_inv}")
        rank = B.shape[1]
        for it in range(n_iters):
            W_S = alps_prune(scaled_XtX, X_norm, L, Q,  W_old - B @ A, self.layer, self.dev, nm_n=prunen, nm_m=prunem, sp=sparsity, rho=0.1)
            A, B = rank_regression(XtX, D_inv, V, W_old, W_S, rank)
            M_diff = W_old - W_S - (B @ A)
            diff_norm = torch.norm(M_diff, p="fro")
            true_loss = 1/2 * torch.trace(M_diff @ self.H @ M_diff.t())
            print(f"For iteration {it}, the data-free distance ||W_old - (W_2_4 + B @ A)||_F^2 = {diff_norm}")
            print(f"For iteration {it}, the true loss ||X(W_old - (W_2_4 + B @ A))||_F^2 = {true_loss}")
        module.base_layer.weight.copy_(W_S)
        module.lora_A.default.weight.copy_(A / scale_sqrt)
        module.lora_B.default.weight.copy_(B / scale_sqrt)


    @torch.no_grad()
    def oats(self, prunen=0, prunem=0, sparsity=0.0, n_iters=80, prune_level = "row"):
        module = self.layer
        W_old = module.base_layer.weight.data.clone()
        scale_sqrt = math.sqrt(module.scaling[module.active_adapter[0]])
        A = module.lora_A.default.weight.clone() * scale_sqrt
        B = module.lora_B.default.weight.clone() * scale_sqrt
        logger.info("Starting OATS")
        logger.debug("W_old shape=%s, A=%s, B=%s", tuple(W_old.shape), tuple(A.shape), tuple(B.shape))
        unstruct_sparse = sparsity
        diag_approx = self.scaler_row.clone().reshape((1,-1)).to(self.dev)
        scaled_weight = W_old * torch.sqrt(diag_approx)
        sparse_component = torch.zeros_like(scaled_weight).to(self.dev)

        target_rank = B.shape[1]
        for it in range(n_iters): 
            # Apply PCA
            U, S, V = torch.linalg.svd(scaled_weight - sparse_component , full_matrices=False)
            S[target_rank:] = 0
            low_rank_component = U @ torch.diag(S) @ V
            sparse_component = scaled_weight - low_rank_component
            # Prune the weight
            W_metric = sparse_component.clone()
            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prunen != 0:
                logger.info("OATS applying N:M sparsity")
                W_metric = torch.abs(W_metric)
                for ii in range(W_metric.shape[1]):
                    if ii % prunem == 0:
                        tmp = W_metric[:,ii:(ii+prunem)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prunem - prunen,dim=1, largest=False)[1], True)
            elif prune_level == "row":
                sort_res = torch.sort(torch.abs(W_metric), dim=-1, stable=True)
                # unstructured pruning
                indices = sort_res[1][:,:int(W_metric.shape[1]* unstruct_sparse)]
                W_mask.scatter_(1, indices, True)
            elif prune_level == "global":
                sort_res = torch.sort(torch.flatten(torch.abs(W_metric)), stable=True)
                indices = sort_res[1][:int(W_metric.numel()* unstruct_sparse)]
                W_mask = torch.flatten(W_mask)
                W_mask[indices] = True
                W_mask = torch.unflatten(W_mask, 0 , (W_metric.shape[0], W_metric.shape[1]))
            else:
                raise ValueError(f"Invalid prune_level: {prune_level}. Expected 'row' or 'global'.")
            sparse_component[W_mask] = 0
            final_weight = (low_rank_component + sparse_component) * (1/torch.sqrt(diag_approx))
            M_diff = W_old - final_weight
            diff_norm = torch.norm(M_diff, p="fro")
            true_loss = 1/2 * torch.trace(M_diff @ self.H @ M_diff.t())
            logger.info("OATS iter %d, df_norm=%.6f, true_loss=%.6f", it, diff_norm, true_loss)
        
        U, S, V = torch.linalg.svd(low_rank_component / torch.sqrt(diag_approx), full_matrices=False)
        S[target_rank:] = 0
        S_sqrt = torch.diag(torch.sqrt(S))
        B = U @ S_sqrt
        A = S_sqrt @ V
        B = B[:, :target_rank]
        A = A[:target_rank, :]
        module.base_layer.weight.copy_(sparse_component / torch.sqrt(diag_approx))
        module.lora_A.default.weight.copy_(A / scale_sqrt)
        module.lora_B.default.weight.copy_(B / scale_sqrt)

    @torch.no_grad()
    def alps(self, prunen=0, prunem=0, sparsity=0.0, percdamp=0.01, hess_diag=False, hess_percdamp=0.0):
        start_hess_inv = sync_time()
        module = self.layer
        W_old = module.weight.data.clone()
        H = self.H
        dead = torch.diag(H) == 0
        print("dead:", torch.where(dead == True))
        H[dead, dead] = 1
        rows = W_old.shape[0]
        columns = W_old.shape[1]
        if not hess_diag:
            damp = percdamp * torch.mean(torch.diag(H))
        else:
            damp = hess_percdamp * torch.diag(H) + percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=H.device)
        H[diag, diag] += damp
        XtX = H.clone()
        X_norm = torch.diag(XtX).sqrt() + 1e-9
        scaled_XtX = XtX / X_norm
        scaled_XtX = (scaled_XtX.T / X_norm).T
        L, Q = torch.linalg.eigh(scaled_XtX.double())
        print(f"Time to calculate Hinv: {sync_time() - start_hess_inv}")
        W_S = alps_prune(scaled_XtX, X_norm, L, Q,  W_old, self.layer, self.dev, nm_n=prunen, nm_m=prunem, sp=sparsity, rho=0.1)
        module.weight.copy_(W_S)
        
        
    @torch.no_grad()
    def sparsegpt(self, prunen=0, prunem=0, sparsity=0.0, percdamp=0.01, hess_diag=False, hess_percdamp=0.0):
        module = self.layer
        W_old = module.weight.data.clone()
        H = self.H
        dead = torch.diag(H) == 0
        print("dead:", torch.where(dead == True))
        H[dead, dead] = 1
        rows = W_old.shape[0]
        columns = W_old.shape[1]
        if not hess_diag:
            damp = percdamp * torch.mean(torch.diag(H))
        else:
            damp = hess_percdamp * torch.diag(H) + percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(columns, device=H.device)
        H[diag, diag] += damp
        XtX = H.clone()
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        W_S = sparsegpt_prune(self.H, Hinv, W_old, self.layer, self.dev, prunen=prunen, prunem=prunem, sparsity=sparsity)
        module.weight.copy_(W_S)

    @torch.no_grad()
    def wanda(self, prunen=0, prunem=0, sparsity=0.0):
        """
        Wanda pruning: prunes weights based on weight magnitude weighted by activation norm.
        Metric: |W| * sqrt(activation_norm)
        """
        module = self.layer
        W_old = module.weight.data.clone()
        logger.info("Starting Wanda pruning")
        logger.debug("W_old shape=%s", tuple(W_old.shape))
        
        # Get diagonal approximation of activation statistics (from scaler_row)
        diag_approx = self.scaler_row.clone().reshape((1,-1)).to(self.dev)
        
        # Compute Wanda metric: |W| * sqrt(activation_norm)
        W_metric = torch.abs(W_old) * torch.sqrt(diag_approx)
        
        # Initialize mask (all False = keep all weights initially)
        W_mask = (torch.zeros_like(W_metric) == 1)
        
        if prunen != 0:
            # Structured N:M sparsity
            logger.info("Wanda applying %d:%d sparsity", prunen, prunem)
            for ii in range(W_metric.shape[1]):
                if ii % prunem == 0:
                    tmp = W_metric[:, ii:(ii+prunem)].float()
                    # Get indices of smallest N values in each group of M
                    W_mask.scatter_(1, ii + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)
        else:
            # Unstructured pruning
            logger.info("Wanda applying unstructured sparsity=%.2f", sparsity)
            sort_res = torch.sort(W_metric, dim=-1, stable=True)
            # Get indices of smallest values up to sparsity ratio
            indices = sort_res[1][:, :int(W_metric.shape[1] * sparsity)]
            W_mask.scatter_(1, indices, True)
        
        # Apply mask: set pruned weights to zero
        W_old[W_mask] = 0
        module.weight.copy_(W_old)
        logger.info("Wanda pruning completed")


    def free(self):
        self.H = None
        gc.collect()
        torch.cuda.empty_cache()