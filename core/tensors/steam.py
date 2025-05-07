from __future__ import annotations
from ..tensors.tensorlike import TensorLike
from ..tensors.block_diag import BlockDiagTensor
from ..tensors.permutation import bit_rev, BitRevPermutationTensor

from ..layers.learnable_permutation import PermutationSTE

from ..utils.butterfly import blockdiag_butterfly_project

from torch import Tensor, nn
import torch

import math

class STEAMTensor(TensorLike):
    def __new__(cls, tensor: Tensor, permutations: Tensor,*args, **kwargs): # type: ignore
        if not isinstance(tensor, Tensor): # type: ignore
            raise TypeError("STEAMTensor must be initialized with a Tensor")
        if tensor.ndim != 4:
            raise ValueError("STEAMTensor must have 4 dimensions")
        
        if tensor.shape[1] != tensor.shape[2] != tensor.shape[3]:
            raise ValueError("STEAMTensor must have 4 dimensions with the second and third dimensions equal")

        return tensor.as_subclass(cls)
    


    def __init__(self, tensor: Tensor, permutations: Tensor, *args, **kwargs): # type: ignore
        _device = tensor.device
        if permutations.device != _device:
            # Ou lever une erreur, ou déplacer permutations. Pour l'instant, avertissement.
            print(f"Warning: STEAMTensor received tensor on {tensor.device} but permutations on {permutations.device}. Moving permutations.")
            permutations = permutations.to(_device)
            
        self.m: int = tensor.shape[1]
        self.n: int = self.m**2

        self.permutations = permutations
        self.Pbar = BitRevPermutationTensor(self.n, device=_device)

    
    @property 
    def L(self)-> BlockDiagTensor:
        return BlockDiagTensor(self[1, ...])

    
    
    @property 
    def R(self)-> BlockDiagTensor:
        return BlockDiagTensor(self[0, ...])
    

    @property
    def P0(self) -> Tensor:
        return self.permutations[0, ...]
    
    @property
    def P2(self) -> Tensor:
        return self.permutations[1, ...]
    

    @L.setter
    def L(self, new_L: BlockDiagTensor):
        if new_L.shape != (self.m, self.m, self.m):
            raise ValueError(f"Dimension mismatch: L must have shape ({self.m}, {self.m}, {self.m}), but got {new_L.shape}")
        self[1, ...] = new_L.clone().detach().data


    @R.setter
    def R(self, new_R: BlockDiagTensor):
        if new_R.shape != (self.m, self.m, self.m):
            raise ValueError(f"Dimension mismatch: R must have shape ({self.m}, {self.m}, {self.m}), but got {new_R.shape}")
        self[0, ...] = new_R.clone().detach().data


    @P0.setter
    def P0(self, new_P0: Tensor):
        if new_P0.shape != (self.n, self.n):
            raise ValueError(f"Dimension mismatch: P0 must have shape ({self.n}, {self.n}), but got {new_P0.shape}")
        self.permutations[0, ...] = new_P0.clone().detach().data

    
    @P2.setter
    def P2(self, new_P2: Tensor):
        if new_P2.shape != (self.n, self.n):
            raise ValueError(f"Dimension mismatch: P2 must have shape ({self.n}, {self.n}), but got {new_P2.shape}")
        self.permutations[1, ...] = new_P2.clone().detach().data


    def __rmatmul__(self, other: Tensor | TensorLike) -> Tensor:
        return STEAMTensor._rmatmul(other, self)
    
    def __matmul__(self, other: Tensor | STEAMTensor) -> Tensor: # type: ignore
        return STEAMTensor._matmul(self, other)


    @staticmethod
    def _matmul(steam: "STEAMTensor" | nn.Module , other: Tensor | TensorLike) -> Tensor:
        match other:
            case TensorLike():
                return STEAMTensor._matmul(steam, other.dense)
            
            case Tensor():
                if other.shape[0] != steam.n:
                    raise ValueError(f"Dimension mismatch: Matmul requires Tensor dim {steam.n}, but got {other.shape[0]}")
                return steam.P2 @ (steam.L @ (steam.Pbar @ (steam.R @ (steam.P0 @ other))))
            
            case _:
                return NotImplemented
            

    @staticmethod
    def _rmatmul(other: Tensor | TensorLike, steam: "STEAMTensor" | nn.Module) -> Tensor:
        match other:
            case TensorLike():
                return STEAMTensor._rmatmul(other.dense, steam)
            
            case Tensor():
                if other.shape[-1] != steam.n:
                    raise ValueError(f"Dimension mismatch: Matmul requires Tensor dim {steam.n}, but got {other.shape[0]}")
                return other @ steam.P2 @ steam.L @ steam.Pbar @ steam.R @ steam.P0
            
            case _:
                return NotImplemented
            


    @staticmethod
    def random(m: int) -> "STEAMTensor":
        n = m*m
        R = torch.randn(m, m, m)
        L = torch.randn(m, m, m)

        P0 = PermutationSTE.apply(torch.randn(n, n)) # type: ignore
        P2 = PermutationSTE.apply(torch.randn(n, n)) # type: ignore
        return STEAMTensor(torch.stack([R, L]), torch.stack([P0, P2])) # type: ignore
    

    @property
    def dense(self)-> Tensor:
        return self.P2 @ (self.L @ (self.Pbar @ (self.R @ self.P0)))
    

    @staticmethod
    def from_dense(A: Tensor, T: int = 100, alpha: float = 0.001)-> "STEAMTensor":
        if A.ndim != 2:
            raise ValueError("Tensor must be 2D")
        
        n = A.shape[0]
        if ( m_sqrt:=math.isqrt(n) )**2 != n :
            raise ValueError(f"Input tensor must be perfect square, but got shape {A.shape}")
        m = m_sqrt # m est la racine de n pour STEAM
        _device = A.device

        # Initialize permutations and other learnable/intermediate tensors on the correct device
        Pbar_obj = BitRevPermutationTensor(n, device=_device)
        Pbar_dense = Pbar_obj.dense
        
        # Initializing P0, P2 as identity matrices for learnable permutations might be more standard
        # Or, if they are truly meant to start as bit-reversal, clone from Pbar_dense
        # Assuming they start close to bit-reversal for this example, but require grad for STE.
        P0 = Pbar_dense.clone().detach().requires_grad_(True)
        P2 = Pbar_dense.clone().detach().requires_grad_(True)
        # X0, X2 are pre-images for STE, also need to be on the correct device and require grad
        X0 = Pbar_dense.clone().detach().requires_grad_(True) 
        X2 = Pbar_dense.clone().detach().requires_grad_(True)

        # Initialize R and L (factors of the core matrix) on the correct device
        # blockdiag_butterfly_project returns factors already on the device of its input.
        # Initial projection: A_proj = Pbar @ P2.T @ A @ P0.T (target for R, L)
        # Or simpler: R, L are factors of (A @ P0.T) or (Pbar.T @ A @ P0.T)
        # The paper projects M_hat = Pbar P_2^T A P_0^T Pbar
        # Here, the code seems to use: R, L = blockdiag_butterfly_project(A @ P0_current.T)
        # Let's stick to the loop's first projected matrix for initial R, L
        R_curr, L_curr = blockdiag_butterfly_project(A @ P0.data.T, sizes=(m,m)) # Use .data for initial non-grad state

        L_best, R_best, P0_best, P2_best = L_curr.clone(), R_curr.clone(), P0.data.clone(), P2.data.clone()
        norm_best = float("inf")

        for t in range(T):
            L_dense = BlockDiagTensor(L_curr).dense # Ensure BlockDiagTensor.dense respects device
            R_dense = BlockDiagTensor(R_curr).dense
            P0_current_dense = P0.data # From STE, already a permutation matrix
            P2_current_dense = P2.data

            # Gradient update step, ensure all ops are on _device
            term_matrix = P2_current_dense @ L_dense @ Pbar_dense @ R_dense @ P0_current_dense
            error_matrix = term_matrix - A
            
            # Simplified nu calculation as in the original snippet (check for stability/correctness)
            # The norm calculation should be on tensors residing on _device
            norm_factor_denom = torch.norm(L_dense @ Pbar_dense @ R_dense)**2
            nu = 1 / (alpha * norm_factor_denom) if norm_factor_denom > 1e-9 else 1.0/(alpha*1e-9)
            
            # Update X0, X2
            grad_X0_term = (P2_current_dense @ L_dense @ Pbar_dense @ R_dense).T @ error_matrix
            X0 = X0 - nu * grad_X0_term

            grad_X2_term = error_matrix @ (L_dense @ Pbar_dense @ R_dense @ P0_current_dense).T
            X2 = X2 - nu * grad_X2_term
            
            # Apply STE to get new P0, P2
            P0 = PermutationSTE.apply(X0) # PermutationSTE.forward handles device
            P2 = PermutationSTE.apply(X2)

            # Re-calculate R, L based on new P0, P2
            # The projection target is Pbar @ P2.T @ A @ P0.T
            target_for_RL = Pbar_dense @ P2.T @ A @ P0.T
            R_curr, L_curr = blockdiag_butterfly_project(target_for_RL, sizes=(m,m))

            current_norm = (torch.norm( (P2 @ BlockDiagTensor(L_curr).dense @  Pbar_dense @ BlockDiagTensor(R_curr).dense @ P0) - A).item())
            current_norm /= torch.norm(A).item()

            if current_norm < norm_best:
                norm_best = current_norm
                L_best, R_best, P0_best, P2_best = L_curr.clone(), R_curr.clone(), P0.clone(), P2.clone()
                # print(f"STEAM from_dense iter {t}: current norm = {current_norm:.4e}, best norm = {norm_best:.4e}")

        stacked_factors = torch.stack([R_best, L_best]) # Will be on _device
        stacked_perms = torch.stack([P0_best, P2_best]) # Will be on _device
        return STEAMTensor(stacked_factors, stacked_perms)
