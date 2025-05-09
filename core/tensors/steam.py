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
        self.m: int = tensor.shape[1]
        self.n: int = self.m**2

        self.permutations = permutations
        self.Pbar = BitRevPermutationTensor(self.n)

    
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
    
    def to(self, *args, **kwargs) -> "STEAMTensor":
        data_on_device = torch.Tensor.to(self, *args, **kwargs)
        instance = data_on_device.as_subclass(STEAMTensor)
        instance.permutations = self.permutations.to(*args, **kwargs)
        instance.Pbar = self.Pbar.to(*args, **kwargs)
        instance.m = self.m
        instance.n = self.n
        return instance
    

    @staticmethod
    def from_dense(A: Tensor, T: int = 100, alpha: float = 0.001)-> "STEAMTensor":
        if A.ndim != 2:
            raise ValueError("Tensor must be 2D")
        
        n = A.shape[0]
        if ( m:=math.isqrt(n) )**2 != n :
            raise ValueError(f"Input tensor must be perfect square, but got shape {A.shape}")

        Pbar = bit_rev(n).dense
        P0 = bit_rev(n).dense
        P2 = bit_rev(n).dense
        X0 = bit_rev(n).dense
        X2 = bit_rev(n).dense

        R, L = blockdiag_butterfly_project(A @ P0.T)

        L_best, R_best, P0_best, P2_best = None, None, None, None
        norm_best = float("inf")

        for t in range(T):
            #print("t = ", t)
            nu = 1/(alpha 
                                         * torch.norm(BlockDiagTensor(L).dense @  Pbar @ BlockDiagTensor(R).dense)**2
                                        )

            X0 -= nu*(
                        ( P2 @ BlockDiagTensor(L).dense @ Pbar @ BlockDiagTensor(R).dense).T 
                        @ ((P2 @ BlockDiagTensor(L).dense @ Pbar @ BlockDiagTensor(R).dense @ P0) - A) 
                    )

            X2 -= nu*( 
                        ( (P2 @ BlockDiagTensor(L).dense @ Pbar @ BlockDiagTensor(R).dense @ P0) - A) 
                        @ (BlockDiagTensor(L).dense @ Pbar @ BlockDiagTensor(R).dense @ P0).T 
                    )

            P0 = PermutationSTE.apply(X0)
            P2 = PermutationSTE.apply(X2)

            R, L = blockdiag_butterfly_project( Pbar @ P2.T @ A @ P0.T )

            
            if (current_norm:=(torch.norm( (P2 @ BlockDiagTensor(L).dense @  Pbar @ BlockDiagTensor(R).dense @ P0) - A).item()/torch.norm(A))) < norm_best:
                norm_best = current_norm
                L_best = L
                R_best = R
                P0_best = P0
                P2_best = P2

                print("current norm =", current_norm, "best norm =", norm_best)


        return STEAMTensor(torch.stack([R_best, L_best]), torch.stack([P0_best, P2_best]))
