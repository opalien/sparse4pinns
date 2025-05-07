from __future__ import annotations
from ..tensors.tensorlike import TensorLike
from ..tensors.block_diag import BlockDiagTensor
from ..tensors.permutation import bit_rev, BitRevPermutationTensor

from torch import Tensor, nn
import torch
import math

from ..utils.butterfly import blockdiag_butterfly_project


from  einops import rearrange, einsum # type: ignore


class MonarchTensor(TensorLike):
    def __new__(cls, tensor: Tensor, *args, **kwargs): # type: ignore
        if not isinstance(tensor, Tensor): # type: ignore
            raise TypeError("MonarchTensor must be initialized with a Tensor")
        if tensor.ndim != 4:
            raise ValueError("MonarchTensor must have 4 dimensions")
        
        if tensor.shape[1] != tensor.shape[2] != tensor.shape[3]:
            raise ValueError("MonarchTensor must have 4 dimensions with the second and third dimensions equal")

        return tensor.as_subclass(cls)


    def __init__(self, tensor: Tensor, *args, **kwargs): # type: ignore
        self.m: int = tensor.shape[1]
        self.n: int = self.m**2
        _device = tensor.device
        self.P1 = BitRevPermutationTensor(self.n, device=_device)
        self.P2  = BitRevPermutationTensor(self.n, device=_device)


    @property 
    def L(self)-> BlockDiagTensor:
        return BlockDiagTensor(self[1, ...])


    @property 
    def R(self)-> BlockDiagTensor:
        return BlockDiagTensor(self[0, ...])


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


    @property
    def dense(self) -> Tensor:
        return self.P2 @ self.L @ self.P1 @ self.R
    
    @staticmethod
    def from_dense(A: Tensor, m: int | None = None) -> "MonarchTensor":
        if A.ndim != 2:
            raise ValueError("Tensor must be 2D")
        
        n = A.shape[0]
        _device = A.device

        if m is None:
            if (m_sqrt := math.isqrt(n))**2 != n:
                raise ValueError(f"Input tensor must be perfect square if m is not provided, but got shape {A.shape}")
            m = m_sqrt
        
        if m*m != n:
            raise ValueError(f"m*m must be equal to n, but {m*m} != {n}")

        P1_init = BitRevPermutationTensor(n, device=_device)
        
        A_perm = A @ P1_init.dense.T 
        
        w1_bfly, w2_bfly = blockdiag_butterfly_project(A_perm)
        
        R = torch.zeros((m, m, m), dtype=A.dtype, device=_device) 
        L = torch.zeros((m, m, m), dtype=A.dtype, device=_device)

        for i in range(m):
            R[i, :, :] = w1_bfly[i, :, :]
            L[i, :, :] = w2_bfly[i, :, :]
        
        return MonarchTensor(torch.stack([R, L]))


    def __rmatmul__(self, other: Tensor | TensorLike) -> Tensor:
        return MonarchTensor._rmatmul(other, self)


    def __matmul__(self, other: Tensor | "MonarchTensor") -> Tensor: # type: ignore
        return MonarchTensor._matmul(self, other)


    @staticmethod
    def _matmul(monarch: MonarchTensor | nn.Module, other: Tensor | TensorLike) -> Tensor | NotImplementedError:
        match other:
            case TensorLike():
                return MonarchTensor._matmul(monarch, other.dense)
            
            case Tensor():
                if other.shape[0] != monarch.n:
                    raise ValueError(f"Dimension mismatch: MonarchTensor matmul requires Tensor dim {monarch.n}, but got {other.shape[-1]}")
                
                return monarch.P2 @ (monarch.L @ (monarch.P1 @ (monarch.R @ other)))
            
            case _:
                return NotImplemented


    @staticmethod
    def _rmatmul(other: Tensor | TensorLike, monarch: MonarchTensor | nn.Module) -> Tensor | NotImplementedError:
        match other:
            case TensorLike():
                return MonarchTensor._rmatmul(other.dense, monarch)
            
            case Tensor():
                if other.shape[-1] != monarch.n:
                    raise ValueError(f"Dimension mismatch: MonarchTensor rmatmul requires Tensor dim {monarch.n}, but got {other.shape[-1]}")
                
                return other @ monarch.P2 @ monarch.L @ monarch.P1 @ monarch.R
            
            case _:
                return NotImplemented



    @staticmethod
    def random(m: int) -> "MonarchTensor":
        R = torch.randn(m, m, m)
        L = torch.randn(m, m, m)

        W = torch.full((2, m, m, m), torch.nan)
        W[0, ...] = R
        W[1, ...] = L

        return MonarchTensor(
            torch.stack([R, L])
        )
