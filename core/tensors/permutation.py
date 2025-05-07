from __future__ import annotations
from types import NotImplementedType
import torch
from torch import Tensor

from einops import rearrange, einsum # type: ignore

from ..tensors.tensorlike import Tensorable, TensorLike

import math

integer_dtypes = {
            torch.int8, torch.uint8, 
            torch.int16, torch.int32, torch.int64
}


def is_permutation_vector(tensor: Tensor) -> bool:
    n = tensor.shape[0]
    if tensor.ndim != 1:
        return False
    try:
        counts = torch.bincount(tensor, minlength=n)
        return (counts.shape[0] == n) and bool(torch.all(counts == 1).item())

    except RuntimeError:
        raise ValueError("Tensor must be a 1D tensor of non-negative integers")
    

def is_permutation_matrix(tensor: Tensor) -> bool:
    n, m = tensor.shape

    if n != m:
        return False    
    if not torch.all((tensor == 0.) | (tensor == 1.)):
        return False    
    if not torch.all(torch.sum(tensor, dim=1) == 1):
        return False    
    if not torch.all(torch.sum(tensor, dim=0) == 1):
        return False
    
    return True



class PermutationTensor(TensorLike):
    def __new__(cls, data: Tensor | list, *args, **kwargs): # type: ignore

        if isinstance(data, list):
            data = torch.tensor(data)

        if not isinstance(data, Tensor): # type: ignore
            raise TypeError("PermutationTensor must be initialized with a Tensor")
        if data.ndim != 1:
            raise ValueError("PermutationTensor must have 1 dimension")
        
        if data.dtype not in integer_dtypes:
            raise ValueError("PermutationTensor must be of type int")

        if not is_permutation_vector(data):
            raise ValueError("PermutationTensor must be a permutation")
    
        return data.as_subclass(cls)
    

    def __matmul__(self, other: Tensor | PermutationTensor | Tensorable) -> Tensor:
        return self._matmul(self, other) # type: ignore
    
    def __rmatmul__(self, other: Tensor | PermutationTensor | Tensorable) -> Tensor:
        return self._rmatmul(other, self)
    

    @staticmethod
    def _matmul(permutation: PermutationTensor, other: Tensor | PermutationTensor | Tensorable) -> PermutationTensor | Tensor | NotImplementedType: # type: ignore
        perm_size = permutation.shape[0]

        match other:

            case PermutationTensor():
                if other.shape[0] != perm_size:
                    raise ValueError(f"Dimension mismatch: PermutationTensor shapes must match ({perm_size} != {other.shape[0]})")
                return PermutationTensor(permutation[other])
            
            case Tensorable():
                return PermutationTensor._matmul(permutation, other.dense)
            
            case Tensor() if other.ndim > 0:
                if other.shape[0] != perm_size:
                    raise ValueError(f"Dimension mismatch: PermutationTensor shapes must match ({perm_size} != {other.shape[0]})")
                #return other[permutation.as_subclass(Tensor)]
                perm_indices = permutation.as_subclass(Tensor)
                result = other[perm_indices]
                return result 
            
            case _ :
                return NotImplemented
            

    @staticmethod
    def _rmatmul(other: Tensor | PermutationTensor | Tensorable, permutation: PermutationTensor) -> PermutationTensor | Tensor | NotImplementedType: # type: ignore
        perm_size = permutation.shape[0]

        match other:
            case PermutationTensor():
                if other.shape[0] != perm_size:
                    raise ValueError(f"Dimension mismatch: PermutationTensor shapes must match ({perm_size} != {other.shape[0]})")
                return PermutationTensor(other[permutation])
            

            case Tensorable():
                return PermutationTensor._rmatmul(other.dense, permutation)
            

            case Tensor() if 0 < other.ndim < 3 :
                
                if other.ndim == 1:
                    if other.shape[0] != perm_size:
                        raise ValueError(f"Dimension mismatch: PermutationTensor shapes must match ({perm_size} != {other.shape[0]})")
                
                    inv_perm = torch.empty_like(permutation)
                    inv_perm[permutation] = torch.arange(perm_size, device=permutation.device, dtype=permutation.dtype)
                    return Tensor(other[inv_perm])
                
                if other.ndim == 2:
                    if other.shape[-1] != perm_size: 
                         raise ValueError(f"Dimension mismatch: Last dimension of Tensor ({other.shape[-1]}) must match PermutationTensor size ({perm_size})")
                    return Tensor(other[..., permutation.as_subclass(Tensor)])
                
            case _:
                return NotImplemented
                

    @property
    def dense(self) -> Tensor:        
        return self @ torch.eye(self.shape[0])
    
    
    @staticmethod
    def from_dense(tensor: Tensor) -> "PermutationTensor":
        if tensor.ndim != 2:
            raise ValueError("Tensor must be 2D")
        
        n, m = tensor.shape
        if n != m:
            raise ValueError(f"Input tensor must be square, but got shape {tensor.shape}")

        perm_indices = torch.argmax(tensor, dim=1).to(torch.int64) 
        
        try:
            return PermutationTensor(perm_indices)
        except ValueError as e:
            raise ValueError(f"Could not construct a valid PermutationTensor from the dense matrix: {e}")



class BitRevPermutationTensor(PermutationTensor):
    def __init__(self, n: int, device: torch.device | str | None = None):
        if (sqrt_n := int(math.isqrt(n)))**2 != n:
            raise ValueError("n must be a perfect square")
        
        # Create the initial range tensor directly on the specified device
        p_range = torch.arange(n, device=device)
        # Rearrange. The output p_br will be on the same device as p_range.
        p_br = rearrange(p_range, "(n1 m) -> (m n1)", n1=sqrt_n, m=sqrt_n)
        
        super().__init__(p_br) # p_br is now on the specified device
        self.n = n

    def __matmul__(self, other: Tensor | PermutationTensor | Tensorable) -> Tensor:
        return self._matmul(self, other) # type: ignore
    
    def __rmatmul__(self, other: Tensor | PermutationTensor | Tensorable) -> Tensor:
        return self._rmatmul(other, self) #type: ignore
    
    
    @staticmethod
    def _matmul(this: BitRevPermutationTensor, other: Tensor | PermutationTensor | Tensorable):
        match other:

            case PermutationTensor():
                if other.shape[0] != this.n:
                    raise ValueError(f"Dimension mismatch: PermutationTensor shapes must match ({this.n} != {other.shape[0]})")
                return PermutationTensor(rearrange(other, "(n m) B -> (m n) B", n=this.sqrt_n, m=this.sqrt_n))

            case Tensorable():
                return BitRevPermutationTensor._matmul(this, other.dense)
            

            case Tensor():
                if other.ndim > 0 and other.shape[0] != this.n:
                    raise ValueError(f"Dimension mismatch: PermutationTensor shapes must match ({this.n} != {other.shape[0]})")
                
                return rearrange(other, "(n m) B -> (m n) B", n=this.sqrt_n, m=this.sqrt_n)
            
            case _:
                return NotImplemented


    @staticmethod
    def _rmatmul(other: Tensor | PermutationTensor | Tensorable, this: BitRevPermutationTensor):
        match other:
            case PermutationTensor():
                if other.shape[-1] != this.n:
                    raise ValueError(f"Dimension mismatch: PermutationTensor shapes must match ({this.n} != {other.shape[0]})")
                return PermutationTensor(rearrange(other, "B (n m) -> B (m n)", n=this.sqrt_n, m=this.sqrt_n))

            case Tensorable():
                return BitRevPermutationTensor._rmatmul(other.dense, this)
            

            case Tensor():
                if other.ndim > 0 and other.shape[-1] != this.n:
                    raise ValueError(f"Dimension mismatch: PermutationTensor shapes must match ({this.n} != {other.shape[0]})")
                
                return rearrange(other, "B (n m) -> B (m n)", n=this.sqrt_n, m=this.sqrt_n)
            
            case _:
                return NotImplemented
            
    









def bit_rev(n: int, device: torch.device | str | None = None) -> PermutationTensor:
    if (sqrt_n:=int(math.isqrt(n)))**2 != n:
        raise ValueError("n must be a perfect square")
    
    p_range = torch.arange(n, device=device)
    p_br = rearrange(p_range, "(n1 m) -> (m n1)", n1=sqrt_n, m=sqrt_n)
    return PermutationTensor(p_br)