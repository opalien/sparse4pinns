import torch
from torch import Tensor

import numpy as np
from typing import Callable

from core.datasets.pinn_dataset import PINNDataset


class TrainAnyDataset(PINNDataset):
    def __init__(self, u:Callable[[Tensor], Tensor], n_elements: int, n_colloc: int, shape: list[tuple[int, ...]], t_max: float = 1.0):
        super().__init__()

        self.shape = shape

        self.u = u

        # JE DETESTE NUMPY !!!
        x=np.random.uniform(*shape) if len(shape) > 1 else np.array([np.random.uniform(*shape[0])])
        #print("x : ", x)
        elements = [(
            a:=torch.tensor(np.concatenate( (np.array([0]), x))),
            torch.as_tensor(self.u(a), dtype=torch.float32) if len(shape) > 1 else torch.as_tensor([self.u(a)], dtype=torch.float32)
        ) for _ in range(n_elements)]

        for dim in range(len(shape)):
            x=np.random.uniform(*shape) if len(shape) > 1 else np.array([np.random.uniform(*shape[0])])
            x[dim] = shape[dim][0]
            elements.extend([(
                a:=torch.tensor(np.concatenate(( np.array([np.random.uniform(0, t_max)]),  x )) ),
                torch.as_tensor(self.u(a), dtype=torch.float32) if len(shape) > 1 else torch.as_tensor([self.u(a)], dtype=torch.float32)
            ) for _ in range(n_elements)])

            x=np.random.uniform(*shape) if len(shape) > 1 else np.array([np.random.uniform(*shape[0])])
            x[dim] = shape[dim][1]
            elements.extend([(
                a:=torch.tensor(np.concatenate(( np.array([np.random.uniform(0, t_max)]),  x )) ),
                torch.as_tensor(self.u(a), dtype=torch.float32) if len(shape) > 1 else torch.as_tensor([self.u(a)], dtype=torch.float32)
            ) for _ in range(n_elements)])


        colloc = [torch.tensor(np.concatenate((np.array([np.random.uniform(0, t_max)]), np.random.uniform(*shape) if len(shape) > 1 else np.array([np.random.uniform(*shape[0])]) ))) for _ in range(n_colloc)]

        self.set_elements(elements)
        self.set_colloc(colloc)


class TestAnyDataset(PINNDataset):
    def __init__(self, u:Callable[[Tensor], Tensor], n_elements: int, shape: list[tuple[int, ...]], t_max: float = 1.0):
        super().__init__()

        self.shape = shape

        self.u = u

        x=np.random.uniform(*shape) if len(shape) > 1 else np.array([np.random.uniform(*shape[0])])
        elements = [(
            a:=torch.tensor(np.concatenate(( np.array([np.random.uniform(0, t_max)]),  x )) ),
            torch.as_tensor(self.u(a), dtype=torch.float32) if len(shape) > 1 else torch.as_tensor([self.u(a)], dtype=torch.float32)
        ) for _ in range(n_elements)]

        self.set_elements(elements)

    
            
        
            
            
        