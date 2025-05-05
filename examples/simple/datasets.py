import torch
from torch import Tensor

import numpy as np
from numpy.typing import NDArray

from core.datasets.pinn_dataset import PINNDataset


class SimpleDataset(PINNDataset):
    def u(self, a: Tensor | NDArray[np.float64]) -> Tensor | NDArray[np.float64]:
        _, x = a[0], a[1:]
        match x:
            case _ if isinstance(x, Tensor):
                return np.cos(a[0]-a[1])#np.exp(-(np.pi/2)**2 * t) * torch.sin((np.pi/2)*(torch.tensor([x[0]])+1)) # type: ignore
            case _ if isinstance(x, np.ndarray):
                return np.cos(a[0]-a[1])
            case _:
                raise ValueError("x must be either a Tensor or a numpy array")



class TrainSimpleDataset(SimpleDataset):

    def __init__(self, n_elements: int, n_colloc: int):
        super().__init__()

        x_min = -1.
        x_max = 1.
        t_min = 0.
        t_max = 1.

        elements = [(
            a:=torch.tensor([0] + [np.random.uniform(x_min, x_max)]),
            torch.as_tensor(self.u(a), dtype=torch.float32)
        ) for _ in range(n_elements)]

        elements.extend([(
            a:=torch.tensor([np.random.uniform(0, t_max),  x_min] ),
            torch.as_tensor(self.u(a), dtype=torch.float32)
        ) for _ in range(n_elements)])

        elements.extend([(
            a:=torch.tensor([np.random.uniform(0, t_max),  x_max] ),
            torch.as_tensor(self.u(a), dtype=torch.float32) 
        ) for _ in range(n_elements)])


        colloc = [torch.tensor([np.random.uniform(t_min, t_max)] 
                               + [np.random.uniform(x_min, x_max)]) 
                               for _ in range(n_colloc)]
        
        self.set_elements(elements)
        self.set_colloc(colloc)

        
        
class TestSimpleDataset(SimpleDataset):

    def __init__(self, n_elements: int):
        super().__init__()

        x_min = -1
        x_max = 1
        t_min = 0
        t_max = 1

        elements = [(
            a:=torch.tensor([np.random.uniform(t_min, t_max)] + [np.random.uniform(x_min, x_max)]),
            torch.as_tensor(self.u(a), dtype=torch.float32)
        ) for _ in range(n_elements)]

        self.set_elements(elements)

        