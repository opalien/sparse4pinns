import torch
from torch import Tensor

from typing import Callable

from core.datasets.pinn_dataset import PINNDataset


class TrainAnyDataset(PINNDataset):
    def __init__(self, u:Callable[[Tensor], Tensor], n_elements: int, n_colloc: int, shape: list[tuple[int, ...]], t_max: float = 1.0):
        super().__init__()

        self.shape = shape
        self.u = u

        # shape = [(-1, 1), (0, 2)]
        shape = torch.tensor(shape, dtype=torch.float32)
        
        elements: list[Tensor, Tensor] = [(
            a:=torch.concatenate([torch.zeros(1), torch.rand(len(shape))*(shape[:, 1] - shape[:, 0]) + shape[:, 0]]),
            self.u(a)
        ) for _ in range(n_elements)]
        

        for dim in range(len(shape)):

            X= torch.rand(n_elements, len(shape))*(shape[:, 1] - shape[:, 0]) + shape[:, 0]
            X[:, dim] = torch.ones(n_elements)*shape[dim, 0]

            elements.extend([(
                a:= torch.concatenate([torch.rand(1)*t_max, x]),
                self.u(a)
            ) for x in X])

            X= torch.rand(n_elements, len(shape))*(shape[:, 1] - shape[:, 0]) + shape[:, 0]
            X[:, dim] = torch.ones(n_elements)*shape[dim, 1]

            elements.extend([(
                a:= torch.concatenate([torch.rand(1)*t_max, x]),
                self.u(a)
            ) for x in X])


        colloc = [torch.concatenate([torch.rand(1)*t_max, torch.rand(len(shape))*(shape[:, 1] - shape[:, 0]) + shape[:, 0]])
                   for _ in range(n_colloc)]



