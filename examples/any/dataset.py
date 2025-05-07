import torch
from torch import Tensor
from typing import Callable

from core.datasets.pinn_dataset import PINNDataset


class TrainAnyDataset(PINNDataset):
    def __init__(self, u:Callable[[Tensor], Tensor], n_elements: int, n_colloc: int, shape: list[tuple[int, ...]], t_max: float = 1.0):
        super().__init__()

        shape = torch.tensor(shape, dtype=torch.float32)
        self.u = u

        # shape = [(-1, 1), (0, 2)]       

        # examples points
        elements: list[tuple[Tensor, Tensor]] = [(
            a:=torch.concatenate([torch.zeros(1), torch.rand(len(shape))*(shape[:, 1] - shape[:, 0]) + shape[:, 0]]),
            self.u(a)
        ) for _ in range(n_elements)]
        

        for dim in range(len(shape)):
            # lower bound
            X= torch.rand(n_elements, len(shape))*(shape[:, 1] - shape[:, 0]) + shape[:, 0]
            X[:, dim] = torch.ones(n_elements)*shape[dim, 0]

            elements.extend([(
                a:= torch.concatenate([torch.rand(1)*t_max, x]),
                self.u(a)
            ) for x in X])

            # upper bound
            X= torch.rand(n_elements, len(shape))*(shape[:, 1] - shape[:, 0]) + shape[:, 0]
            X[:, dim] = torch.ones(n_elements)*shape[dim, 1]

            elements.extend([(
                a:= torch.concatenate([torch.rand(1)*t_max, x]),
                self.u(a)
            ) for x in X])


        # collocation points
        colloc = [torch.concatenate([torch.rand(1)*t_max, torch.rand(len(shape))*(shape[:, 1] - shape[:, 0]) + shape[:, 0]])
                   for _ in range(n_colloc)]


        self.set_elements(elements)
        self.set_colloc(colloc)


class TestAnyDataset(PINNDataset):
    def __init__(self, u:Callable[[Tensor], Tensor], n_elements: int, shape: list[tuple[int, ...]], t_max: float = 1.0):
        super().__init__()

        shape = torch.tensor(shape, dtype=torch.float32)
        self.u = u

        elements: list[tuple[Tensor, Tensor]] = [(
            a:=torch.concatenate([torch.rand(1)*t_max, torch.rand(len(shape))*(shape[:, 1] - shape[:, 0]) + shape[:, 0]]),
            self.u(a)
        ) for _ in range(n_elements)]

        self.set_elements(elements)
