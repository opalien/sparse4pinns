from core.models.pinn import PINN, DLinear
from torch import Tensor
from typing import Iterable, Callable
from torch import nn


class AnyPINN(PINN):
    def __init__(self,  layers: Iterable[DLinear],
                        pde: Callable[[PINN, Tensor], Tensor],
                        activation: type[nn.Module]=nn.Tanh, 
                        lmda:float =1.0) -> None:
        super().__init__(layers, activation, lmda)

        self.pde_func = pde


    def pde(self, u_pred: Tensor) -> Tensor:
        return self.pde_func(self, u_pred)
        
