import torch
from torch import nn
from collections.abc import Iterable
from torch import Tensor
#import copy
#import torch.autograd.functional as F
#from torch.func import vmap, jacrev, hessian

from ..layers.monarch import MonarchLinear
from ..layers.steam import STEAMLinear

DLinear = nn.Linear | MonarchLinear | STEAMLinear



class PINN(nn.Module):

    def __init__(self,  layers: Iterable[DLinear],
                        activation: type[nn.Module]=nn.Tanh, 
                        lmda:float =1.0) -> None:
        
        super().__init__() # type: ignore
        
        self.layers = nn.ModuleList(layers)
        self.activation = activation()
        self.lmda = lmda

        self.a_in   : Tensor | None = None
        self.u_pred : Tensor | None = None

        self.data_loss_v : Tensor | None = None
        self.pde_loss_v  : Tensor | None = None
        self.loss_v      : Tensor | None = None

    


    def forward(self, a: Tensor) -> Tensor:
        a = a.clone().detach().requires_grad_(True)
        self.a_in = a
        for layer in self.layers[:-1]:
            a = self.activation(layer(a))
        u = self.layers[-1](a)
        self.u_pred = u
        return u
    

    def J_u(self, u_pred: Tensor | None = None) -> Tensor:
        u_pred = self.u_pred if u_pred is None else u_pred
        if u_pred is None or self.a_in is None:
            raise ValueError("Call forward before calling J_u")
        
        J = torch.autograd.grad(
            outputs=u_pred,
            inputs=self.a_in,
            grad_outputs=torch.ones_like(u_pred),
            create_graph=True,
            retain_graph=True,
        )[0]

        return J[-u_pred.size(0):] if u_pred.size(0) > 0 else J
    

    def H_u(self, u_pred: Tensor | None = None) -> Tensor:
        u_pred = self.u_pred if u_pred is None else u_pred
        if u_pred is None or self.a_in is None:
            raise ValueError("Call forward before calling H_u")
        
        J = self.J_u(u_pred)
        _, I = self.a_in.shape
        H_cols: list[Tensor]  =[]

        for j in range(I):
            grad_compoment = J[:,j]

            H_col = torch.autograd.grad(
                outputs=grad_compoment,
                inputs=self.a_in,
                grad_outputs=torch.ones_like(grad_compoment),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]

            #if H_col is None:
            #    H_col = torch.zeros(B, I, device=self.a_in.device)
            #
            H_cols.append(H_col)
        
        H = torch.stack(H_cols, dim=2)
        return H[-u_pred.size(0):] if u_pred.size(0) > 0 else H



    def du_dt(self, u_pred: Tensor | None = None) -> Tensor:
        return self.J_u(u_pred)[:, 0]
    

    def du_dx(self, u_pred: Tensor | None = None) -> Tensor:
        return self.J_u(u_pred)[:, 1:]
    

    def du_dtt(self, u_pred: Tensor | None = None) -> Tensor:
       return self.H_u(u_pred)[:, 0, 0]
   

    def du_dxx(self, u_pred: Tensor | None = None):
       return torch.diagonal(self.H_u(u_pred), offset=0, dim1=-2, dim2=-1)[:, 1:]
           

    def data_loss(self, u_pred: Tensor, u: Tensor)-> Tensor:   
        match u_pred.size(0):
            case 0:
                return torch.tensor(0.0, device=u_pred.device)
            
            case _:
                #print(f"u_pred.shape = {u_pred.shape}, u.shape = {u.shape}")
                # TODO : REVOIR CAR PAS NORMAL, VIVE NUMPY !
                if u_pred.shape != u.shape:
                    u = u.unsqueeze(-1)

                #print(f"u_pred.shape = {u_pred.shape}, u.shape = {u.shape}")
                return torch.nn.functional.mse_loss(u_pred, u)


    def pde_loss(self, u_pred: Tensor)-> Tensor:
        match u_pred.size(0):
            case 0:
                return torch.tensor(0.0, device=u_pred.device)

            case _:
                #print(" MSE = ", torch.nn.functional.mse_loss(pde:=self.pde(u_pred), torch.zeros_like(pde)))
                return torch.nn.functional.mse_loss(pde:=self.pde(u_pred), torch.zeros_like(pde))
    

    def pde(self, u_pred: Tensor) -> Tensor:
        raise NotImplementedError("pde method must be implemented in a subclass")


    def loss(self, u: Tensor, idx: int | None =None):
        if self.u_pred is None:
            raise ValueError("Call forward before calling data_loss")
        
        idx = self.u_pred.size(0) if idx is None else idx

        match idx:
            case 0:
                self.data_loss_v = torch.tensor(0.0, device=self.u_pred.device)
                self.pde_loss_v = self.pde_loss(self.u_pred)
                self.loss_v = self.lmda * self.pde_loss_v
            
            case _ if idx == self.u_pred.size(0):
                self.data_loss_v = self.data_loss(self.u_pred, u)
                self.pde_loss_v = torch.tensor(0.0, device=self.u_pred.device)
                self.loss_v = self.data_loss_v
            
            case _:
                self.data_loss_v = self.data_loss(self.u_pred[:idx], u[:idx])   
                self.pde_loss_v = self.pde_loss(self.u_pred[idx:])
                self.loss_v = self.data_loss_v + self.lmda * self.pde_loss_v
        
        return self.loss_v
    

    def get_loss(self) -> Tensor:
        if self.loss_v is None:
            raise ValueError("Call loss before calling get_loss")
        return self.loss_v


    def get_data_loss(self) -> Tensor:
        if self.data_loss_v is None:
            raise ValueError("Call loss before calling get_data_loss")
        return self.data_loss_v


    def get_pde_loss(self) -> Tensor:
        if self.pde_loss_v is None:
            raise ValueError("Call loss before calling get_pde_loss")
        return self.pde_loss_v