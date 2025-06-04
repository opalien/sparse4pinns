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
        # Clone and enable gradients for automatic differentiation
        # Don't detach to maintain computational graph for optimizers like LBFGS
        a = a.clone().requires_grad_(True)
        self.a_in = a
        for layer in self.layers[:-1]:
            a = self.activation(layer(a))
        u = self.layers[-1](a)
        self.u_pred = u
        return u
    

    def J_u(self, u_pred: Tensor | None = None, a_in: Tensor | None = None) -> Tensor:
        u_pred = self.u_pred if u_pred is None else u_pred
        if u_pred is None:
            raise ValueError("Call forward before calling J_u")
        
        # Use provided a_in or fall back to self.a_in
        if a_in is None:
            if self.a_in is None:
                raise ValueError("Call forward before calling J_u")
            a_in = self.a_in

        # Handle both 2D and 3D tensors (3D can come from dataloader batching)
        if u_pred.dim() == 3:
            # If 3D, squeeze the batch dimension if batch size is 1
            if u_pred.shape[0] == 1:
                u_pred = u_pred.squeeze(0)
            else:
                raise ValueError(f"Cannot handle 3D u_pred with batch size > 1: {u_pred.shape}")
        
        if a_in.dim() == 3:
            # If 3D, squeeze the batch dimension if batch size is 1  
            if a_in.shape[0] == 1:
                a_in = a_in.squeeze(0)
            else:
                raise ValueError(f"Cannot handle 3D a_in with batch size > 1: {a_in.shape}")

        B, O_dim = u_pred.shape
        if a_in.shape[0] != B:
            raise ValueError(f"Batch size mismatch between u_pred ({B}) and a_in ({a_in.shape[0]})")
        
        I_dim = a_in.shape[1]

        # Handle empty tensors
        if B == 0:
            return torch.zeros((0, O_dim, I_dim), device=a_in.device, dtype=a_in.dtype, requires_grad=True)

        jacobian_components: list[Tensor] = []
        for i in range(O_dim):
            output_component = u_pred[:, i]
            
            grad_outputs_i = torch.ones_like(output_component, device=u_pred.device)
            
            grad_result = torch.autograd.grad(
                outputs=output_component,
                inputs=a_in,
                grad_outputs=grad_outputs_i,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )
            grad_i = grad_result[0]

            if grad_i is None:
                grad_i = torch.zeros((B, I_dim), device=a_in.device, dtype=a_in.dtype, requires_grad=True)
            
            jacobian_components.append(grad_i)

        J = torch.stack(jacobian_components, dim=1)
        
        return J
    

    def H_u(self, u_pred: Tensor | None = None, a_in: Tensor | None = None) -> Tensor:
        u_pred = self.u_pred if u_pred is None else u_pred
        if u_pred is None:
            raise ValueError("Call forward before calling H_u")
        
        # Use provided a_in or fall back to self.a_in
        if a_in is None:
            if self.a_in is None:
                raise ValueError("Call forward before calling H_u")
            a_in = self.a_in
        
        J = self.J_u(u_pred, a_in) 
        
        B, O_dim, I_dim = J.shape
        
        hessians_for_outputs: list[Tensor] = []

        for o in range(O_dim):
            J_o = J[:, o, :]
            
            H_o_cols: list[Tensor] = []
            for i in range(I_dim):
                grad_component_J_o_i = J_o[:, i]
                
                grad_result = torch.autograd.grad(
                    outputs=grad_component_J_o_i,
                    inputs=a_in,
                    grad_outputs=torch.ones_like(grad_component_J_o_i, device=a_in.device),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )
                H_o_col_i = grad_result[0]

                if H_o_col_i is None:
                    H_o_col_i = torch.zeros((B, I_dim), device=a_in.device, dtype=a_in.dtype, requires_grad=True)
                
                H_o_cols.append(H_o_col_i)
            
            H_o = torch.stack(H_o_cols, dim=2)
            hessians_for_outputs.append(H_o)

        H = torch.stack(hessians_for_outputs, dim=1)
        return H



    def _get_corresponding_a_in(self, u_pred: Tensor | None) -> Tensor | None:
        """Get the corresponding slice of a_in for a given u_pred slice."""
        if u_pred is None or self.a_in is None or self.u_pred is None:
            return None
        
        B = u_pred.shape[0]
        total_B = self.u_pred.shape[0]
        
        if B == total_B:
            return self.a_in
        elif B < total_B:
            # Calculate the starting index of the slice
            # The most common pattern is u_pred[idx:] for PDE loss
            start_idx = total_B - B
            return self.a_in[start_idx:]
        else:
            raise ValueError(f"u_pred batch size ({B}) cannot be larger than stored batch size ({total_B})")


    def du_dt(self, u_pred: Tensor | None = None) -> Tensor:
        a_in = self._get_corresponding_a_in(u_pred) if u_pred is not None else None
        return self.J_u(u_pred, a_in)[:, :, 0]
    

    def du_dx(self, u_pred: Tensor | None = None) -> Tensor:
        a_in = self._get_corresponding_a_in(u_pred) if u_pred is not None else None
        return self.J_u(u_pred, a_in)[:, :, 1:]
    

    def du_dtt(self, u_pred: Tensor | None = None) -> Tensor:
       a_in = self._get_corresponding_a_in(u_pred) if u_pred is not None else None
       return self.H_u(u_pred, a_in)[:, :, 0, 0]
   

    def du_dxx(self, u_pred: Tensor | None = None) -> Tensor:
       a_in = self._get_corresponding_a_in(u_pred) if u_pred is not None else None
       return torch.diagonal(self.H_u(u_pred, a_in), offset=0, dim1=-2, dim2=-1)[:, :, 1:]
           

    def data_loss(self, u_pred: Tensor, u: Tensor)-> Tensor:   
        match u_pred.size(0):
            case 0:
                return torch.tensor(0.0, device=u_pred.device, requires_grad=True)
            
            case _:
                #print(f"u_pred.shape = {u_pred.shape}, u.shape = {u.shape}")
                # TODO : REVOIR CAR PAS NORMAL, VIVE NUMPY !
                if u_pred.shape != u.shape:
                    u = u.unsqueeze(-1)

                #print(f"u_pred.shape = {u_pred.shape}, u.shape = {u.shape}")
                return torch.nn.functional.mse_loss(u_pred, u)


    def pde_loss(self, u_pred: Tensor, a_in: Tensor | None = None)-> Tensor:
        match u_pred.size(0):
            case 0:
                return torch.tensor(0.0, device=u_pred.device, requires_grad=True)

            case _:
                pde_result = self.pde(u_pred, a_in)
                # Handle empty PDE results (which can happen with empty input tensors)
                if pde_result.numel() == 0:
                    return torch.tensor(0.0, device=u_pred.device, requires_grad=True)
                return torch.nn.functional.mse_loss(pde_result, torch.zeros_like(pde_result))
    

    def pde(self, u_pred: Tensor, a_in: Tensor | None = None) -> Tensor:
        raise NotImplementedError("pde method must be implemented in a subclass")


    def loss(self, u: Tensor, idx: int | None =None):
        if self.u_pred is None:
            raise ValueError("Call forward before calling data_loss")
        
        idx = self.u_pred.size(0) if idx is None else idx

        match idx:
            case 0:
                self.data_loss_v = torch.tensor(0.0, device=self.u_pred.device, requires_grad=True)
                self.pde_loss_v = self.pde_loss(self.u_pred, self.a_in)
                self.loss_v = self.lmda * self.pde_loss_v
            
            case _ if idx == self.u_pred.size(0):
                self.data_loss_v = self.data_loss(self.u_pred, u)
                self.pde_loss_v = torch.tensor(0.0, device=self.u_pred.device, requires_grad=True)
                self.loss_v = self.data_loss_v
            
            case _:
                self.data_loss_v = self.data_loss(self.u_pred[:idx], u[:idx])   
                # Recompute forward pass for collocation points to maintain computational graph
                colloc_a_in = self.a_in[idx:] if self.a_in is not None else None
                if colloc_a_in is not None:
                    # Compute forward pass without overwriting stored values
                    colloc_u_pred = self._forward_without_storing(colloc_a_in)
                    self.pde_loss_v = self.pde_loss(colloc_u_pred, colloc_a_in)
                else:
                    self.pde_loss_v = torch.tensor(0.0, device=self.u_pred.device, requires_grad=True)
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


    def _forward_without_storing(self, a: Tensor) -> Tensor:
        """Compute forward pass without overwriting stored a_in and u_pred."""
        a = a.clone().requires_grad_(True)
        for layer in self.layers[:-1]:
            a = self.activation(layer(a))
        u = self.layers[-1](a)
        return u