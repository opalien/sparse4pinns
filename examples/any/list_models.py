import torch
from torch import Tensor
from core.models.pinn import PINN

from typing import Callable
import numpy as np
from numpy.typing import NDArray

import os

import pickle
#from ..any.solvers.burger import BurgerSolutionInterpolator


from examples.any.solvers.burger_loader import BurgerSolutionInterpolator, load_solution_interpolator
from examples.any.solvers.schrodinger_loader import SchrodingerSolutionInterpolator, load_schrodinger_interpolator


dir = os.path.join("examples", "any", "solvers")
burger_pickle_path = os.path.join(dir, "burger_solution.pkl")
schrodinger_pickle_path = os.path.join(dir, "schrodinger_solution.pkl")

def burger_pde(this: PINN, u_pred: Tensor, a_in: Tensor | None = None) -> Tensor:
    du_dt = this.du_dt(u_pred) if a_in is None else this.J_u(u_pred, a_in)[:, :, 0]
    du_dx = this.du_dx(u_pred) if a_in is None else this.J_u(u_pred, a_in)[:, :, 1:]
    du_dxx = this.du_dxx(u_pred) if a_in is None else torch.diagonal(this.H_u(u_pred, a_in), offset=0, dim1=-2, dim2=-1)[:, :, 1:]
    
    # Squeeze the derivatives properly
    du_dt = du_dt.squeeze(1) if du_dt.dim() > 1 else du_dt
    du_dx = du_dx.squeeze() if du_dx.dim() > 1 else du_dx  # Use squeeze() to remove all singleton dimensions
    du_dxx = du_dxx.squeeze() if du_dxx.dim() > 1 else du_dxx  # Use squeeze() to remove all singleton dimensions
    u = u_pred.squeeze(1)

    #print(f'{du_dt=}, {du_dx=}, {du_dxx=}, {u=}')
    #print(f'{du_dt.shape=}, {du_dx.shape=}, {du_dxx.shape=}, {u.shape=}')

    return du_dt + u*du_dx - (0.01/np.pi)*du_dxx


burger_interpolator: BurgerSolutionInterpolator = load_solution_interpolator(burger_pickle_path)
burger_solution = burger_interpolator.evaluate

# Load Schrödinger solution interpolator
try:
    schrodinger_interpolator: SchrodingerSolutionInterpolator = load_schrodinger_interpolator(schrodinger_pickle_path)
    schrodinger_solution = schrodinger_interpolator.evaluate
    print("Schrödinger solution loaded successfully")
except Exception as e:
    print(f"Warning: Could not load Schrödinger solution: {e}")
    print("Using fallback Gaussian solution for testing")
    
    def schrodinger_fallback_solution(a: Tensor) -> Tensor:
        """
        Fallback solution - simple Gaussian for testing when numerical solution is not available.
        This is NOT a true solution to the Schrödinger equation!
        """
        if a.dim() == 1:
            t, x = a[0], a[1]
            # Simple decaying Gaussian (not physically correct, just for shape)
            sigma = 1.0 + 0.1 * t  # Slowly spreading
            u = torch.exp(-x*x / (2*sigma*sigma)) * torch.cos(0.5*t)
            v = torch.exp(-x*x / (2*sigma*sigma)) * torch.sin(0.5*t) * 0.1  # Small imaginary part
            return torch.tensor([u, v])
        else:
            t, x = a[:, 0:1], a[:, 1:2]
            sigma = 1.0 + 0.1 * t
            u = torch.exp(-x*x / (2*sigma*sigma)) * torch.cos(0.5*t)
            v = torch.exp(-x*x / (2*sigma*sigma)) * torch.sin(0.5*t) * 0.1
            return torch.cat([u, v], dim=1)
    
    schrodinger_solution = schrodinger_fallback_solution





def simple_solution(a: Tensor) -> Tensor:
    _, x = a[0], a[1:]
    return torch.cos(a[0]-a[1]).view(1)#np.exp(-(np.pi/2)**2 * t) * torch.sin((np.pi/2)*(torch.tensor([x[0]])+1)) # type: ignore



def simple_pde(this: PINN, u_pred: Tensor, a_in: Tensor | None = None) -> Tensor:
    du_dt = this.du_dt(u_pred) if a_in is None else this.J_u(u_pred, a_in)[:, :, 0]
    du_dx = this.du_dx(u_pred) if a_in is None else this.J_u(u_pred, a_in)[:, :, 1:]
    
    # Squeeze the derivatives properly
    du_dt = du_dt.squeeze(1) if du_dt.dim() > 1 else du_dt
    du_dx = du_dx.squeeze() if du_dx.dim() > 1 else du_dx  # Use squeeze() to remove all singleton dimensions
    
    return du_dt + du_dx





# Schrödinger equation implementation - uses numerical solution
def schrodinger_pde(this: PINN, h_pred: Tensor, a_in: Tensor | None = None) -> Tensor:
    """
    Nonlinear Schrödinger equation: i*h_t + 0.5*h_xx + |h|^2*h = 0
    where h = u + i*v (complex), so h_pred = [u, v] with shape [batch_size, 2]
    
    Separating into real and imaginary parts:
    Real part: -v_t + 0.5*u_xx + (u^2 + v^2)*u = 0
    Imaginary part: u_t + 0.5*v_xx + (u^2 + v^2)*v = 0
    """
    # Handle empty tensor case
    if h_pred.size(0) == 0:
        return torch.zeros((0, 2), device=h_pred.device, dtype=h_pred.dtype, requires_grad=True)
    
    # Split into real (u) and imaginary (v) parts
    u_pred = h_pred[:, 0:1]  # Real part, shape [batch_size, 1]
    v_pred = h_pred[:, 1:2]  # Imaginary part, shape [batch_size, 1]
    
    # Compute derivatives for real part (u)
    du_dt = this.du_dt(u_pred) if a_in is None else this.J_u(u_pred, a_in)[:, :, 0]
    du_dxx = this.du_dxx(u_pred) if a_in is None else torch.diagonal(this.H_u(u_pred, a_in), offset=0, dim1=-2, dim2=-1)[:, :, 1:]
    
    # Compute derivatives for imaginary part (v)
    dv_dt = this.du_dt(v_pred) if a_in is None else this.J_u(v_pred, a_in)[:, :, 0]
    dv_dxx = this.du_dxx(v_pred) if a_in is None else torch.diagonal(this.H_u(v_pred, a_in), offset=0, dim1=-2, dim2=-1)[:, :, 1:]
    
    # Squeeze the derivatives properly
    du_dt = du_dt.squeeze() if du_dt.dim() > 1 else du_dt
    du_dxx = du_dxx.squeeze() if du_dxx.dim() > 1 else du_dxx
    dv_dt = dv_dt.squeeze() if dv_dt.dim() > 1 else dv_dt
    dv_dxx = dv_dxx.squeeze() if dv_dxx.dim() > 1 else dv_dxx
    
    # Extract u and v values
    u = u_pred.squeeze()
    v = v_pred.squeeze()
    
    # Compute |h|^2 = u^2 + v^2
    h_magnitude_squared = u*u + v*v
    
    # Nonlinear Schrödinger equation components:
    # Real part: -v_t + 0.5*u_xx + (u^2 + v^2)*u = 0
    # Imaginary part: u_t + 0.5*v_xx + (u^2 + v^2)*v = 0
    real_part = -dv_dt + 0.5*du_dxx + h_magnitude_squared*u
    imaginary_part = du_dt + 0.5*dv_dxx + h_magnitude_squared*v
    
    # Return concatenated tensor with shape [batch_size, 2]
    return torch.cat([real_part.unsqueeze(1), imaginary_part.unsqueeze(1)], dim=1).squeeze()

list_models = {
    "simple": {
        "pde": simple_pde,
        "solution": simple_solution,
        "bounds": [[0, 1], [0, 1]]
    },

    "burger": {
        "pde": burger_pde,
        "solution": burger_interpolator.evaluate,
        "bounds": [[0, 1], [-1, 1]]
    },
    
    "schrodinger": {
        "pde": schrodinger_pde,
        "solution": schrodinger_solution,
        "bounds": [[0, np.pi/2], [-5, 5]]
    }
}