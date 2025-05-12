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


dir = os.path.join("examples", "any", "solvers")
burger_pickle_path = os.path.join(dir, "burger_solution.pkl")

def burger_pde(this: PINN, u_pred: Tensor) -> Tensor:
    du_dt = this.du_dt(u_pred)
    du_dx = this.du_dx(u_pred).squeeze(1)
    du_dxx = this.du_dxx(u_pred).squeeze(1)
    u = u_pred.squeeze(1)#[u_pred.size(0)-1:]

    #print(f'{du_dt=}, {du_dx=}, {du_dxx=}, {u=}')
    #print(f'{du_dt.shape=}, {du_dx.shape=}, {du_dxx.shape=}, {u.shape=}')

    return du_dt + u*du_dx - (0.01/np.pi)*du_dxx


burger_interpolator: BurgerSolutionInterpolator = load_solution_interpolator(burger_pickle_path)
burger_solution = burger_interpolator.evaluate





def simple_solution(a: Tensor) -> Tensor:
    _, x = a[0], a[1:]
    return torch.cos(a[0]-a[1]).view(1)#np.exp(-(np.pi/2)**2 * t) * torch.sin((np.pi/2)*(torch.tensor([x[0]])+1)) # type: ignore



def simple_pde(this: PINN, u_pred: Tensor) -> Tensor:
    du_dt = this.du_dt(u_pred)
    du_dx = this.du_dx(u_pred).squeeze(1)
    return du_dt + du_dx





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
    }
}