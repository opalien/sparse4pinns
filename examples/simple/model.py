from core.models.pinn import PINN
from torch import Tensor

class SimplePINN(PINN):

    def pde(self, u_pred: Tensor, a_in: Tensor | None = None) -> Tensor:
        # Handle empty tensor case
        if u_pred.size(0) == 0:
            return u_pred.squeeze(-1)  # Return empty tensor with correct shape
            
        du_dt = self.du_dt(u_pred) if a_in is None else self.J_u(u_pred, a_in)[:, :, 0]
        du_dx = self.du_dx(u_pred) if a_in is None else self.J_u(u_pred, a_in)[:, :, 1:]

        # TODO s'occuper du squeeze: VIVE NUMPY !
        return du_dt + du_dx.squeeze(1)