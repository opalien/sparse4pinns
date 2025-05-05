from core.models.pinn import PINN
from torch import Tensor

class SimplePINN(PINN):

    def pde(self, u_pred: Tensor) -> Tensor:
        du_dt = self.du_dt(u_pred)
        du_dx = self.du_dx(u_pred)

        # TODO s'occuper du squeeze: VIVE NUMPY !
        return du_dt + du_dx.squeeze(1)