from core.datasets.pinn_dataset import *

from core.layers.block_diag import *
from core.layers.monarch import *
from core.layers.parameterlike import *

from core.models.pinn import *

from core.tensors.monarch import *
from core.tensors.permutation import *
from core.tensors.tensorlike import *
from core.tensors.block_diag import *

from core.utils.butterfly import *
from core.utils.convert import *
from core.utils.save import *
from core.utils.train import *

from experiments.simple.model import *

import copy

x = torch.randn(2, 4)

if False:
    #########################

    L = nn.Linear(4, 4)

    T = L.weight
    b = L.bias


    M = MonarchTensor.from_dense(T)  

    MP = MonarchParameter(M)

    ML = MonarchLinear.from_tensors(M, b)


    ##############################

    L2 = nn.Linear(4, 4)
    L2.weight.data = ML.dense.data
    L2.bias.data = ML.bias.data

    T2 = L2.weight
    b2 = L2.bias

    M2 = MonarchTensor.from_dense(T2)

    MP2 = MonarchParameter(M2)

    ML2 = MonarchLinear.from_tensors(M2, b2)


    ML3 = MonarchLinear.from_tensors(MonarchTensor.from_dense(L2.weight), L2.bias)




##############################

if True:
    x = torch.randn(2, 1)

    M1 = MonarchTensor.random(2)
    M2 = MonarchTensor.random(2)

    b1 = torch.randn(4)
    b2 = torch.randn(4)

    ML1 = MonarchLinear.from_tensors(M1, b1)
    ML2 = MonarchLinear.from_tensors(M2, b2)


    La = nn.Linear(1, 4)
    Lu = nn.Linear(4, 1)

    L1 = nn.Linear(4, 4)
    L1.weight.data = M1.dense.data
    L1.bias.data = b1

    L2 = nn.Linear(4, 4)
    L2.weight.data = M2.dense.data
    L2.bias.data = b2


    pinn = SimplePINN(
        layers=[
            La, L1, L2, Lu
        ]
    )

    pinn_monarch = SimplePINN(
        layers=[
            La, ML1, ML2, Lu
        ]
    )


            


