from ..models.pinn import PINN

from ..layers.monarch import MonarchLinear, MonarchTensor
from ..layers.steam import STEAMLinear, STEAMTensor

from torch import nn, Tensor
import copy

def linear_to_monarch(model: PINN):
    layers: list[nn.Module] = []
    for layer in model.layers:
        
        if isinstance(layer, nn.Linear) and layer.out_features == layer.in_features:

            layers.append(
                MonarchLinear.from_tensors(MonarchTensor.from_dense(layer.weight), layer.bias)
            )

        else:            
            layers.append(layer)
    
    model.layers = nn.ModuleList(layers)
    return model


def linear_to_steam(model: PINN, T=100, alpha=0.001):
    layers: list[nn.Module] = []
    for layer in model.layers:
        
        if isinstance(layer, nn.Linear) and layer.out_features == layer.in_features:

            layers.append(
                STEAMLinear.from_tensors(STEAMTensor.from_dense(layer.weight, T, alpha), layer.bias)
            )
        else:
            layers.append(layer)

    model.layers = nn.ModuleList(layers)
    return model
                