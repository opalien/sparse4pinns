from torch import nn
import abc
from ..tensors.tensorlike import Tensorable


class ParameterLike(Tensorable, nn.Module, abc.ABC):
    pass

