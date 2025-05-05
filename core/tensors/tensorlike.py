from torch import Tensor
import abc


TensorMeta = type(Tensor)

class TensorABCMeta(TensorMeta, abc.ABCMeta):
    pass


class Tensorable(abc.ABC):

    @property
    @abc.abstractmethod
    def dense(self) -> Tensor:
        raise NotImplementedError
    


class TensorLike(Tensorable, Tensor, abc.ABC, metaclass=TensorABCMeta):
    pass