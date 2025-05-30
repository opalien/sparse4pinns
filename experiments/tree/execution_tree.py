import torch

import os
import random
import string
import itertools
from typing import Callable

from typing import cast
from typing import Any
from core.utils.convert import convert
import copy

from core.datasets.pinn_dataset import PINNDataloader
from core.models.pinn import PINN
from examples.any.model import AnyPINN

from experiments.tree.tree_elements import Node, Edge


optimizers = {
    "lbfgs": torch.optim.LBFGS,
    "adam": torch.optim.Adam
}

factors = ["linear", "monarch"]


class ExecutionTree:
    def __init__(self, pinn: PINN, train_dataloader: PINNDataloader, test_dataloader: PINNDataloader, device: str | torch.device, work_dir: str, steps:list[int]= [i*100 for i in range(10)] , alea: str | None = None, language: Callable[[list[tuple[str, str]]], bool] = lambda x: True):
        
        
        self.work_dir_root = work_dir
        

        lettres = string.ascii_letters
        self.alea = ''.join(random.choice(lettres) for _ in range(10)) if alea is None else alea


        self.work_dir = os.path.join(work_dir, self.alea)
        os.makedirs(self.work_dir, exist_ok=True)


        self.epochs = [steps[i] - steps[i - 1] for i in range(1, len(steps))]
        self.steps = steps

        self.possible_params: list[tuple[str, str]] = list(itertools.product(factors, optimizers.keys())) 
        
        self.nodes: list[Node] = [Node(pinn, 0, 0, self.possibles_edges, 0, "linear")] # type: ignore
        self.edges: list[Edge] = []


        self.buffer_device = torch.device("cpu")
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.language = language



    def get_next_edge(self):
        params: tuple[str, str] | None = None
        parent: Node | None = None
        for parent in reversed(self.nodes):
            if len(parent.possibles_params) == 0 or parent.current_steps >= self.steps[-1] or parent.pinn is None:
                continue
            params = parent.possibles_params.pop(0)
            break

        if params is None or parent is None:
            return None
        

        model = cast(AnyPINN, convert(copy.deepcopy(parent.pinn), parent.factor, params[0] ))

        child = Node(
            pinn=model,
            total_epoch=parent.total_epoch + self.epochs[parent.current_steps],
            id=len(self.nodes),
            possibles_params=self.possible_params,
            current_steps=parent.current_steps + 1,
            factor=params[0]
        )

        return Edge(
            parent=parent,
            child=child,
            factor=params[0],
            optimizer=params[1],
            epoch=self.epochs[parent.current_steps]
        )


    def 

