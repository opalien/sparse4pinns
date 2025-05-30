import torch

import os
import random
import string
import itertools
from typing import Callable

from typing import cast
from typing import Any
from core.utils.convert import convert
from core.utils.save import save_result
import copy

from core.datasets.pinn_dataset import PINNDataloader
from core.models.pinn import PINN
from examples.any.model import AnyPINN
from core.utils.train import train, train_lbfgs

from experiments.tree.tree_elements import Node


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

        
        node0 = Node(pinn=pinn,
                     total_epoch=0,
                     id=0,
                     possibles_params=self.possible_params,
                     current_steps=0,
                     factor="linear",
                     optimizer="adam",
                     parent=None,
                     epoch=0)

        self.nodes: list[Node] = [node0]


        self.buffer_device = torch.device("cpu")
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.language = language



    def get_next_node(self):
        params: tuple[str, str] | None = None
        parent: Node | None = None
        for parent in reversed(self.nodes):
            if len(parent.possibles_params) == 0 or parent.current_steps >= self.steps[-1] or parent.pinn is None:
                continue
            params = parent.possibles_params.pop(0)
            break

        if params is None or parent is None or parent.pinn is None:
            return None
        
        #model = cast(AnyPINN, convert(copy.deepcopy(parent.pinn), parent.factor, params[0] ))

        return Node(
            pinn=None,
            total_epoch=parent.total_epoch + self.epochs[parent.current_steps],
            id=len(self.nodes),
            possibles_params=self.possible_params,
            current_steps=parent.current_steps + 1,
            factor=params[0],
            optimizer=params[1],
            parent=parent,
            epoch= self.epochs[parent.current_steps+1]
        )


    def train_one_step(self, node: Node):
        
        if node.pinn is None:
            raise ValueError("Node must have a pinn and an optimizer set before training.")
        
        error = ""
        match node.optimizer:
            case "adam":
                optimizer = optimizers[node.optimizer](node.pinn.parameters(), lr=0.001)

                try:
                    train_losses, _, _, test_losses, times = train(node.pinn, self.train_dataloader, optimizer, self.device, node.epoch, self.test_dataloader)
                except Exception as e:
                    print(f"Error during training with Adam: {e}")
                    return None


            case "lbfgs":
                optimizer = optimizers[node.optimizer](
                    node.pinn.parameters(),
                    lr=1.0,
                    max_iter=10,
                    max_eval=20,
                    tolerance_grad=1e-7,
                    tolerance_change=1e-9,
                    history_size=150,
                    line_search_fn="strong_wolfe"
                )

                try:
                    train_losses, _, _, test_losses, times = train_lbfgs(node.pinn, self.train_dataloader, optimizer, self.device, node.epoch, self.test_dataloader)
                except Exception as e:
                    print(f"Error during training with LBFGS: {e}")
                    return None
                
            case _:
                raise ValueError(f"Unknown optimizer: {node.optimizer}")
            
        
        node.pinn.to(self.buffer_device)
        return {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "times": times,
            "n": node.pinn.layers[0].out_features,
            "k": len(node.pinn.layers),
            "layers": str(node.pinn.layers),
            "factor": node.factor,
            "optimizer": node.optimizer,
            "epoch": node.epoch,
            "total_epoch": node.total_epoch,
            "parent_id": node.parent.id if node.parent is not None else -1,
            "child_id": node.id,
            "error": error
        }
    

    def run(self):
        while (node:= self.get_next_node()) is not None:
            if not self.language(node.get_word()):
                continue

            node.set_model()
            while (results:=self.train_one_step(node)) is None:
                print(f"Error during training of node {node.id}, retrying with a new model.")
                node.set_model()
            
            node.id = len(self.nodes)
            results["child_id"] = node.id
            self.nodes.append(node)
            save_result(os.path.join(self.work_dir, f"results.json"), results)


