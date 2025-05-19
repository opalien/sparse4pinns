import torch

import os
import string
import random
import numpy as np
from examples.any.model import AnyPINN
from core.datasets.pinn_dataset import PINNDataloader

from core.utils.train import train, train_lbfgs
from core.utils.convert import convert
from core.utils.save import save_result
import copy
from typing import cast



optimizers = {
    "lbfgs": torch.optim.LBFGS,
    "adam": torch.optim.Adam
}

lr = 0.001

class Node:
    def __init__(self, pinn: AnyPINN | None, total_epoch: int, id: int, possibles_edges: list[tuple[str, str]], current_steps: int, factor: str):
        self.pinn = pinn
        self.total_epoch = total_epoch
        self.id = id
        self.possibles_edges = copy.deepcopy(possibles_edges)
        self.current_steps = current_steps
        self.factor = factor

class Edge:
    def __init__(self, parent: Node, child: Node, factor: str, optimizer: str, epoch: int):
        self.parent = parent
        self.child = child
        self.factor = factor
        self.optimizer = optimizer
        self.epoch = epoch


class ExecutionTree:
    def __init__(self, epoch_max: int, n_steps: int, device: str | torch.device, train_dataloader: PINNDataloader, test_dataloader: PINNDataloader, pinn: AnyPINN, log: bool = False):
        
        lettres = string.ascii_letters
        alea = ''.join(random.choice(lettres) for _ in range(10))
        print(f"Séquence aléatoire générée: {alea}")

        self.save_path = os.path.join("results", "any", f'results_{alea}.json')

        self.epoch_max = epoch_max
        self.n_steps = n_steps
        
        if log:
            self.steps_pos = np.logspace(1, np.log2(epoch_max), n_steps, base=2, endpoint=True, dtype=int)
        else:
            self.steps_pos = [epoch_max // n_steps * i for i in range(n_steps+1) if epoch_max // n_steps * i <= epoch_max]


        self.epochs = [self.steps_pos[i] - self.steps_pos[i - 1] for i in range(1, len(self.steps_pos))]
        
        self.possibles_edges = [(factor, opt) for opt in optimizers.keys() for factor in ["monarch", "linear"] ]

        # PINN, epoch_total, id
        self.nodes: list[Node] = [Node(pinn, 0, 0, self.possibles_edges, 0, "linear")]

        # parent, child, factor, optimizers, epoch
        self.edges: list[Edge] = []

        self.buffer_device = torch.device("cpu")
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    
    
    def get_next_edge(self):        
        i = 0    
        parent : Node | None = None
        for i, parent in [(i, self.nodes[i]) for i in reversed(range(len(self.nodes)))]:
            if parent.pinn is not None and (parent.total_epoch >= self.epoch_max or len(parent.possibles_edges) == 0 or parent.current_steps >= len(self.epochs)):
                parent.pinn = None
                parent.possibles_edges = []
                continue

            if parent.pinn is None:
                continue  

            break

        if parent is None or parent.pinn is None:
            return None

        factor, optimizer = parent.possibles_edges.pop()
        epoch = self.epochs[parent.current_steps]

        child = Node(None, parent.total_epoch + epoch, -1, self.possibles_edges, parent.current_steps+1, factor)
        edge = Edge(parent, child, factor, optimizer, epoch)

        return edge
    
    def train_one_step(self, edge: Edge):
        if edge.parent.pinn is None:
            raise ValueError("Parent PINN is None")
        
        print(f"Training with {edge.parent.factor} -> {edge.factor}")
        
        model: AnyPINN = cast(AnyPINN, convert(copy.deepcopy(edge.parent.pinn), edge.parent.factor, edge.factor ))

        optimizer = optimizers[edge.optimizer](model.parameters(), lr=lr)

        if edge.optimizer == "lbfgs":
            train_losses, _, _, test_losses, times = train_lbfgs(model, self.train_dataloader, optimizer, self.device, edge.epoch, self.test_dataloader)
        else:
            train_losses, _, _, test_losses, times = train(model, self.train_dataloader, optimizer, self.device, edge.epoch, self.test_dataloader)
        model.to(self.buffer_device)
        edge.child.pinn = model

        print("model trained")


        dict_model_trained = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "times": times,
            "n": model.layers[0].out_features,
            "k": len(model.layers),
            "layers": str(model.layers),
            "factor": edge.factor,
            "optimizer": edge.optimizer,
            "epoch": edge.epoch,
            "total_epoch": edge.child.total_epoch,
            "parent_id": edge.parent.id,
            "child_id": edge.child.id
        }

        return dict_model_trained
    

    def run(self):
        while (edge := self.get_next_edge()) is not None:
            dict_model = self.train_one_step(edge)
            

            edge.child.id = len(self.nodes)
            dict_model["child_id"] = edge.child.id

            self.nodes.append(edge.child)
            self.edges.append(edge)

            save_result(self.save_path, dict_model)


if __name__ == "__main__":
    from examples.any.list_models import list_models
    from examples.any.dataset import TrainAnyDataset, TestAnyDataset
    import torch.nn as nn
    print("imports done")
    n = 9
    k = 1
    p_model = list_models["burger"]

    time_bounds = p_model["bounds"][0]
    spatial_bounds = p_model["bounds"][1:]
    t_max_for_dataset = time_bounds[1]

    print("burger function done")

    train_dataset = TrainAnyDataset(
    p_model["solution"],
    n_elements=10,
    n_colloc=10, 
    shape=spatial_bounds,
    t_max=t_max_for_dataset
    )

    print("train dataset done")

    test_dataset = TestAnyDataset(
        p_model["solution"],
        n_elements=10,
        shape=spatial_bounds,
        t_max=t_max_for_dataset
    )
    print("test dataset done")
    #print(f"{train_dataset.elements[:10]=} \n \n {train_dataset.colloc[:10]=} \n \n {test_dataset.elements[:10]=} \n \n")
    train_dataloader  = train_dataset.get_dataloader(10)
    test_dataloader = test_dataset.get_dataloader(10)

    layers = [
        nn.Linear(2, n),
        *[nn.Linear(n, n) for _ in range(k)],
        nn.Linear(n, 1),
    ]

    linear_model = AnyPINN(layers, p_model["pde"])
    print("linear model done")

    execution_tree = ExecutionTree(epoch_max=100, n_steps=10, device="cpu", train_dataloader=train_dataloader, test_dataloader=test_dataloader, pinn=linear_model)
    execution_tree.run()
