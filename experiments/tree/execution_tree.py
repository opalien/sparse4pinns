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
import pickle

from experiments.tree.monoid import monoid_phases



optimizers = {
    "lbfgs": torch.optim.LBFGS,
    "adam": torch.optim.Adam
}

lr = 0.0001

class Node:
    def __init__(self, pinn: AnyPINN | None, total_epoch: int, id: int, possibles_edges: list[tuple[str, str]], current_steps: int, factor: str):
        self.pinn = pinn
        self.total_epoch = total_epoch
        self.id = id
        self.possibles_edges = copy.deepcopy(possibles_edges)
        self.current_steps = current_steps
        self.factor = factor

    def __str__(self):
        return f"Node(id={self.id}, pinn={self.pinn is not None}, total_epoch={self.total_epoch}, current_steps={self.current_steps}, factor={self.factor})"

class Edge:
    def __init__(self, parent: Node, child: Node, factor: str, optimizer: str, epoch: int):
        self.parent = parent
        self.child = child
        self.factor = factor
        self.optimizer = optimizer
        self.epoch = epoch

        self.parent_id = parent.id
        self.child_id = child.id

    def __str__(self):
        return f"Edge(parent_id={self.parent_id}, child_id={self.child_id}, factor={self.factor}, optimizer={self.optimizer}, epoch={self.epoch})"



class ExecutionTree:
    def __init__(self, epoch_max: int, n_steps: int, device: str | torch.device, train_dataloader: PINNDataloader, test_dataloader: PINNDataloader, pinn: AnyPINN, work_dir: str, scheduler: str = "linear", alea: str | None = None ):
        
        self.work_dir_root = work_dir




        lettres = string.ascii_letters
        self.alea = ''.join(random.choice(lettres) for _ in range(10)) if alea is None else alea
        print(f"Séquence aléatoire: {self.alea}")

        self.work_dir = os.path.join(work_dir, self.alea)
        os.makedirs(self.work_dir, exist_ok=True)

        #self.save_path = os.path.join(self.work_dir, self.alea, 'results.json')

        self.epoch_max = epoch_max
        self.n_steps = n_steps
        
        if scheduler == "log":
            #self.steps_pos = np.logspace(0, np.log2(epoch_max), n_steps, base=2, endpoint=True, dtype=int)
            #self.steps_pos = [0] + [i*i for i in range(1, len(self.steps_pos))]
            puiss = [1]
            while puiss[-1] < epoch_max:
                puiss.append(puiss[-1] * 2)
            self.steps_pos = [0] + puiss[-n_steps:]
            
        else:
            self.steps_pos = [epoch_max // n_steps * i for i in range(n_steps+1) if epoch_max // n_steps * i <= epoch_max]


        self.epochs = [self.steps_pos[i] - self.steps_pos[i - 1] for i in range(1, len(self.steps_pos))]
        
        self.possibles_edges = [(factor, opt) for opt in optimizers.keys() for factor in ["monarch", "linear"] ]

        # PINN, epoch_total, id
        node0 = Node(pinn, 0, 0, self.possibles_edges, 0, "linear")
        node0.possibles_edges.pop(2)
        self.nodes: list[Node] = [node0]

        # parent, child, factor, optimizers, epoch
        self.edges: list[Edge] = []

        self.buffer_device = torch.device("cpu")
        self.device = device if isinstance(device, torch.device) else torch.device(device)

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.monoid_sequence = []

    
    def set_alea(self, alea: str):
        self.alea = alea
        self.work_dir = os.path.join(self.work_dir_root, self.alea)
        os.makedirs(self.work_dir, exist_ok=True)
        print(f"work_dir: {self.work_dir}")
    
#    def get_next_edge(self):        
#        i = 0    
#        parent : Node | None = None
#        for i, parent in [(i, self.nodes[i]) for i in reversed(range(len(self.nodes)))]:
#            if parent.pinn is not None and (parent.total_epoch >= self.epoch_max or len(parent.possibles_edges) == 0 or parent.current_steps >= len(self.epochs)):
#                #parent.pinn = None
#                parent.possibles_edges = []
#                continue
#
#            if parent.pinn is None:
#                continue  
#
#            break
#
#        if parent is None or parent.pinn is None:
#            return None
#
#        factor, optimizer = parent.possibles_edges.pop()
#        epoch = self.epochs[parent.current_steps]
#
#        child = Node(None, parent.total_epoch + epoch, -1, self.possibles_edges, parent.current_steps+1, factor)
#        edge = Edge(parent, child, factor, optimizer, epoch)
#        edge.parent_id = parent.id
#        return edge
    

    def get_next_edge(self):
        edge_proto: tuple[str, str] | None = None
        parent: Node | None = None
        for parent in reversed(self.nodes):
            if len(parent.possibles_edges) == 0 or parent.current_steps >= (self.n_steps):
                continue
            edge_proto = parent.possibles_edges.pop()
            break
        if edge_proto is None or parent is None:
            return None
        
        child = Node(None, parent.total_epoch + self.epochs[parent.current_steps], -1, self.possibles_edges, parent.current_steps+1, edge_proto[0])
        edge = Edge(parent, child, edge_proto[0], edge_proto[1], self.epochs[parent.current_steps])
        return edge

    def train_one_step(self, edge: Edge):
        if edge.parent.pinn is None:
            raise ValueError("Parent PINN is None")
        
        print(f"Training with {edge.parent.factor} -> {edge.factor}")
        
        model: AnyPINN = cast(AnyPINN, convert(copy.deepcopy(edge.parent.pinn), edge.parent.factor, edge.factor ))

        print(model)
        error = ""
        if edge.optimizer == "lbfgs":
            optimizer = optimizers[edge.optimizer](
                model.parameters(),
                lr=1.0,
                max_iter=50,
                max_eval=80,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                history_size=150,
                line_search_fn="strong_wolfe"
            )
            try:
                train_losses, _, _, test_losses, times = train_lbfgs(model, self.train_dataloader, optimizer, self.device, edge.epoch, self.test_dataloader)
            except Exception as e:
                edge.child.pinn = None
                edge.child.pinn = None
                edge.child.possibles_edges = []
                print(f"Error training with LBFGS: {e}, leaf deleted")
                error = str(e)
                train_losses, test_losses, times = None, None, None
                import time
                time.sleep(10.)
                exit(0) # ANUULATE THE TREE -> it will be relaunched
        else:
            optimizer = optimizers[edge.optimizer](model.parameters(), lr=0.001)
            try:
                train_losses, _, _, test_losses, times = train(model, self.train_dataloader, optimizer, self.device, edge.epoch, self.test_dataloader)
            except Exception as e:
                edge.child.pinn = None
                edge.child.possibles_edges = []
                print(f"Error training with {edge.optimizer}: {e}, leaf deleted")
                error = str(e)
                train_losses, test_losses, times = None, None, None
                import time
                time.sleep(10.)
                exit(0) # ANUULATE THE TREE

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
            "child_id": edge.child.id,
            "error": error
        }

        return dict_model_trained
    

    def run(self):
        while (edge := self.get_next_edge()) is not None:
            dict_model = self.train_one_step(edge)
            

            edge.child.id = len(self.nodes)
            dict_model["child_id"] = edge.child.id
            edge.child_id = edge.child.id

            self.nodes.append(edge.child)
            self.edges.append(edge)

            save_result(os.path.join(self.work_dir, f"results.json"), dict_model)


    def one_step(self, edge: Edge | None = None):        
        
        

        if edge is not None:
            if (edge.factor, edge.optimizer) not in monoid_phases(self.monoid_sequence):
                print(f"Monoid sequence: {self.monoid_sequence+[(edge.factor, edge.optimizer)]} not validated")
                return    
            self.monoid_sequence.append((edge.factor, edge.optimizer))

            dict_model = self.train_one_step(edge)
            save_result(os.path.join(self.work_dir, f"results.json"), dict_model)

            edge.child.id = len(self.nodes)
            dict_model["child_id"] = edge.child.id
            edge.child_id = edge.child.id
            self.nodes.append(edge.child)
            self.edges.append(edge)

        edges = self.get_all_edges()
        print(f"edges: {edges}")

        edges_to_add=  []
        for edge in edges:
            if (edge.factor, edge.optimizer) in monoid_phases(self.monoid_sequence):
                edges_to_add.append(edge)
            
        edges = edges_to_add
        
        match len(edges):
            case 0:
                self.train_dataloader = None
                self.test_dataloader = None
                save_result(os.path.join(self.work_dir, f"infos.json"), {"is_leaf": True, "monoid_sequence": self.monoid_sequence})
            case _:
                pickle.dump(self, open(os.path.join(self.work_dir, f"tree.pkl"), "wb"))
                os.makedirs(os.path.join(self.work_dir, "edges"), exist_ok=True)
                for i, edge in enumerate(edges):                    
                    pickle.dump(edge, open(os.path.join(self.work_dir, "edges", f"edges_{i}.pkl"), "wb"))
                save_result(os.path.join(self.work_dir, f"infos.json"), {"is_leaf": False, "monoid_sequence": self.monoid_sequence})


        with open(os.path.join(self.work_dir, f"finished"), 'w') as _:
            pass
        return

        #################
        
        edges = self.get_all_edges()
        if len(edges) == 0:
            pickle.dump(self, open(os.path.join(self.work_dir, f"tree.pkl"), "wb"))
            save_result(os.path.join(self.work_dir, f"is_finished.json"), {"is_finished": True, "is_leaf": True})
            with open(os.path.join(self.work_dir, f"end"), 'w') as _:
                pass
            return

        pickle.dump(edges, open(os.path.join(self.work_dir, f"edges.pkl"), "wb"))
        pickle.dump(self, open(os.path.join(self.work_dir, f"tree.pkl"), "wb"))
        save_result(os.path.join(self.work_dir, f"is_finished.json"), {"is_finished": True, "is_leaf": False})
        with open(os.path.join(self.work_dir, f"end"), 'w') as _:
            pass


    def get_all_edges(self):
        edges: list[Edge] = []
        while (edge := self.get_next_edge()) is not None:
            edges.append(edge)
        return edges

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
