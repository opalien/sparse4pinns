from torch import nn
from core.utils.train import train, accuracy
import torch

from core.layers.monarch import MonarchLinear
from core.layers.steam import STEAMLinear

from core.utils.save import save_result

from examples.any.list_models import list_models
from core.utils.convert import linear_to_monarch, linear_to_steam

from examples.any.dataset import TrainAnyDataset, TestAnyDataset
import argparse

from examples.any.model import AnyPINN


import os
import time
import string
import random

from experiments.any.execution_tree import ExecutionTree

match os.cpu_count():
    case None:  torch.set_num_threads(1)
    case n:     torch.set_num_threads(n)


parser = argparse.ArgumentParser(description="PDE solving.")
parser.add_argument("problem", help="The pde to solve")
parser.add_argument("-m", "--m_matrix", type=int, default=1, help="coté de la matrice (défaut: 1)")
parser.add_argument("-k", "--k_layers", type=int, default=1, help="Un nombre (défaut: 1)")
args = parser.parse_args()


lettres = string.ascii_letters
alea = ''.join(random.choice(lettres) for _ in range(10))
print(f"Séquence aléatoire générée: {alea}")
save_path = os.path.join("results", "any", f'results_{alea}.json')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(1, 2).to(device)

p_model = list_models[args.problem]
n = args.m_matrix**2
k = args.k_layers

if __name__ == '__main__':
    time_bounds = p_model["bounds"][0]
    spatial_bounds = p_model["bounds"][1:]

    t_max_for_dataset = time_bounds[1]

    train_dataset = TrainAnyDataset(
        p_model["solution"],
        n_elements=1000,
        n_colloc=10000, 
        shape=spatial_bounds,
        t_max=t_max_for_dataset
    )
    test_dataset = TestAnyDataset(
        p_model["solution"],
        n_elements=1000,
        shape=spatial_bounds,
        t_max=t_max_for_dataset
    )

    train_dataloader  = train_dataset.get_dataloader(1000)
    test_dataloader = test_dataset.get_dataloader(1000)




        #####################################################


    layers = [
        nn.Linear(2, n),
        *[nn.Linear(n, n) for _ in range(k)],
        nn.Linear(n, 1),
    ]

    linear_model = AnyPINN(layers, p_model["pde"])

    tree = ExecutionTree(16, 4, device, train_dataloader, test_dataloader, linear_model)
    tree.run()