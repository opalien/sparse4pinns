from torch import nn
import torch

from examples.any.list_models import list_models

from examples.any.dataset import TrainAnyDataset, TestAnyDataset
import argparse

from examples.any.model import AnyPINN

from typing import cast

import os
import string
import random
import json

from experiments.tree.execution_tree import ExecutionTree


parser = argparse.ArgumentParser(description="PDE solving.")
parser.add_argument("problem", help="The pde to solve")
parser.add_argument("-m", "--m_matrix", type=int, default=1, help="coté de la matrice (défaut: 1)")
parser.add_argument("-k", "--k_layers", type=int, default=1, help="Un nombre (défaut: 1)")
parser.add_argument("-l", "--language", type=str, help="path to the language o the learning ")
args = parser.parse_args()



lettres = string.ascii_letters
alea = ''.join(random.choice(lettres) for _ in range(10))
print(f"Séquence aléatoire générée: {alea}")
save_path = os.path.join("results", "tree", f'results_{alea}.json')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
x = torch.randn(1, 2).to(device)

p_model = list_models[args.problem]
n = args.m_matrix**2
k = args.k_layers

list_language = json.load(open(args.language, "r"))["bests"] if args.language else []
language = lambda x: x in [element[:len(x)] for element in list_language]

if __name__ == '__main__':
    time_bounds: list[int] = cast(list[int], p_model["bounds"][0])
    spatial_bounds = cast(list[int], p_model["bounds"][1:])

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


    layers = [
        nn.Linear(2, n),
        *[nn.Linear(n, n) for _ in range(k)],
        nn.Linear(n, 1),
    ]

    linear_model = AnyPINN(layers, p_model["pde"])

    #tree = ExecutionTree(4, 2, device, train_dataloader, test_dataloader, linear_model)
    tree = ExecutionTree(
        pinn=linear_model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        work_dir=os.path.join("results", "tree"),
        steps=[i*500 for i in range(4)],
        alea=alea,
        language=language
    )
    tree.run()