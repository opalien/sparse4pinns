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

match os.cpu_count():
    case None:  torch.set_num_threads(1)
    case n:     torch.set_num_threads(n)


parser = argparse.ArgumentParser(description="PDE solving.")
parser.add_argument("problem", help="The pde to solve")
parser.add_argument("-e", "--epoch", type=int, default=100, help="Un nombre (défaut: 100)")
parser.add_argument("-r", "--repetition", type=int, default=1, help="Un nombre (défaut: 1)")
parser.add_argument("-m", "--m_matrix", type=int, default=1, help="coté de la matrice (défaut: 1)")
parser.add_argument("-k", "--k_layers", type=int, default=1, help="Un nombre (défaut: 1)")
parser.add_argument("-f", "--factorization", type=str, default="linear", help="monarch, steam (défaut: monarch)")

args = parser.parse_args()

lettres = string.ascii_letters
alea = ''.join(random.choice(lettres) for _ in range(10))
print(f"Séquence aléatoire générée: {alea}")


save_path = os.path.join("results", "any", f'results_{alea}.json')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(1, 2).to(device)

epoch = args.epoch


N = [args.m_matrix**2]#m*m for m in range(5, 40)]
K = [args.k_layers]
R = range(args.repetition)

if args.problem not in list_models:
    raise ValueError(f"Problem {args.problem} not found in list_models.")

p_model = list_models[args.problem]

lr = 0.0001

if __name__ == "__main__":
    for (n, k, r) in [(n, k, r)   
                        for r in R 
                        for n in N 
                        for k in K ]:
        
        print("n, k, r = ", n, k, r)

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

        #print(f"{train_dataset.elements[:10]=} \n \n {train_dataset.colloc[:10]=} \n \n {test_dataset.elements[:10]=} \n \n")

        train_dataloader  = train_dataset.get_dataloader(1000)
        test_dataloader = test_dataset.get_dataloader(1000)




        #####################################################


        layers = [
            nn.Linear(2, n),
            *[nn.Linear(n, n) for _ in range(k)],
            nn.Linear(n, 1),
        ]

        linear_model = AnyPINN(layers, p_model["pde"])

        optimizer = torch.optim.Adam(linear_model.parameters(), lr=lr)
        train_losses_linear, _, _, test_losses_linear, times_linear = train(linear_model, train_dataloader, optimizer, device, epoch, test_dataloader)
        print("linear_model trained")

        t0 = time.time()
        linear_model(x)
        inference_time = time.time() - t0

        dict_linear = {
            "train_losses": train_losses_linear,
            "test_losses": test_losses_linear,
            "times": times_linear,
            "n": n,
            "k": k,
            "layers": str(layers),
            "model": "linear",
            "inference_time": inference_time,
            "equation" : args.problem,
            "alea" : alea,
        }
        save_result(save_path, dict_linear)


        if args.factorization == "monarch":
            t0 = time.time()
            monarch_linear_trained = linear_to_monarch(linear_model)
            transformation_time = time.time() - t0
            print("architecture : ", monarch_linear_trained)

            print("Accuracy monarch_linear_trained :", acc:=accuracy(monarch_linear_trained, test_dataloader, device))

            t0 = time.time()
            monarch_linear_trained(x)
            inference_time = time.time() - t0

            optimizer = torch.optim.Adam(monarch_linear_trained.parameters(), lr=lr)
            train_losses_linear, _, _, test_losses_linear, times_linear = train(monarch_linear_trained, train_dataloader, optimizer, device, epoch, test_dataloader)
            print("linear_model trained")

            dict_monarch_linear_trained = {
                "train_losses": train_losses_linear,
                "test_losses": test_losses_linear,
                "times": times_linear,
                "n": n,
                "k": k,
                "layers": str(layers),
                "model": "monarch_linear_trained",
                "inference_time": inference_time,
                "transformation_time": transformation_time,
                "accuracy": acc,
                "alea" : alea,  
            }
            save_result(save_path, dict_monarch_linear_trained)
        
        else:
            t0 = time.time()
            monarch_linear_trained = linear_to_steam(linear_model)
            transformation_time = time.time() - t0
            print("architecture : ", monarch_linear_trained)

            print("Accuracy monarch_linear_trained :", acc:=accuracy(monarch_linear_trained, test_dataloader, device))

            t0 = time.time()
            monarch_linear_trained(x)
            inference_time = time.time() - t0

            optimizer = torch.optim.Adam(monarch_linear_trained.parameters(), lr=lr)
            train_losses_linear, _, _, test_losses_linear, times_linear = train(monarch_linear_trained, train_dataloader, optimizer, device, epoch, test_dataloader)
            print("linear_model trained")

            dict_monarch_linear_trained = {
                "train_losses": train_losses_linear,
                "test_losses": test_losses_linear,
                "times": times_linear,
                "n": n,
                "k": k,
                "layers": str(layers),
                "model": "steam_linear_trained",
                "inference_time": inference_time,
                "transformation_time": transformation_time,
                "accuracy": acc,
                "alea" : alea,
            }
            save_result(save_path, dict_monarch_linear_trained)


        ###############################################################################

        if args.factorization == "monarch":
            layers = [
                nn.Linear(2, n),
                *[MonarchLinear(n, n) for _ in range(k)],
                nn.Linear(n, 1),
            ]

            monarch_model = AnyPINN(layers, p_model["pde"])
            optimizer = torch.optim.Adam(monarch_model.parameters(), lr=lr)
            train_losses_monarch, _, _, test_losses_monarch, times_monarch = train(monarch_model, train_dataloader, optimizer, device, epoch, test_dataloader)
            print("monarch_model trained")

            t0 = time.time()
            monarch_model(x)
            inference_time = time.time() - t0

            dict_monarch = {
                "train_losses": train_losses_monarch,
                "test_losses": test_losses_monarch,
                "times": times_monarch,
                "n": n,
                "k": k,
                "layers": str(layers),
                "model": "monarch",
                "inference_time": inference_time,
                "equation" : args.problem,
                "alea" : alea,
            }

            print("Accuracy monarch_model :", accuracy(monarch_model, test_dataloader, device))
            save_result(save_path, dict_monarch)
        #####################################################

        else:
            layers = [
            nn.Linear(2, n),
            *[STEAMLinear(n, n) for _ in range(k)],
            nn.Linear(n, 1),
            ]

            steam_model = AnyPINN(layers, p_model["pde"])
            optimizer = torch.optim.Adam(steam_model.parameters(), lr=lr)
            train_losses_steam, _, _, test_losses_steam, times_steam = train(steam_model, train_dataloader, optimizer, device, epoch, test_dataloader)
            print("steam_model trained")

            t0 = time.time()
            steam_model(x)
            inference_time = time.time() - t0

            dict_stem = {
                "train_losses": train_losses_steam,
                "test_losses": test_losses_steam,
                "times": times_steam,
                "n": n,
                "k": k,
                "layers": str(layers),
                "model": "stem",
                "inference_time": inference_time,
                "equation" : args.problem,
                "alea" : alea,
            }

            print("Accuracy steam_model :", accuracy(steam_model, test_dataloader, device))
            save_result(save_path, dict_stem)

        
        
        print("Finished !!")
        