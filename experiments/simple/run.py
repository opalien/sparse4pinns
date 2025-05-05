from torch import nn
from core.utils.train import train, accuracy
import torch

from core.layers.monarch import MonarchLinear
from core.layers.steam import STEAMLinear

from core.utils.save import save_result

from examples.simple.model import SimplePINN
from examples.simple.datasets import TrainSimpleDataset, TestSimpleDataset

from core.utils.convert import linear_to_monarch

import os
import time

match os.cpu_count():
    case None:  torch.set_num_threads(1)
    case n:     torch.set_num_threads(n)

save_path = os.path.join("results", "simple", "results_monarch.json")
epoch = 100


N = [9]#m*m for m in range(5, 40)]
K = [1]
I = range(100)



if __name__ == "__main__":
    for (n, k, i) in [(n, k, i) for i in I for n in N for k in K ]:
        print(f"i: {i}, n: {n}, k: {k}")
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x = torch.randn(1, 2).to(device)

        train_dataset = TrainSimpleDataset(1000, 10_000)
        test_dataset  = TestSimpleDataset(1000)

        train_dataloader  = train_dataset.get_dataloader(200)
        test_dataloader = test_dataset.get_dataloader(200)

        

        #####################################################
        layers = [
            nn.Linear(2, n),
            *[nn.Linear(n, n) for _ in range(k)],
            nn.Linear(n, 1),
        ]
    
        linear_model = SimplePINN(layers)
        optimizer = torch.optim.Adam(linear_model.parameters(), lr=0.001)
        train_losses_linear, _, _, test_losses_linear, times_linear = train(linear_model, train_dataloader, optimizer, device, 0, test_dataloader)
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
        }

        print("Accuracy linear_model :", accuracy(linear_model, test_dataloader, device))

        #####################################################

        t0 = time.time()
        monarch_linear_trained = linear_to_monarch(linear_model)
        transformation_time = time.time() - t0
        print("architecture : ", monarch_linear_trained)

        print("Accuracy monarch_linear_trained :", acc:=accuracy(monarch_linear_trained, test_dataloader, device))

        t0 = time.time()
        monarch_linear_trained(x)
        inference_time = time.time() - t0



        dict_monarch_linear_trained = {
            "times": times_linear,
            "n": n,
            "k": k,
            "layers": str(layers),
            "model": "monarch_linear_trained",
            "inference_time": inference_time,
            "transformation_time": transformation_time,
            "accuracy": acc,
        }
        #####################################################
        
        layers = [
            nn.Linear(2, n),
            *[MonarchLinear(n, n) for _ in range(k)],
            nn.Linear(n, 1),
        ]

        monarch_model = SimplePINN(layers)
        optimizer = torch.optim.Adam(monarch_model.parameters(), lr=0.001)
        train_losses_monarch, _, _, test_losses_monarch, times_monarch = train(monarch_model, train_dataloader, optimizer, device, 0, test_dataloader)
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
        }

        print("Accuracy monarch_model :", accuracy(monarch_model, test_dataloader, device))

        #####################################################

        layers = [
        nn.Linear(2, n),
        *[STEAMLinear(n, n) for _ in range(k)],
        nn.Linear(n, 1),
        ]

        steam_model = SimplePINN(layers)
        optimizer = torch.optim.Adam(steam_model.parameters(), lr=0.001)
        train_losses_steam, _, _, test_losses_steam, times_steam = train(steam_model, train_dataloader, optimizer, device, 10, test_dataloader)
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
        }


        ############################################################















        ###########################################################

        #save_result(save_path, dict_linear)
        #save_result(save_path, dict_monarch_linear_trained)
        #save_result(save_path, dict_monarch)
        #save_result(save_path, dict_stem)
        print("Results saved")
        