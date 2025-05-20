import torch
from torch import nn

import timeit
import time

from core.layers.monarch import MonarchLinear
from core.layers.steam import STEAMLinear

from examples.any.model import AnyPINN
from examples.any.list_models import list_models

import string
import random
import os

from core.utils.save import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lettres = string.ascii_letters
alea = ''.join(random.choice(lettres) for _ in range(10))
print(f"Séquence aléatoire générée: {alea}")


save_path = os.path.join("results", "timeit", f'results_{alea}.json')

M = range(1, 40)
number = 1_000

print("device = ", device)


p_model = list_models["burger"]

for k in range(1, 10):    
    t_linear = []
    t_monarch = []
    t_steam = []
    for m in M:
        print(f"m = {m}, k = {k}")
        n = m*m

        x = torch.randn(10, 2).to(device)
        
        #linear = nn.Linear(n, n).to(device)
        #monarch = MonarchLinear(n, n).to(device)
        #steam = STEAMLinear(n, n).to(device)

        layers = [
                nn.Linear(2, n),
                *[nn.Linear(n, n) for _ in range(k)],
                nn.Linear(n, 1),
            ]        
        linear_model = AnyPINN(layers, p_model["pde"]).to(device)

        layers = [
            nn.Linear(2, n),
            *[MonarchLinear(n, n) for _ in range(k)],
            nn.Linear(n, 1),
        ]
        monarch_model = AnyPINN(layers, p_model["pde"]).to(device)

        #layers = [
        #    nn.Linear(2, n),
        #    *[STEAMLinear(n, n) for _ in range(k)],
        #    nn.Linear(n, 1),
        #]
        #for layer in layers[1:k+1]:
        #    layer.P0 = layer.Pbar
        #    layer.P2 = layer.Pbar
        #    
        #steam_model = torch.compile(AnyPINN(layers, p_model["pde"])).to(device)

        def timeit_linear():
            linear_model(x)
        def timeit_monarch():
            monarch_model(x)
        #def timeit_steam():
        #    steam_model(x)

        t_l = timeit.timeit(timeit_linear, number=number, timer=time.process_time)
        t_m = timeit.timeit(timeit_monarch, number=number, timer=time.process_time)
        #t_s = timeit.timeit(timeit_steam, number=number, timer=time.process_time)

        t_linear.append(t_l)
        t_monarch.append(t_m)
        #t_steam.append(t_s)

        print("t_linear", t_l)
        print("t_monarch", t_m)
        #print("t_steam", t_s)

    dict_to_save = {
        "m": M,
        "k": k,
        "t_linear": t_linear,
        "t_monarch": t_monarch
        #"t_steam": t_steam
    }

    save_result(save_path, dict_to_save )



from matplotlib import pyplot as plt
plt.plot(M, t_linear, label="Linear")
plt.plot(M, t_monarch, label="Monarch")
#plt.plot(M, t_steam, label="STEAM")
plt.xlabel("m (n = m*m)")
plt.ylabel("Time (s)")
plt.legend()
plt.grid(True)
plt.savefig("timeit.png")



