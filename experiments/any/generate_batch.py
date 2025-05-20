import os
import numpy as np
M = [i for i in np.linspace(5, 40, 3, dtype=int)]
K = [1, 4] #range(1, 5)
models = ["linear"] #monarch", "steam"]
equations = ["burger"]
epoch = 1000
rep = 1
LOG = [0, 1]

if os.path.exists("experiments/any/params"):
    os.remove("experiments/any/params")

for r in range(rep):
    for m in M:
        for k in K:
            for model in models:
                for equation in equations:
                    for log in LOG:
                        if log:
                            epoch = 1024
                        else:
                            epoch = 1000

                        with open(f"experiments/any/params", "a") as f: 
                            f.write(f"{equation} -m {m} -k {k} -e {epoch} -r 1 -f {model} -l {log}\n")
