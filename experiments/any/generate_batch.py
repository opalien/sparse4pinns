import os

M = range(5, 40, 4)
K = [1, 4] #range(1, 5)
models = ["linear"] #monarch", "steam"]
equations = ["burger"]
epoch = 1000
rep = 1
LOG = [0, 1]

if os.path.exists("experiments/any/params"):
    os.remove("experiments/any/params")

for r in range(rep):
    for k in K:
        for m in M:
            for model in models:
                for equation in equations:
                    for log in LOG:
                        with open(f"experiments/any/params", "a") as f: 
                            f.write(f"{equation} -m {m} -k {k} -e {epoch} -r 1 -f {model} -l {log}\n")
