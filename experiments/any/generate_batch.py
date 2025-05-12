import os

M = range(5, 51)
K = range(1, 5)
models = ["monarch", "steam"]
equations = ["burger"]
epoch = 1000
rep = 1

if os.path.exists("experiments/any/params"):
    os.remove("experiments/any/params")

for r in range(rep):
    for k in K:
        for m in M:
            for model in models:
                for equation in equations:
                    with open(f"experiments/any/params", "a") as f: 
                        f.write(f"{equation} -m {m} -k {k} -e {epoch} -r 1 -f {model}\n")
