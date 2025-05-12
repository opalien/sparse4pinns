import os

M = range(5, 51)
K = range(1, 5)
models = ["monarch", "steam"]
equations = ["burger"]
epoch = 1000
rep = 5

if os.path.exists("experiments/any/params"):
    os.remove("experiments/any/params")

for (m, k, model, equation) in [(m, k, model, equation) for m in M for k in K for model in models for equation in equations]:
    with open(f"experiments/any/params", "a") as f:
        for r in range(rep):
            f.write(f"{equation} -m {m} -k {k} -e {epoch} -r 1 -f {model}\n")

