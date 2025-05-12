import os

M = range(1, 51)
K = range(1, 5)
models = ["monarch", "steam"]
equations = ["simple", "burger"]
epoch = 250
rep = 10

if os.path.exists("params"):
    os.remove("params")

for (m, k, model, equation) in [(m, k, model, equation) for m in M for k in K for model in models for equation in equations]:
    with open(f"experiments/any/params", "a") as f:
        for r in range(rep):
            f.write(f"{equation} -m {m} -k {k} -e {epoch} -r {r} -f {model}\n")

