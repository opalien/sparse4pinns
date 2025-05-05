import os

M = range(1, 51)
K = range(1, 5)
epoch = 250
rep = 10

if os.path.exists("params"):
    os.remove("params")

for (m, k) in [(m, k) for m in M for k in K]:
    with open(f"params", "a") as f:
        for r in range(rep):
            f.write(f"simple -m {m} -k {k} -e {epoch} -r 1\n")

