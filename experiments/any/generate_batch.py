import os

M = range(50)
K = range(4)
epoch = 250
rep = 10

if os.path.exists("params.txt"):
    os.remove("params.txt")

for (m, k) in [(m, k) for m in M for k in K]:
    with open(f"params", "a") as f:
        f.write(f"simple -m {m} -k {k} -e {epoch} -r {rep}\n")

