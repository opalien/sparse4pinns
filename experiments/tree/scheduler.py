

import torch
import torch.nn as nn
import os
import argparse
from time import sleep
import subprocess

from experiments.tree.execution_tree import ExecutionTree
from examples.any.dataset import TrainAnyDataset, TestAnyDataset
from examples.any.list_models import list_models
from examples.any.model import AnyPINN


parser = argparse.ArgumentParser(description="PDE tree solving.")
parser.add_argument("problem", help="The pde to solve")
parser.add_argument("-e", "--epoch", type=int, default=100, help="Un nombre (défaut: 100)")
parser.add_argument("-m", "--m_matrix", type=int, default=5, help="coté de la matrice (défaut: 5)")
parser.add_argument("-k", "--k_layers", type=int, default=1, help="Un nombre (défaut: 1)")
parser.add_argument("-s", "--scheduler", type=str, default="linear", help="log, linear (défaut: linear)")
parser.add_argument("-p", "--steps", type=int, default=2, help="Un nombre (défaut: 2)")
parser.add_argument("-c", "--continued", type=bool, default=False, help="Continue from last instance")

args = parser.parse_args()

epoch_max = args.epoch
m = args.m_matrix
n = m * m
k = args.k_layers
scheduler = args.scheduler
steps = args.steps
continued = args.continued

match os.cpu_count():
    case None:  torch.set_num_threads(1)
    case u:     torch.set_num_threads(u)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


work_dir = os.path.join("results", "tree", args.problem)
os.makedirs(work_dir, exist_ok=True)


MAX_CURRENT_PROCESSES = 3
list_processes = []

def slurm_check():
    global list_processes
    global MAX_CURRENT_PROCESSES

    running_count = 0
    for p in list_processes:
        if p.poll() is None:  # poll() retourne None si le processus est toujours en cours
            running_count += 1

    print(f"running_count: {running_count}")
    
    return running_count < MAX_CURRENT_PROCESSES

if __name__ == "__main__":

    if not continued:
        p_model = list_models[args.problem]
        time_bounds = p_model["bounds"][0]
        spatial_bounds = p_model["bounds"][1:]
        t_max_for_dataset = time_bounds[1]

        train_dataloader = TrainAnyDataset(
            p_model["solution"],
            n_elements=10,
            n_colloc=100, 
            shape=spatial_bounds,
            t_max=t_max_for_dataset
        ).get_dataloader(10)
            
        test_dataloader = TestAnyDataset(
            p_model["solution"],
            n_elements=10,
            shape=spatial_bounds,
            t_max=t_max_for_dataset
        ).get_dataloader(10)


        layers = [
            nn.Linear(2, n),
            *[nn.Linear(n, n) for _ in range(k)],
            nn.Linear(n, 1),
        ]    
        model = AnyPINN(layers, p_model["pde"])

        print(f"model: {model}")

        tree = ExecutionTree(epoch_max, steps, device, train_dataloader, test_dataloader, model, work_dir, scheduler=scheduler, alea="0")
        tree.one_step()
        del tree

    loop = True
    while loop:

        for dir in os.listdir(work_dir):
            if os.path.exists(os.path.join(work_dir, dir, "finished")) and os.path.exists(os.path.join(work_dir, dir, "edges")):
                # if edges is non empty
                for edge_path in os.listdir(os.path.join(work_dir, dir, "edges")):
                    if edge_path.endswith(".using"):
                        continue
                    edge_path = os.path.join(work_dir, dir, "edges", edge_path)
                    path = os.path.join(work_dir, dir)
                    # with slurm
                    # os.command("srun python experiments/tree/run.py " + edge_path) 

                    #without slurm
                    #if not os.path.exists(edge_path+".using"):
                    #    os.system(f"python -m experiments.tree.run -path {path} -edge {edge_path} -alea {str(alea)}")
                    #    alea += 1
                    #    print(f"running {edge_path}")

                    if not os.path.exists(edge_path + ".using"):
                        # Construction de la commande sous forme de liste pour Popen
                        #extract alea from edge_path
                        alea = edge_path.split("_")[-1].split(".")[0]
                        alea = dir+str(alea)
                        command_list = [
                            "python", "-m", "experiments.tree.run",
                            "-path", path,
                            "-edge", edge_path,
                            "-alea", alea
                        ]
                        # Lancement de la commande en mode non bloquant
                        while not slurm_check():
                            sleep(10.)
                        list_processes.append(subprocess.Popen(command_list))
                        print(f"Launched process for: {edge_path}")

                    else:
                        os.remove(edge_path+".using")
        
        sleep(10.)
                    
            


    
    
#    #os.makedirs(os.path.join(work_dir, tree.alea), exist_ok=True)
#    #pickle.dump(tree, open(os.path.join(work_dir, tree.alea, "tree.pkl"), "wb", ))
#
#    tree.one_step()
#
#
#    tree_forked = pickle.load(open(os.path.join(work_dir, tree.alea, "tree.pkl"), "rb"))
#    tree_forked.run()





