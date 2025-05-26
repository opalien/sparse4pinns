import argparse
import torch
import pickle
import os
from time import sleep
import threading
import sys
parser = argparse.ArgumentParser(description="PDE one step tree solving.")
parser.add_argument("-path", type=str, help="The path to the work directory")
parser.add_argument("-edge", type=str, help="The path to the edge")
parser.add_argument("-alea", type=str, help="The alea")
args = parser.parse_args()

path = args.path
edge_path = args.edge
alea = args.alea


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def ensure_activity(stop_signal: threading.Event):
    interval = 0.05
    while not stop_signal.is_set():
        if not os.path.exists(edge_path+".using"):
            with open(edge_path+".using", "w") as _:
                pass
        stop_signal.wait(timeout=interval) # Attend l'intervalle ou le signal d'arrêt

    if os.path.exists(edge_path+".using"):
        os.remove(edge_path+".using")

if __name__ == "__main__":
    stop_event = threading.Event()
    
    #lancer ensure_activity en parallele
    ensure_activity_thread = threading.Thread(target=ensure_activity, args=(stop_event,))
    ensure_activity_thread.daemon = True
    ensure_activity_thread.start()

    #lancer les calculs
    tree = pickle.load(open(os.path.join(path, "tree.pkl"), "rb"))
    edge = pickle.load(open(edge_path, "rb"))
    print(tree.nodes)
    #edge.parent = tree.nodes[edge.parent_id]
    #edge.child = tree.nodes[edge.child_id]

    tree.set_alea(alea)
    tree.one_step(edge)

    del edge
    del tree
    os.remove(edge_path)

    # Nettoyage
    thereisedgetowork = False
    for edge in os.listdir(os.path.join(path, "edges")):
        if edge.endswith(".pkl"):
            thereisedgetowork = True
            break
    
    if not thereisedgetowork:
        tree = pickle.load(open(os.path.join(path, "tree.pkl"), "rb"))
        tree.train_dataloader = None
        tree.test_dataloader = None
        for node in tree.nodes[:-1]:
            node.pinn = None

        pickle.dump(tree, open(os.path.join(path, "tree.pkl"), "wb"))
        del tree

        


    #arreter ensure_activityquand les calculs sont finis
    os.remove(edge_path+".using")
    print("done")
    sys.exit(0)

    
    