import matplotlib.pyplot as plt
import json
import os
import numpy as np
from typing import List, Dict, Any, Hashable
import statistics

from core.utils.save import load_results


##### PARAMS #####

save_path = os.path.join("results", "simple", "results_merged.json")
data = load_results(save_path)

save_fig_path = os.path.join("results", "simple", "plots")

models_trained = ["linear", "monarch", "stem"]
models_from_linear = ["monarch_linear_trained", "steam_linear_trained"]

K = [1]
N = [m*m for m in range(5, 40)]


###### Recup ######



data_projected = {model: {
    "best_epochs" : [],
    "best_accuracies" : [],
    "best_cumul_times" : [],
    "cumul_time_totals" : [],
    "n" : [],
    "k" : [],
    "inferences_time" : [],
} for model in models_trained}


for one_train in data:


    model = one_train["model"]
    if model not in models_trained:
        continue
    
    if one_train["n"] == 25:
        continue

    if model == "stem":
        print("n = ", one_train["n"])

    best_epoch = np.argmin(one_train["test_losses"])
    best_accuracy = one_train["test_losses"][best_epoch]
    best_cumul_time = sum(one_train["times"][:best_epoch+1])

    cumul_time_total = sum(one_train["times"])
    n = one_train["n"]
    k = one_train["k"]

    inference_time = one_train["inference_time"]


    data_projected[model]["best_epochs"].append(best_epoch)
    data_projected[model]["best_accuracies"].append(best_accuracy)
    data_projected[model]["best_cumul_times"].append(best_cumul_time)
    data_projected[model]["cumul_time_totals"].append(cumul_time_total)
    data_projected[model]["n"].append(n)
    data_projected[model]["k"].append(k)
    data_projected[model]["inferences_time"].append(inference_time)

#####


data_projected_converted = {model: {
    "n" : [],
    "k" : [],
    "inference_times" : [],
    "transformation_times" : [],
    "accuracies" : [],
    "cumul_time_totals" : [],
} for model in models_from_linear}

####### Accuracy VS N ########

for model in data_projected.keys():
    plt.scatter(data_projected[model]["n"], data_projected[model]["best_accuracies"], label=model)
plt.xlabel("N")
plt.ylabel("Accuracy")
plt.title(f"Accuracy VS N")
plt.legend()
plt.yscale('log')
plt.savefig(f"{save_fig_path}/accuracy_vs_n.png")
plt.show()
plt.clf()
plt.close()
plt.cla()




####### Accuracy VS Time ########

for model in data_projected.keys():
    plt.scatter(data_projected[model]["best_cumul_times"], data_projected[model]["best_accuracies"], label=model)
plt.xlabel("Cumul Time")
plt.ylabel("Accuracy")
plt.title(f"Accuracy VS Cumul Time")
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.savefig(f"{save_fig_path}/accuracy_vs_cumul_time.png")
plt.clf()
plt.close()
plt.cla()


######## Time VS N ########

for model in data_projected.keys():
    plt.scatter(data_projected[model]["n"], data_projected[model]["cumul_time_totals"], label=model)
plt.xlabel("N")
plt.ylabel("Total Training Time")
plt.title(f"Total Training Time VS N")
plt.legend()
plt.yscale('log')
plt.savefig(f"{save_fig_path}/time_vs_n.png")
plt.clf()
plt.close()
plt.cla()


######## argmax(test_model) VS N ########

for model in data_projected.keys():
    plt.scatter(data_projected[model]["n"], data_projected[model]["best_epochs"], label=model)
plt.xlabel("N")
plt.ylabel("Best Epoch (argmin Test Loss)")
plt.title("Best Epoch vs N")
plt.legend()
# plt.yscale('log')
plt.savefig(f"{save_fig_path}/best_epoch_vs_n.png")
plt.clf()
plt.close()
plt.cla()


######## inference_time VS N ########

for model in data_projected.keys():
    plt.scatter(data_projected[model]["n"], data_projected[model]["inferences_time"], label=model)
plt.xlabel("N")
plt.ylabel("Inference Time (s)")
plt.title("Inference Time vs N")
plt.legend()
plt.yscale('log')
plt.savefig(f"{save_fig_path}/inference_time_vs_n.png")
plt.clf()
plt.close()
plt.cla()


######## Accuracy VS inference_time ########

for model in data_projected.keys():
    plt.scatter(data_projected[model]["inferences_time"], data_projected[model]["best_accuracies"], label=model)
plt.xlabel("Inference Time (s)")
plt.ylabel("Accuracy (Best Test Loss)")
plt.title("Accuracy vs Inference Time")
plt.legend()
plt.yscale('log') 
plt.xscale('log')
plt.savefig(f"{save_fig_path}/accuracy_vs_inference_time.png")
plt.clf()
plt.close()
plt.cla()
