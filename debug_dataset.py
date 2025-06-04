#!/usr/bin/env python3

import torch
import sys
sys.path.append('/home/opalien/Documents/other/29_05_25/sparse4pinns')

from examples.any.dataset import TrainAnyDataset
from examples.any.list_models import list_models

# Test the burger dataset
p_model = list_models["burger"]
time_bounds = p_model["bounds"][0]
spatial_bounds = p_model["bounds"][1:]
t_max_for_dataset = time_bounds[1]

# Create dataset with same parameters as tree execution
train_dataset = TrainAnyDataset(
    p_model["solution"],
    n_elements=1000,
    n_colloc=10000, 
    shape=spatial_bounds,
    t_max=t_max_for_dataset
)

print(f"Number of elements (boundary data): {len(train_dataset.elements)}")
print(f"Number of collocation points: {len(train_dataset.colloc)}")
print(f"Total dataset size: {len(train_dataset)}")

# Test a few batches
train_dataloader = train_dataset.get_dataloader(1000)

print("\nTesting batches:")
for i, (a, u, idx) in enumerate(train_dataloader):
    print(f"Batch {i}: Total size: {a.shape[0]}, Boundary data: {idx}, Collocation points: {a.shape[0] - idx}")
    if i >= 3:  # Check first few batches
        break
