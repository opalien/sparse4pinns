#!/usr/bin/env python3

import torch
import sys
sys.path.append('/home/opalien/Documents/other/29_05_25/sparse4pinns')

from examples.any.dataset import TrainAnyDataset
from examples.any.list_models import list_models
from examples.any.model import AnyPINN
from torch import nn

# Test the burger PDE loss calculation
p_model = list_models["burger"]
time_bounds = p_model["bounds"][0]
spatial_bounds = p_model["bounds"][1:]
t_max_for_dataset = time_bounds[1]

# Create small dataset for testing
train_dataset = TrainAnyDataset(
    p_model["solution"],
    n_elements=10,  # Very small for debugging
    n_colloc=20, 
    shape=spatial_bounds,
    t_max=t_max_for_dataset
)

print(f"Number of elements (boundary data): {len(train_dataset.elements)}")
print(f"Number of collocation points: {len(train_dataset.colloc)}")

# Create small model
layers = [
    nn.Linear(2, 4),
    nn.Linear(4, 4),
    nn.Linear(4, 1),
]

model = AnyPINN(layers, p_model["pde"])

# Test with a single batch
train_dataloader = train_dataset.get_dataloader(30)  # Get all data in one batch

for i, (a, u, idx) in enumerate(train_dataloader):
    print(f"\nBatch {i}: Total size: {a.shape[0]}, Boundary data: {idx}, Collocation points: {a.shape[0] - idx}")
    
    # Forward pass
    model(a)
    
    # Calculate loss
    loss = model.loss(u, idx)
    
    print(f"Total Loss: {loss.item():.6e}")
    print(f"Data Loss: {model.get_data_loss().item():.6e}")
    print(f"PDE Loss: {model.get_pde_loss().item():.6e}")
    
    # Check the PDE result directly for collocation points
    if idx < a.shape[0]:  # If there are collocation points
        colloc_u_pred = model.u_pred[idx:]
        colloc_a_in = model.a_in[idx:]
        
        print(f"\nCollocation points: {colloc_u_pred.shape[0]}")
        print(f"Collocation inputs shape: {colloc_a_in.shape}")
        print(f"Collocation outputs shape: {colloc_u_pred.shape}")
        
        # Test PDE directly
        pde_result = model.pde(colloc_u_pred, colloc_a_in)
        print(f"PDE result shape: {pde_result.shape}")
        print(f"PDE result values (first 5): {pde_result[:5].detach().numpy()}")
        print(f"PDE result mean: {pde_result.mean().item():.6e}")
        print(f"PDE result std: {pde_result.std().item():.6e}")
        
        # Manual PDE loss calculation
        manual_pde_loss = torch.nn.functional.mse_loss(pde_result, torch.zeros_like(pde_result))
        print(f"Manual PDE loss: {manual_pde_loss.item():.6e}")
    
    break  # Only test first batch
