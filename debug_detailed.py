#!/usr/bin/env python3

import torch
import sys
sys.path.append('/home/opalien/Documents/other/29_05_25/sparse4pinns')

from examples.any.dataset import TrainAnyDataset
from examples.any.list_models import list_models
from examples.any.model import AnyPINN
from torch import nn

# Test the burger PDE loss calculation with detailed debugging
p_model = list_models["burger"]
time_bounds = p_model["bounds"][0]
spatial_bounds = p_model["bounds"][1:]
t_max_for_dataset = time_bounds[1]

# Create small dataset for testing
train_dataset = TrainAnyDataset(
    p_model["solution"],
    n_elements=5,  # Very small for debugging
    n_colloc=10, 
    shape=spatial_bounds,
    t_max=t_max_for_dataset
)

# Create small model
layers = [
    nn.Linear(2, 4),
    nn.Linear(4, 4),
    nn.Linear(4, 1),
]

model = AnyPINN(layers, p_model["pde"])

# Test with a single batch
train_dataloader = train_dataset.get_dataloader(15)  # Get all data in one batch

for i, (a, u, idx) in enumerate(train_dataloader):
    print(f"\nBatch {i}: Total size: {a.shape[0]}, Boundary data: {idx}, Collocation points: {a.shape[0] - idx}")
    
    # Forward pass
    model(a)
    
    # Check the PDE result directly for collocation points
    if idx < a.shape[0]:  # If there are collocation points
        colloc_u_pred = model.u_pred[idx:]
        colloc_a_in = model.a_in[idx:]
        
        print(f"\nCollocation debugging:")
        print(f"colloc_u_pred.shape: {colloc_u_pred.shape}")
        print(f"colloc_a_in.shape: {colloc_a_in.shape}")
        
        # Test Jacobian directly
        J = model.J_u(colloc_u_pred, colloc_a_in)
        print(f"J.shape: {J.shape}")
        print(f"J[:, :, 0].shape: {J[:, :, 0].shape}")
        print(f"J[:, :, 1:].shape: {J[:, :, 1:].shape}")
        
        # Test derivatives step by step
        du_dt = J[:, :, 0]
        du_dx = J[:, :, 1:]
        print(f"du_dt.shape before squeeze: {du_dt.shape}")
        print(f"du_dx.shape before squeeze: {du_dx.shape}")
        
        du_dt_squeezed = du_dt.squeeze(1) if du_dt.dim() > 1 else du_dt
        du_dx_squeezed = du_dx.squeeze() if du_dx.dim() > 1 else du_dx  # Use squeeze() to remove all singleton dimensions
        print(f"du_dt.shape after squeeze: {du_dt_squeezed.shape}")
        print(f"du_dx.shape after squeeze: {du_dx_squeezed.shape}")
        
        # Test Hessian
        H = model.H_u(colloc_u_pred, colloc_a_in)
        print(f"H.shape: {H.shape}")
        du_dxx = torch.diagonal(H, offset=0, dim1=-2, dim2=-1)[:, :, 1:]
        print(f"du_dxx.shape before squeeze: {du_dxx.shape}")
        du_dxx_squeezed = du_dxx.squeeze() if du_dxx.dim() > 1 else du_dxx  # Use squeeze() to remove all singleton dimensions
        print(f"du_dxx.shape after squeeze: {du_dxx_squeezed.shape}")
        
        u_squeezed = colloc_u_pred.squeeze(1)
        print(f"u.shape after squeeze: {u_squeezed.shape}")
        
        # Test PDE calculation step by step
        term1 = du_dt_squeezed
        term2 = u_squeezed * du_dx_squeezed  
        term3 = (0.01/3.14159) * du_dxx_squeezed
        
        print(f"term1.shape: {term1.shape}")
        print(f"term2.shape: {term2.shape}")
        print(f"term3.shape: {term3.shape}")
        
        # Now test PDE
        pde_result = model.pde(colloc_u_pred, colloc_a_in)
        print(f"Final PDE result shape: {pde_result.shape}")
        print(f"PDE result values (first 5): {pde_result[:5].detach().numpy()}")
        
    break  # Only test first batch
