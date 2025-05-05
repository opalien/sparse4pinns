from core.models.pinn import PINN
from core.datasets.pinn_dataset import PINNDataloader


from torch.optim import Optimizer
from torch.nn import Module
import torch
from torch import Tensor


import time 


def accuracy(model: Module, test_loader: PINNDataloader, device: torch.device) -> float:
    model.eval()
    model.to(device)
    total_MSE: float = 0.0
    num_batches: int = 0
    with torch.no_grad():
        for i, (a, u, idx) in enumerate(test_loader): # type: ignore
            
            a: Tensor = a.float().to(device)
            u: Tensor = u.float().to(device)
    

            model(a)
            MSE = model.loss(u)

            total_MSE += MSE.item()
            num_batches += 1
    
    return total_MSE / num_batches if num_batches > 0 else 0.0


def train_one_epoch(model: PINN, train_loader: PINNDataloader, optimizer: Optimizer, device: torch.device):
    model.train()
    model.to(device)

    total_loss: float = 0.0
    total_pde_loss: float = 0.0
    total_data_loss: float = 0.0

    num_batches: int = 0
    for i, (a, u, idx) in enumerate(train_loader): # type: ignore
        a: Tensor = a.float().to(device)
        u: Tensor = u.float().to(device)
        optimizer.zero_grad()

        model(a)

        loss = model.loss(u, idx)
        pde_loss = model.get_pde_loss()
        data_loss = model.get_pde_loss()

        
        if torch.isnan(loss) or torch.isinf(loss):
            if torch.isnan(pde_loss) or torch.isinf(pde_loss):
                raise ValueError(f"Warning: Invalid PDE loss detected: {pde_loss.item()}. Skipping batch.")
            if torch.isnan(data_loss) or torch.isinf(data_loss):
                raise ValueError(f"Warning: Invalid data loss detected: {data_loss.item()}. Skipping batch.")
            raise ValueError(f"Warning: Invalid loss detected: {loss.item()}. Skipping batch.")
            
            
        
        if loss.backward() != None: # type: ignore
            raise ValueError(f"Warning: backward loss invalid: {loss.item()}. Skipping batch.")
        
        optimizer.step()

        total_loss += loss.item()
        total_pde_loss += pde_loss.item()
        total_data_loss += data_loss.item()
        num_batches += 1


    return (total_loss / num_batches, total_pde_loss / num_batches, total_data_loss / num_batches) if num_batches > 0 else (0.0, 0.0, 0.0)

    

def train(model: PINN, train_loader: PINNDataloader, optimizer: Optimizer, device: torch.device, epochs: int, test_loader: PINNDataloader | None = None, verbose: bool = True):
    model.to(device)

    train_losses: list[float] = []
    pde_losses: list[float] = []
    data_losses: list[float] = []
    times: list[float] = []

    test_losses: list[float] | None = [] if test_loader is not None else None


    for epoch in range(epochs):
        t0 = time.time()
        trail_loss, pde_loss, data_loss = train_one_epoch(model, train_loader, optimizer, device)
        t1 = time.time()
        times.append(t1 - t0)
        train_losses.append(trail_loss)
        pde_losses.append(pde_loss)
        data_losses.append(data_loss)


        if test_losses is not None and test_loader is not None:
            test_losses.append(accuracy(model, test_loader, device))

        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Data Loss: {data_losses[-1]:.4f}, PDE Loss: {pde_losses[-1]:.4f}",  end="")
            if test_losses is not None:
                print(f", Test Loss: {test_losses[-1]:.4f}", end="")
            print()

    return train_losses, pde_losses, data_losses, test_losses, times
    