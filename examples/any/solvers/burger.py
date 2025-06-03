import fenics as fn
import torch
import numpy as np
import math
import os
import pickle
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import sys
import time # Pour mesurer le temps

from ..solvers.burger_loader import *

# Désactiver la journalisation FEniCS trop verbeuse si souhaité
fn.set_log_level(fn.LogLevel.WARNING) # Ou ERROR

SOLVER_DIR = os.path.join('examples', 'any', 'solvers')





# --- Fonction de Résolution de Burgers (Améliorée) ---
def solve_burger(T: float=1.0, num_steps: int=500, nu: float=0.01 / math.pi, nx: int=500, output_pickle_file: str='burger_solution.pkl') -> Optional[str]:
    """
    Résout l'équation de Burgers 1D visqueuse en utilisant FEniCS avec des améliorations:
    - Éléments P2 Lagrange pour une meilleure précision spatiale.
    - Schéma de Crank-Nicolson pour une précision temporelle d'ordre 2.
    - Tolérances du solveur non linéaire plus strictes.

    Args:
        T: Temps final de la simulation.
        num_steps: Nombre de pas de temps.
        nu: Viscosité cinématique.
        nx: Nombre d'intervalles dans le maillage spatial (nx+1 points).
        output_pickle_file: Chemin vers le fichier pickle de sortie pour l'interpolateur.

    Returns:
        Le chemin vers le fichier pickle sauvegardé, ou None en cas d'erreur.
    """
    start_time = time.time()
    dt = T / num_steps
    output_dir = os.path.dirname(output_pickle_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f'Created directory: {output_dir}')
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return None

    print(f'\n--- Starting FEniCS Simulation ---')
    print(f'Parameters: T={T}, dt={dt:.4e}, nx={nx}, nu={nu:.4f}')
    print(f'Method: P2 Elements, Crank-Nicolson Time Stepping')

    # 1. Maillage et Espace Fonctionnel (P2 Éléments)
    mesh = fn.IntervalMesh(nx, -1.0, 1.0)
    V = fn.FunctionSpace(mesh, 'P', 2) # *** Changement : P1 -> P2 ***
    print(f"Function space: P{V.ufl_element().degree()}, Number of DoFs: {V.dim()}")

    dof_coordinates = V.tabulate_dof_coordinates() # This is a NumPy array

    # 2. Conditions aux Limites (Dirichlet u=0 aux bords)
    def boundary(x, on_boundary):
        return on_boundary
    bc = fn.DirichletBC(V, fn.Constant(0.0), boundary)

    # 3. Condition Initiale
    u_0_expr = fn.Expression('-sin(DOLFIN_PI*x[0])', degree=3)
    u_n = fn.interpolate(u_0_expr, V)

    # 4. Formulation Variationnelle (Crank-Nicolson)
    u = fn.Function(V)
    v = fn.TestFunction(V)
    u_mid = 0.5 * (u + u_n)
    F = (
        (u - u_n) / dt * v * fn.dx
        + u_mid * u_mid.dx(0) * v * fn.dx
        + nu * fn.dot(fn.grad(u_mid), fn.grad(v)) * fn.dx
    )
    J = fn.derivative(F, u)
    problem = fn.NonlinearVariationalProblem(F, u, bc, J)
    solver = fn.NonlinearVariationalSolver(problem)
    solver_prm = solver.parameters["newton_solver"]
    solver_prm["relative_tolerance"] = 1e-8
    solver_prm["absolute_tolerance"] = 1e-9
    solver_prm["maximum_iterations"] = 30
    solver_prm["relaxation_parameter"] = 1.0
    print(f"Newton solver tolerances: Rel={solver_prm['relative_tolerance']:.1e}, Abs={solver_prm['absolute_tolerance']:.1e}")

    # 7. Boucle Temporelle
    times_list = [0.0]
    solution_coeffs_list = [u_n.vector().get_local().copy()] # List of NumPy arrays

    t = 0.0
    print(f"Starting time stepping loop for {num_steps} steps...")
    progress_interval = max(1, num_steps // 20)

    for n in range(num_steps):
        t += dt
        try:
            num_iter, converged = solver.solve()
            if not converged:
                 print(f"\nWARNING: Newton solver did NOT converge at step {n+1}, t={t:.4f} after {num_iter} iterations.")
        except Exception as e:
             print(f"\nERROR: FEniCS solver failed at step {n+1}, t={t:.4f}")
             print(f"Error message: {e}")
             return None
        u_n.assign(u)
        times_list.append(t)
        solution_coeffs_list.append(u.vector().get_local().copy())
        if (n + 1) % progress_interval == 0 or n == num_steps - 1:
            print(f"  Step {n+1}/{num_steps} completed (t = {t:.4f}) - Newton iters: {num_iter}")

    simulation_time = time.time() - start_time
    print(f'FEniCS simulation finished. Total time: {simulation_time:.2f} seconds.')

    # 8. Post-traitement et Sauvegarde - Convert to PyTorch Tensors here
    # Determine default dtype for PyTorch (typically float32 unless specified)
    # We use float64 for consistency with original numpy code & FEniCS precision
    tensor_dtype = torch.float64

    times_tensor = torch.tensor(times_list, dtype=tensor_dtype)
    
    # solution_coeffs_list contains NumPy arrays. Stack them first.
    solution_values_fenics_order_np = np.vstack(solution_coeffs_list)
    solution_values_tensor_fenics_order = torch.from_numpy(solution_values_fenics_order_np).to(dtype=tensor_dtype)

    # dof_coordinates is NumPy array from FEniCS
    # Get unique sorted coordinates and their original indices from the FEniCS DoF ordering
    # Using numpy for unique since it's already a numpy array and handles return_index
    unique_coords_np, unique_indices_np = np.unique(dof_coordinates[:, 0], return_index=True)
    
    spatial_coords_sorted_tensor = torch.from_numpy(unique_coords_np).to(dtype=tensor_dtype)
    # Use the NumPy indices to sort/select from the PyTorch tensor
    solution_values_sorted_coords_tensor = solution_values_tensor_fenics_order[:, torch.from_numpy(unique_indices_np)]


    print(f'Final data shape for interpolator: times({times_tensor.shape}), coords({spatial_coords_sorted_tensor.shape}), values({solution_values_sorted_coords_tensor.shape})')

    print('Creating interpolator object...')
    try:
        interpolator = BurgerSolutionInterpolator(times=times_tensor, 
                                                spatial_coords=spatial_coords_sorted_tensor, 
                                                solution_values=solution_values_sorted_coords_tensor)
    except Exception as e:
        print(f"Error creating BurgerSolutionInterpolator: {e}")
        import traceback
        traceback.print_exc()
        return None

    print(f'Saving interpolator to {output_pickle_file}...')
    try:
        with open(output_pickle_file, 'wb') as f:
            pickle.dump(interpolator, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Save successful.')
        return output_pickle_file
    except Exception as e:
        print(f'Error saving pickle file to {output_pickle_file}: {e}')
        return None

# --- Point d'entrée principal ---
if __name__ == '__main__':

    pickle_filename = 'burger_solution.pkl' # Changed name to reflect torch usage
    pickle_path = os.path.join(SOLVER_DIR, pickle_filename)

    print('--- Step 1: Run simulation with P2 elements and Crank-Nicolson ---')
    saved_path = solve_burger(
        T=1.0,
        num_steps=5000,
        nx=2000,
        nu=0.01 / math.pi,
        output_pickle_file=pickle_path
    )

    if saved_path and os.path.exists(saved_path):
        print(f'\n--- Step 2: Load and use the improved interpolator from {saved_path} ---')
        try:
            print(f'Loading using patched loader...')
            loaded_interpolator = load_solution_interpolator(saved_path)

            if loaded_interpolator is None:
                raise RuntimeError('Failed to load interpolator.')

            print('Load successful.')

            # Determine device for test_points (e.g. cuda if available, else cpu)
            # For this script, CPU is fine as FEniCS is CPU-bound
            # and interpolator is now device-aware from its stored tensors.
            # test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # print(f"Using device: {test_device} for test points")

            # Points de test pour l'évaluation (as PyTorch tensor)
            test_points_data = [
                [0.15, 0.5], [0.5, 0.0], [0.9, -0.8], [1.0, 0.2],
                [0.0, 0.1], [1.0, -1.0], [0.6, 1.0],
                [1.1, 0.1], [-0.1, 0.3], [0.6, -1.1], [0.7, 1.2]
            ]
            test_points = torch.tensor(test_points_data, dtype=torch.float64) # Use float64 for consistency

            print('\nEvaluating test points:')
            results = loaded_interpolator(test_points) # Utilise __call__

            print("-" * 40)
            # Convert points and results to CPU numpy for printing if they aren't already
            test_points_np = test_points.cpu().numpy()
            results_np = results.cpu().numpy()
            for i, p_np in enumerate(test_points_np):
                print(f'  u(t={p_np[0]:.3f}, x={p_np[1]:.3f}) = {results_np[i]:.6f}')
            print("-" * 40)

            if loaded_interpolator:
                 plt.figure(figsize=(10, 6))
                 final_time_index = loaded_interpolator.num_time_steps -1
                 # Convert to numpy for plotting
                 spatial_coords_np = loaded_interpolator.spatial_coords.cpu().numpy()
                 solution_values_np = loaded_interpolator.solution_values.cpu().numpy()
                 times_np = loaded_interpolator.times.cpu().numpy()

                 plt.plot(spatial_coords_np,
                          solution_values_np[0,:],
                          'b-', label=f't = {times_np[0]:.2f} (Initial)')
                 plt.plot(spatial_coords_np,
                          solution_values_np[final_time_index // 2,:],
                          'g--', label=f't = {times_np[final_time_index // 2]:.2f} (Mid)')
                 plt.plot(spatial_coords_np,
                          solution_values_np[final_time_index,:],
                          'r:', label=f't = {times_np[final_time_index]:.2f} (Final)')
                 plt.xlabel('x')
                 plt.ylabel('u(t, x)')
                 plt.title(f'Burger Solution Profiles (P2, CN, nx={loaded_interpolator.num_spatial_points-1}, dt={(loaded_interpolator.t_max)/loaded_interpolator.num_time_steps:.2e})')
                 plt.legend()
                 plt.grid(True)
                 plt.ylim(-1.1, 1.1)
                 plt.show()

        except Exception as e:
            print('\nError during loading or testing the pickled object:')
            print(e)
            import traceback
            traceback.print_exc()
    else:
        print('\nSimulation did not run successfully or did not save the interpolator.')

    print('\nScript finished.')