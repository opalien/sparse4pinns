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

# Désactiver la journalisation FEniCS trop verbeuse si souhaité
fn.set_log_level(fn.LogLevel.WARNING) # Ou ERROR

SOLVER_DIR = os.path.join('examples', 'any', 'solvers')

# --- Début : Classes pour le chargement Pickle ---
# Ces classes restent inchangées, elles gèrent le chargement correct
# de l'objet BurgerSolutionInterpolator, même si défini dans __main__
class BurgerSolutionInterpolator:
    # La définition de cette classe sera placée plus bas,
    # mais on la déclare ici pour que _PatchedUnpickler la connaisse.
    pass

class _PatchedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__' and name == 'BurgerSolutionInterpolator':
            # Redirige vers la définition actuelle de la classe
            return BurgerSolutionInterpolator
        try:
            return super().find_class(module, name)
        except AttributeError:
            # Tente d'importer le module si non trouvé directement
            __import__(module)
            return super().find_class(module, name)

def load_solution_interpolator(pickle_path: str) -> Optional['BurgerSolutionInterpolator']:
    try:
        with open(pickle_path, 'rb') as f:
            # Utilise notre Unpickler patché
            unpickler = _PatchedUnpickler(f)
            interpolator = unpickler.load()
        # Vérifie si l'objet chargé est bien du type attendu
        if not isinstance(interpolator, BurgerSolutionInterpolator):
            print(f'Warning: Loaded object from {pickle_path} is not of type BurgerSolutionInterpolator. Got {type(interpolator)}')
            # Retourne None ou lève une erreur selon la politique souhaitée
            # return None
            raise TypeError(f'Loaded object is not a BurgerSolutionInterpolator: {type(interpolator)}')
        return interpolator
    except FileNotFoundError:
        print(f'Error: Pickle file not found at {pickle_path}')
        return None
    except (pickle.UnpicklingError, AttributeError, ImportError, EOFError, NameError) as e:
        print(f"Error loading pickle file '{pickle_path}': {e}")
        if '__main__' in str(e) and 'BurgerSolutionInterpolator' in str(e):
            print("This seems to be the common '__main__' pickle loading issue, ensure the class definition is available.")
        return None
    except Exception as e:
        print(f'An unexpected error occurred while loading {pickle_path}: {e}')
        return None
# --- Fin : Classes pour le chargement Pickle ---


# --- Classe BurgerSolutionInterpolator ---
class BurgerSolutionInterpolator:
    def __init__(self, times: torch.Tensor, spatial_coords: torch.Tensor, solution_values: torch.Tensor):
        """
        Initialise l'interpolateur avec les données de simulation.
        Assure que les coordonnées spatiales sont triées.
        """
        if not all(isinstance(arr, torch.Tensor) for arr in [times, spatial_coords, solution_values]):
            raise TypeError('Inputs times, spatial_coords, and solution_values must be PyTorch Tensors.')

        # Assurer que les coordonnées spatiales sont triées (essentiel pour interpolation)
        # torch.diff produces a tensor of length n-1. Add a positive value to ensure all diffs are checked.
        if not torch.all(torch.diff(spatial_coords) >= -1e-9): # Allow for small negative due to float precision before sort
            print("Warning: Spatial coordinates were not sorted or had issues. Sorting them now.")
            sort_indices = torch.argsort(spatial_coords)
            self.spatial_coords: torch.Tensor = spatial_coords[sort_indices]
            # Assurer que les valeurs de solution correspondent aux coordonnées triées
            if solution_values.shape[1] == spatial_coords.shape[0]:
                 self.solution_values: torch.Tensor = solution_values[:, sort_indices]
            else:
                 raise ValueError("Mismatch between spatial_coords and solution_values dimensions after attempting sort.")
        else:
            self.spatial_coords = spatial_coords
            self.solution_values = solution_values

        self.times: torch.Tensor = times
        self.device = self.times.device # Store device for later use
        self.dtype = self.times.dtype   # Store dtype

        # Validations des dimensions
        if self.times.ndim != 1 or self.spatial_coords.ndim != 1 or self.solution_values.ndim != 2:
            raise ValueError(f'Incorrect dimensions: times ({self.times.ndim}), spatial_coords ({self.spatial_coords.ndim}), solution_values ({self.solution_values.ndim})')
        if self.times.shape[0] != self.solution_values.shape[0]:
            raise ValueError(f'Mismatch time steps ({self.times.shape[0]}) vs solution rows ({self.solution_values.shape[0]}).')
        if self.spatial_coords.shape[0] != self.solution_values.shape[1]:
            raise ValueError(f'Mismatch spatial points ({self.spatial_coords.shape[0]}) vs solution columns ({self.solution_values.shape[1]}).')

        # Stockage des métadonnées utiles
        self.num_time_steps: int = self.times.shape[0]
        self.num_spatial_points: int = self.spatial_coords.shape[0]
        self.t_min: float = torch.min(self.times).item()
        self.t_max: float = torch.max(self.times).item()
        self.xmin: float = torch.min(self.spatial_coords).item()
        self.xmax: float = torch.max(self.spatial_coords).item()

        print(f"Interpolator initialized: {self.num_time_steps} time steps ({self.t_min:.2f} to {self.t_max:.2f}), "
              f"{self.num_spatial_points} spatial points ({self.xmin:.2f} to {self.xmax:.2f})")

    def _torch_interp1d(self, x_query: torch.Tensor, xp: torch.Tensor, fp_matrix: torch.Tensor) -> torch.Tensor:
        """
        Vectorized 1D linear interpolation similar to np.interp.
        x_query: (N,) tensor of x values to interpolate.
        xp: (M,) tensor of x data points, sorted.
        fp_matrix: (N, M) tensor of y data points (each row corresponds to an x_query's profile).
        Returns: (N,) tensor of interpolated values.
        """
        N = x_query.shape[0]
        M = xp.shape[0]
        results = torch.empty_like(x_query, dtype=self.dtype, device=self.device)

        # Ensure xp has at least one point
        if M == 0:
            raise ValueError("xp (spatial_coords) must not be empty.")
        if M == 1: # All points interpolate to the single value in fp_matrix
            return fp_matrix[:, 0]


        # Handle extrapolation
        left_extrap_mask = x_query <= xp[0]
        results[left_extrap_mask] = fp_matrix[left_extrap_mask, 0]

        right_extrap_mask = x_query >= xp[-1]
        results[right_extrap_mask] = fp_matrix[right_extrap_mask, M - 1]
        
        # Interpolation for points in between
        interp_mask = ~(left_extrap_mask | right_extrap_mask)

        if torch.any(interp_mask):
            x_curr = x_query[interp_mask]
            fp_curr = fp_matrix[interp_mask, :]

            # Find indices i such that xp[i] <= x_curr < xp[i+1]
            # searchsorted(xp, x_curr, side='right') gives k s.t. all xp[j < k] < x_curr and all xp[j >= k] >= x_curr
            # So, the index for the left point xp[i] is k-1.
            right_indices = torch.searchsorted(xp, x_curr, side='right')
            
            # Clamp to avoid issues if x_curr is exactly xp[-1] (handled by right_extrap_mask) or xp[0]
            # For interpolation, left_indices must be in [0, M-2]
            left_indices = torch.clamp(right_indices - 1, 0, M - 2)
            # right_indices corresponding to left_indices+1
            clamped_right_indices = left_indices + 1

            xp_left = xp[left_indices]
            xp_right = xp[clamped_right_indices]
            
            # Gather fp values corresponding to left and right indices for each point in x_curr
            # fp_curr is (num_interp_points, M)
            # left_indices is (num_interp_points,)
            # We need to select elements from each row of fp_curr based on indices in left_indices
            fp_left = fp_curr[torch.arange(fp_curr.shape[0], device=self.device), left_indices]
            fp_right = fp_curr[torch.arange(fp_curr.shape[0], device=self.device), clamped_right_indices]

            denom = xp_right - xp_left
            
            # Where denom is zero (e.g. xp has duplicate values), result is fp_left
            # This also handles if x_curr falls exactly on xp_left
            ratio = torch.zeros_like(denom, dtype=self.dtype, device=self.device)
            valid_denom_mask = denom > 1e-9 # Avoid division by zero or tiny denominators
            
            ratio[valid_denom_mask] = (x_curr[valid_denom_mask] - xp_left[valid_denom_mask]) / denom[valid_denom_mask]
            # Ensure ratio is between 0 and 1 for stability if not perfectly handled by masks
            ratio = torch.clamp(ratio, 0.0, 1.0)

            interp_values = fp_left + ratio * (fp_right - fp_left)
            results[interp_mask] = interp_values
            
        return results

    def evaluate(self, points: torch.Tensor) -> torch.Tensor:
        """Évalue la solution interpolée aux points (t, x) donnés."""
        if not isinstance(points, torch.Tensor):
            try:
                points = torch.as_tensor(points, dtype=self.dtype, device=self.device)
            except Exception as e:
                raise TypeError(f"Input 'points' must be a PyTorch Tensor or convertible, got {type(points)}. Error: {e}")

        original_device = points.device
        points = points.to(device=self.device, dtype=self.dtype)

        # Gérer les entrées à point unique
        if points.ndim == 1:
            if points.shape == (2,):
                points = points.reshape(1, 2) # Traiter comme un tableau (1, 2)
            else:
                raise ValueError(f'Input \'points\' must be shape (N, 2) or (2,) for a single point, got shape {points.shape}')
        elif points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Input 'points' must be shape (N, 2), got shape {points.shape}")

        N = points.shape[0]
        # results = torch.empty(N, dtype=self.dtype, device=self.device) # Will be created by _torch_interp1d
        times_req = points[:, 0]
        coords_req = points[:, 1]

        # --- Interpolation Temporelle (Linéaire) ---
        indices = torch.searchsorted(self.times, times_req, side='right')
        indices = torch.clip(indices, 1, self.num_time_steps - 1)

        idx0 = indices - 1
        idx1 = indices

        t0 = self.times[idx0]
        t1 = self.times[idx1]

        dt_interval = t1 - t0
        valid_interval = dt_interval > 1e-15

        weight1 = torch.zeros_like(times_req, dtype=self.dtype, device=self.device)
        # Perform division only where valid_interval is true
        safe_dt_interval = torch.where(valid_interval, dt_interval, torch.ones_like(dt_interval))
        weight1 = torch.where(valid_interval, (times_req - t0) / safe_dt_interval, weight1)

        weight0 = 1.0 - weight1

        before_start = times_req < self.times[0] # Use actual first time step value
        after_end = times_req > self.times[-1]   # Use actual last time step value

        weight0[before_start] = 1.0
        weight1[before_start] = 0.0
        idx0[before_start] = 0
        idx1[before_start] = 0 

        weight0[after_end] = 0.0
        weight1[after_end] = 1.0
        idx0[after_end] = self.num_time_steps - 1
        idx1[after_end] = self.num_time_steps - 1

        values0 = self.solution_values[idx0, :]
        values1 = self.solution_values[idx1, :]

        values_t_interp = weight0.unsqueeze(1) * values0 + weight1.unsqueeze(1) * values1

        # --- Interpolation Spatiale (Linéaire) ---
        # Vectorized call to the new interpolation function
        results = self._torch_interp1d(coords_req, self.spatial_coords, values_t_interp)
        
        return results.to(original_device)


    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """Permet d'appeler l'objet directement pour l'évaluation."""
        return self.evaluate(points)


# --- Fonction de Résolution de Burgers (Améliorée) ---
def solve_burger(T: float=1.0, num_steps: int=500, nu: float=0.01 / math.pi, nx: int=500, output_pickle_file: str='burger_interpolator_p2_cn.pkl') -> Optional[str]:
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