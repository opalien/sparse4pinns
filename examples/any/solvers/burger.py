import fenics as fn
import numpy as np
import math
import os
import pickle
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from typing import Optional, Tuple
import sys

SOLVER_DIR = os.path.join('examples', 'any', 'solvers')

class BurgerSolutionInterpolator:
    pass

class _PatchedUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if module == '__main__' and name == 'BurgerSolutionInterpolator':
            return BurgerSolutionInterpolator
        try:
            return super().find_class(module, name)
        except AttributeError:
            __import__(module)
            return super().find_class(module, name)

def load_solution_interpolator(pickle_path: str) -> Optional[BurgerSolutionInterpolator]:
    try:
        with open(pickle_path, 'rb') as f:
            unpickler = _PatchedUnpickler(f)
            interpolator = unpickler.load()
        if not isinstance(interpolator, BurgerSolutionInterpolator):
            print(f'Warning: Loaded object from {pickle_path} is not of type BurgerSolutionInterpolator. Got {type(interpolator)}')
        return interpolator
    except FileNotFoundError:
        print(f'Error: Pickle file not found at {pickle_path}')
        return None
    except (pickle.UnpicklingError, AttributeError, ImportError, EOFError, NameError) as e:
        print(f"Error loading pickle file '{pickle_path}': {e}")
        if '__main__' in str(e) and 'BurgerSolutionInterpolator' in str(e):
            print("This seems to be the common '__main__' pickle loading issue.")
        return None
    except Exception as e:
        print(f'An unexpected error occurred while loading {pickle_path}: {e}')
        return None

class BurgerSolutionInterpolator:

    def __init__(self, data_dir: Optional[str]=None, times: Optional[NDArray[np.float64]]=None, spatial_coords: Optional[NDArray[np.float64]]=None, solution_values: Optional[NDArray[np.float64]]=None):
        if data_dir is not None:
            times_path = os.path.join(data_dir, 'times.npy')
            spatial_coords_path = os.path.join(data_dir, 'spatial_coords.npy')
            solution_values_path = os.path.join(data_dir, 'solution_values.npy')
            if not all(os.path.exists(p) for p in [times_path, spatial_coords_path, solution_values_path]):
                raise FileNotFoundError(f'NumPy data files not found in {data_dir}')
            self.times: NDArray[np.float64] = np.load(times_path)
            self.spatial_coords: NDArray[np.float64] = np.load(spatial_coords_path)
            self.solution_values: NDArray[np.float64] = np.load(solution_values_path)
        elif times is not None and spatial_coords is not None and solution_values is not None:
            if not all(isinstance(arr, np.ndarray) for arr in [times, spatial_coords, solution_values]):
                raise TypeError('Inputs must be NumPy arrays if data_dir is not provided.')
            self.times = times
            self.spatial_coords = spatial_coords
            self.solution_values = solution_values
            if not np.all(np.diff(self.spatial_coords) >= 0):
                sort_indices = np.argsort(self.spatial_coords)
                self.spatial_coords = self.spatial_coords[sort_indices]
                self.solution_values = self.solution_values[:, sort_indices]
        else:
            raise ValueError('Provide either data_dir or all three NumPy arrays (times, spatial_coords, solution_values).')
        if self.times.ndim != 1 or self.spatial_coords.ndim != 1 or self.solution_values.ndim != 2:
            raise ValueError('Incorrect dimensions for input arrays.')
        if self.times.shape[0] != self.solution_values.shape[0]:
            raise ValueError(f'Mismatch between number of time steps ({self.times.shape[0]}) and rows in solution values ({self.solution_values.shape[0]}).')
        if self.spatial_coords.shape[0] != self.solution_values.shape[1]:
            raise ValueError(f'Mismatch between number of spatial coordinates ({self.spatial_coords.shape[0]}) and columns in solution values ({self.solution_values.shape[1]}).')
        self.num_time_steps: int = len(self.times)
        self.num_spatial_points: int = len(self.spatial_coords)
        self.t_min: float = self.times.min()
        self.t_max: float = self.times.max()
        self.xmin: float = self.spatial_coords.min()
        self.xmax: float = self.spatial_coords.max()

    def evaluate(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        if not isinstance(points, np.ndarray):
            try:
                points = np.asarray(points, dtype=np.float64)
            except Exception as e:
                raise TypeError(f"Input 'points' must be a NumPy array or convertible, got {type(points)}. Conversion error: {e}")
        if points.ndim == 1:
            if points.shape == (2,):
                points = points.reshape(1, 2)
            else:
                raise ValueError(f'Input \'points\' must be shape (N, 2) or (2,) for a single point, got shape {points.shape}')
        elif points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Input 'points' must be shape (N, 2), got shape {points.shape}")
        N = points.shape[0]
        results = np.empty(N)
        times_req = points[:, 0]
        coords_req = points[:, 1]
        indices = np.searchsorted(self.times, times_req, side='right')
        indices = np.clip(indices, 1, self.num_time_steps - 1)
        idx0 = indices - 1
        idx1 = indices
        t0 = self.times[idx0]
        t1 = self.times[idx1]
        dt_interval = t1 - t0
        valid_interval = dt_interval > 1e-15
        weight1 = np.zeros_like(times_req)
        np.divide(times_req - t0, dt_interval, out=weight1, where=valid_interval)
        weight0 = 1.0 - weight1
        before_start = times_req < self.t_min
        after_end = times_req > self.t_max
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
        values_t_interp = weight0[:, np.newaxis] * values0 + weight1[:, np.newaxis] * values1
        for i in range(N):
            x_req = coords_req[i]
            values_at_t = values_t_interp[i, :]
            results[i] = np.interp(x_req, self.spatial_coords, values_at_t)
        return results

    def __call__(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.evaluate(points)

def solve_burger(T: float=1.0, num_steps: int=50, nu: float=0.01 / math.pi, nx: int=100, output_pickle_file: str='burger_interpolator.pkl') -> Optional[str]:
    dt = T / num_steps
    output_dir = os.path.dirname(output_pickle_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'Created directory: {output_dir}')
    mesh = fn.IntervalMesh(nx, -1.0, 1.0)
    V = fn.FunctionSpace(mesh, 'P', 1)
    spatial_coords = V.tabulate_dof_coordinates().flatten()
    sort_indices = np.argsort(spatial_coords)
    spatial_coords_sorted = spatial_coords[sort_indices]

    def boundary(x, on_boundary):
        return on_boundary
    bc = fn.DirichletBC(V, fn.Constant(0.0), boundary)
    u_0_expr = fn.Expression('-sin(DOLFIN_PI*x[0])', degree=2)
    u_n = fn.interpolate(u_0_expr, V)
    u = fn.Function(V)
    v = fn.TestFunction(V)
    F = (u - u_n) / dt * v * fn.dx + u * u.dx(0) * v * fn.dx + nu * fn.dot(fn.grad(u), fn.grad(v)) * fn.dx
    J = fn.derivative(F, u)
    problem = fn.NonlinearVariationalProblem(F, u, bc, J)
    solver = fn.NonlinearVariationalSolver(problem)
    time_steps_list = [0.0]
    initial_coeffs = u_n.vector().get_local()[sort_indices]
    solution_coeffs_list = [initial_coeffs.copy()]
    t = 0.0
    print(f'Starting FEniCS simulation: T={T}, dt={dt}, nx={nx}, nu={nu:.4f}')
    for n in range(num_steps):
        t += dt
        solver.solve()
        u_n.assign(u)
        time_steps_list.append(t)
        current_coeffs = u.vector().get_local()[sort_indices]
        solution_coeffs_list.append(current_coeffs.copy())
    print('FEniCS simulation finished.')
    times_array = np.array(time_steps_list)
    solution_values_array = np.vstack(solution_coeffs_list)
    print('Creating interpolator object...')
    interpolator = BurgerSolutionInterpolator(times=times_array, spatial_coords=spatial_coords_sorted, solution_values=solution_values_array)
    print(f'Saving interpolator to {output_pickle_file}...')
    try:
        with open(output_pickle_file, 'wb') as f:
            pickle.dump(interpolator, f)
        print('Save successful.')
        return output_pickle_file
    except Exception as e:
        print(f'Error saving pickle file: {e}')
        return None
if __name__ == '__main__':
    pickle_path = os.path.join(SOLVER_DIR, 'burger_solution.pkl')
    print('--- Step 1: Run simulation and save interpolator ---')
    saved_path = solve_burger(T=1.0, num_steps=10000, nx=10000, output_pickle_file=pickle_path)
    if saved_path and os.path.exists(saved_path):
        print('\n--- Step 2: Load and use the interpolator ---')
        try:
            print(f'Loading from {saved_path} using patched loader...')
            loaded_interpolator = load_solution_interpolator(saved_path)
            if loaded_interpolator is None:
                raise RuntimeError('Failed to load interpolator.')
            print('Load successful.')
            test_points = np.array([[0.15, 0.5], [0.5, 0.0], [0.9, -0.8], [1.1, 0.1], [0.6, -1.1]])
            print('\nEvaluating test points:')
            results = loaded_interpolator(test_points)
            for i, p in enumerate(test_points):
                print(f'  u(t={p[0]:.2f}, x={p[1]:.2f}) = {results[i]:.6f}')
        except Exception as e:
            print('\nError during loading or testing the pickled object:')
            print(e)
            import traceback
            traceback.print_exc()
    else:
        print('\nSimulation did not save the interpolator or the file does not exist.')
    print('\nScript finished.')