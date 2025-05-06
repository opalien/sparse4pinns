import fenics as fn
import numpy as np
import math
import os
import pickle
import matplotlib.pyplot as plt
from numpy.typing import NDArray
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
# Reste structurellement la même, mais sera initialisée avec des données potentiellement plus précises
class BurgerSolutionInterpolator:
    def __init__(self, times: NDArray[np.float64], spatial_coords: NDArray[np.float64], solution_values: NDArray[np.float64]):
        """
        Initialise l'interpolateur avec les données de simulation.
        Assure que les coordonnées spatiales sont triées.
        """
        if not all(isinstance(arr, np.ndarray) for arr in [times, spatial_coords, solution_values]):
            raise TypeError('Inputs times, spatial_coords, and solution_values must be NumPy arrays.')

        # Assurer que les coordonnées spatiales sont triées (essentiel pour np.interp)
        if not np.all(np.diff(spatial_coords) >= 0):
            print("Warning: Spatial coordinates were not sorted. Sorting them now.")
            sort_indices = np.argsort(spatial_coords)
            self.spatial_coords: NDArray[np.float64] = spatial_coords[sort_indices]
            # Assurer que les valeurs de solution correspondent aux coordonnées triées
            if solution_values.shape[1] == len(spatial_coords):
                 self.solution_values: NDArray[np.float64] = solution_values[:, sort_indices]
            else:
                 raise ValueError("Mismatch between spatial_coords and solution_values dimensions after attempting sort.")
        else:
            self.spatial_coords = spatial_coords
            self.solution_values = solution_values

        self.times: NDArray[np.float64] = times

        # Validations des dimensions
        if self.times.ndim != 1 or self.spatial_coords.ndim != 1 or self.solution_values.ndim != 2:
            raise ValueError(f'Incorrect dimensions: times ({self.times.ndim}), spatial_coords ({self.spatial_coords.ndim}), solution_values ({self.solution_values.ndim})')
        if self.times.shape[0] != self.solution_values.shape[0]:
            raise ValueError(f'Mismatch time steps ({self.times.shape[0]}) vs solution rows ({self.solution_values.shape[0]}).')
        if self.spatial_coords.shape[0] != self.solution_values.shape[1]:
            raise ValueError(f'Mismatch spatial points ({self.spatial_coords.shape[0]}) vs solution columns ({self.solution_values.shape[1]}).')

        # Stockage des métadonnées utiles
        self.num_time_steps: int = len(self.times)
        self.num_spatial_points: int = len(self.spatial_coords)
        self.t_min: float = self.times.min()
        self.t_max: float = self.times.max()
        self.xmin: float = self.spatial_coords.min()
        self.xmax: float = self.spatial_coords.max()

        print(f"Interpolator initialized: {self.num_time_steps} time steps ({self.t_min:.2f} to {self.t_max:.2f}), "
              f"{self.num_spatial_points} spatial points ({self.xmin:.2f} to {self.xmax:.2f})")

    def evaluate(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Évalue la solution interpolée aux points (t, x) donnés."""
        if not isinstance(points, np.ndarray):
            try:
                points = np.asarray(points, dtype=np.float64)
            except Exception as e:
                raise TypeError(f"Input 'points' must be a NumPy array or convertible, got {type(points)}. Error: {e}")

        # Gérer les entrées à point unique
        if points.ndim == 1:
            if points.shape == (2,):
                points = points.reshape(1, 2) # Traiter comme un tableau (1, 2)
            else:
                raise ValueError(f'Input \'points\' must be shape (N, 2) or (2,) for a single point, got shape {points.shape}')
        elif points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Input 'points' must be shape (N, 2), got shape {points.shape}")

        N = points.shape[0]
        results = np.empty(N)
        times_req = points[:, 0]
        coords_req = points[:, 1]

        # --- Interpolation Temporelle (Linéaire) ---
        # Trouver les indices de temps encadrant chaque temps requis
        indices = np.searchsorted(self.times, times_req, side='right')
        # Limiter les indices pour éviter les erreurs aux bords et gérer l'extrapolation constante
        indices = np.clip(indices, 1, self.num_time_steps - 1)

        idx0 = indices - 1
        idx1 = indices

        t0 = self.times[idx0]
        t1 = self.times[idx1]

        # Éviter la division par zéro si dt est très petit ou nul
        dt_interval = t1 - t0
        # Utiliser un masque pour la division sécurisée
        valid_interval = dt_interval > 1e-15 # Tolérance pour éviter la division par zéro

        # Calculer les poids pour l'interpolation linéaire temporelle
        weight1 = np.zeros_like(times_req)
        np.divide(times_req - t0, dt_interval, out=weight1, where=valid_interval)
        # Si dt_interval est trop petit, weight1 reste 0, weight0 sera 1 (on prend la valeur à t0)

        weight0 = 1.0 - weight1

        # Gérer l'extrapolation (constante aux bords)
        before_start = times_req < self.t_min
        after_end = times_req > self.t_max

        weight0[before_start] = 1.0 # Utiliser la solution initiale
        weight1[before_start] = 0.0
        idx0[before_start] = 0
        idx1[before_start] = 0 # Évite l'erreur d'indice si before_start et valid_interval sont False en même temps

        weight0[after_end] = 0.0 # Utiliser la solution finale
        weight1[after_end] = 1.0
        idx0[after_end] = self.num_time_steps - 1 # Utilise le dernier indice valide
        idx1[after_end] = self.num_time_steps - 1

        # Extraire les solutions aux temps t0 et t1
        values0 = self.solution_values[idx0, :]
        values1 = self.solution_values[idx1, :]

        # Interpoler linéairement en temps
        # Utilise le broadcasting: (N,) * (N, M) -> (N, M)
        values_t_interp = weight0[:, np.newaxis] * values0 + weight1[:, np.newaxis] * values1

        # --- Interpolation Spatiale (Linéaire) ---
        # Pour chaque temps interpolé, interpoler spatialement
        for i in range(N):
            x_req = coords_req[i]
            values_at_t = values_t_interp[i, :]
            # np.interp gère l'extrapolation constante aux bords spatiaux par défaut
            results[i] = np.interp(x_req, self.spatial_coords, values_at_t)

        return results

    def __call__(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
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

    # Obtenir les coordonnées des DoFs et les trier (important pour l'interpolateur)
    dof_coordinates = V.tabulate_dof_coordinates()
    # Pour P2, il peut y avoir des DoFs dupliqués aux mêmes coordonnées (sommets partagés)
    # Nous voulons les coordonnées uniques triées pour l'interpolation spatiale finale.
    # Cependant, la solution FEniCS est ordonnée selon les DoFs internes.
    # Nous devons récupérer les valeurs aux coordonnées uniques *après* la simulation.
    # Pour l'instant, gardons l'ordre des DoFs FEniCS, nous trierons à la fin.
    fenics_dof_indices = np.arange(V.dim()) # Indices dans l'ordre FEniCS

    # 2. Conditions aux Limites (Dirichlet u=0 aux bords)
    def boundary(x, on_boundary):
        return on_boundary
    # Utilisation de fn.Constant pour une meilleure performance
    bc = fn.DirichletBC(V, fn.Constant(0.0), boundary)

    # 3. Condition Initiale
    # Le degré de l'Expression doit être suffisant pour l'interpolation dans V (P2)
    u_0_expr = fn.Expression('-sin(DOLFIN_PI*x[0])', degree=3) # *** Changement : degree=2 -> 3 (ou plus) ***
    # u_n représente la solution au temps précédent (t_n)
    u_n = fn.interpolate(u_0_expr, V)

    # 4. Formulation Variationnelle (Crank-Nicolson)
    u = fn.Function(V)      # Solution au temps courant (t_{n+1}), l'inconnue
    v = fn.TestFunction(V)  # Fonction test

    # Terme intermédiaire pour Crank-Nicolson (moyenne temporelle)
    u_mid = 0.5 * (u + u_n) # *** NOUVEAU : Moyenne pour CN ***

    # Formulation faible : Intégrale( [(u - u_n)/dt]*v + [u_mid * grad(u_mid)]*v + [nu*grad(u_mid)]*grad(v) ) dx = 0
    # Note: u * u.dx(0) est la forme forte. La forme faible standard pour u*u' est u*u'*v
    # Pour la forme conservative (d/dx(0.5*u^2)), on peut écrire: -0.5 * u_mid**2 * v.dx(0) * dx + conditions de bord
    # Ici, on garde la forme non-conservative u*u' * v qui est commune:
    F = (
        (u - u_n) / dt * v * fn.dx                     # Dérivée temporelle
        + u_mid * u_mid.dx(0) * v * fn.dx             # Terme advectif non-linéaire (évalué à t_mid)
        + nu * fn.dot(fn.grad(u_mid), fn.grad(v)) * fn.dx # Terme diffusif (évalué à t_mid)
    ) # *** Changement : u -> u_mid dans les termes spatiaux ***

    # 5. Calcul du Jacobien pour le solveur de Newton
    J = fn.derivative(F, u) # FEniCS calcule la dérivée automatiquement

    # 6. Problème Non-linéaire Variationnel et Solveur
    problem = fn.NonlinearVariationalProblem(F, u, bc, J)
    solver = fn.NonlinearVariationalSolver(problem)

    # Ajustement des paramètres du solveur de Newton pour une meilleure précision
    solver_prm = solver.parameters["newton_solver"]
    solver_prm["relative_tolerance"] = 1e-8 # *** Changement : Tolérance plus stricte ***
    solver_prm["absolute_tolerance"] = 1e-9 # *** Changement : Tolérance plus stricte ***
    solver_prm["maximum_iterations"] = 30    # Augmenter si la convergence est difficile
    solver_prm["relaxation_parameter"] = 1.0 # Pas de sous-relaxation par défaut
    # solver_prm["linear_solver"] = "mumps" # Choisir un solveur linéaire si besoin ( MUMPS est robuste)
    print(f"Newton solver tolerances: Rel={solver_prm['relative_tolerance']:.1e}, Abs={solver_prm['absolute_tolerance']:.1e}")

    # 7. Boucle Temporelle
    times_list = [0.0]
    # Stocker les coefficients de la solution dans l'ordre des DoFs FEniCS
    solution_coeffs_list = [u_n.vector().get_local().copy()]

    t = 0.0
    print(f"Starting time stepping loop for {num_steps} steps...")
    progress_interval = max(1, num_steps // 20) # Afficher la progression ~20 fois

    for n in range(num_steps):
        t += dt

        # Résoudre pour u à t_{n+1}
        try:
            num_iter, converged = solver.solve()
            if not converged:
                 print(f"\nWARNING: Newton solver did NOT converge at step {n+1}, t={t:.4f} after {num_iter} iterations.")
                 # Que faire ici ? Arrêter ? Continuer avec la solution non convergée ?
                 # Pour l'instant, on continue mais on le signale.
        except Exception as e:
             print(f"\nERROR: FEniCS solver failed at step {n+1}, t={t:.4f}")
             print(f"Error message: {e}")
             # Peut-être sauvegarder l'état actuel pour le débogage
             # fn.File("debug_u_fail.pvd") << u
             # fn.File("debug_un_fail.pvd") << u_n
             return None # Arrêter la simulation en cas d'erreur grave

        # Mettre à jour la solution précédente pour le pas suivant
        u_n.assign(u)

        # Stocker les résultats
        times_list.append(t)
        solution_coeffs_list.append(u.vector().get_local().copy())

        # Afficher la progression
        if (n + 1) % progress_interval == 0 or n == num_steps - 1:
            print(f"  Step {n+1}/{num_steps} completed (t = {t:.4f}) - Newton iters: {num_iter}")


    simulation_time = time.time() - start_time
    print(f'FEniCS simulation finished. Total time: {simulation_time:.2f} seconds.')

    # 8. Post-traitement et Sauvegarde
    times_array = np.array(times_list)
    # solution_values_array contient les coeffs DoF ordonnés par FEniCS
    solution_values_array_fenics_order = np.vstack(solution_coeffs_list)

    # Obtenir les coordonnées uniques triées et les valeurs correspondantes
    # pour l'interpolateur qui attend des coordonnées spatiales 1D triées.
    unique_coords, unique_indices = np.unique(dof_coordinates[:, 0], return_index=True)
    # Les valeurs de solution aux coordonnées uniques triées
    solution_values_sorted_coords = solution_values_array_fenics_order[:, unique_indices]
    spatial_coords_sorted = unique_coords # Coordonnées triées

    print(f'Final data shape for interpolator: times({times_array.shape}), coords({spatial_coords_sorted.shape}), values({solution_values_sorted_coords.shape})')

    print('Creating interpolator object...')
    try:
        interpolator = BurgerSolutionInterpolator(times=times_array, spatial_coords=spatial_coords_sorted, solution_values=solution_values_sorted_coords)
    except Exception as e:
        print(f"Error creating BurgerSolutionInterpolator: {e}")
        return None

    print(f'Saving interpolator to {output_pickle_file}...')
    try:
        with open(output_pickle_file, 'wb') as f:
            pickle.dump(interpolator, f, protocol=pickle.HIGHEST_PROTOCOL) # Utiliser le protocole le plus élevé
        print('Save successful.')
        return output_pickle_file
    except Exception as e:
        print(f'Error saving pickle file to {output_pickle_file}: {e}')
        return None

# --- Point d'entrée principal ---
if __name__ == '__main__':

    # Utiliser un nom de fichier reflétant les paramètres P2 et CN
    pickle_filename = 'burger_solution.pkl'
    pickle_path = os.path.join(SOLVER_DIR, pickle_filename)

    print('--- Step 1: Run simulation with P2 elements and Crank-Nicolson ---')
    # Utilisation des paramètres par défaut de la fonction (nx=500, num_steps=500)
    saved_path = solve_burger(
        T=1.0,
        num_steps=1000, # Ajustez si nécessaire
        nx=1000,        # Ajustez si nécessaire
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

            # Points de test pour l'évaluation
            test_points = np.array([
                [0.15, 0.5],   # Intérieur du domaine, temps précoce
                [0.5, 0.0],    # Milieu du domaine, temps intermédiaire
                [0.9, -0.8],   # Près du bord gauche, temps tardif
                [1.0, 0.2],    # Temps final
                # Points potentiellement problématiques pour l'interpolateur
                [0.0, 0.1],    # Temps initial exact
                [1.0, -1.0],   # Coin espace-temps (bord)
                [0.6, 1.0],    # Coin espace-temps (bord)
                # Points hors limites pour tester l'extrapolation
                [1.1, 0.1],    # Temps > T_max
                [-0.1, 0.3],   # Temps < T_min
                [0.6, -1.1],   # x < x_min
                [0.7, 1.2]     # x > x_max
            ])

            print('\nEvaluating test points:')
            results = loaded_interpolator(test_points) # Utilise __call__

            print("-" * 40)
            for i, p in enumerate(test_points):
                print(f'  u(t={p[0]:.3f}, x={p[1]:.3f}) = {results[i]:.6f}')
            print("-" * 40)

            # Optionnel : Visualisation rapide d'un profil temporel
            if loaded_interpolator:
                 plt.figure(figsize=(10, 6))
                 final_time_index = loaded_interpolator.num_time_steps -1
                 plt.plot(loaded_interpolator.spatial_coords,
                          loaded_interpolator.solution_values[0,:],
                          'b-', label=f't = {loaded_interpolator.times[0]:.2f} (Initial)')
                 plt.plot(loaded_interpolator.spatial_coords,
                          loaded_interpolator.solution_values[final_time_index // 2,:],
                          'g--', label=f't = {loaded_interpolator.times[final_time_index // 2]:.2f} (Mid)')
                 plt.plot(loaded_interpolator.spatial_coords,
                          loaded_interpolator.solution_values[final_time_index,:],
                          'r:', label=f't = {loaded_interpolator.times[final_time_index]:.2f} (Final)')
                 plt.xlabel('x')
                 plt.ylabel('u(t, x)')
                 plt.title(f'Burger Solution Profiles (P2, CN, nx={loaded_interpolator.num_spatial_points-1}, dt={(loaded_interpolator.t_max)/loaded_interpolator.num_time_steps:.2e})')
                 plt.legend()
                 plt.grid(True)
                 plt.ylim(-1.1, 1.1) # Ajustez si nécessaire
                 plt.show()


        except Exception as e:
            print('\nError during loading or testing the pickled object:')
            print(e)
            import traceback
            traceback.print_exc()
    else:
        print('\nSimulation did not run successfully or did not save the interpolator.')

    print('\nScript finished.')