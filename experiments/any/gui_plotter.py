import tkinter as tk
from tkinter import ttk, filedialog
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
import glob
from typing import Any, Dict, List, Tuple, Optional, DefaultDict, Set

# --- Global Variables ---
# To store data for the currently selected experiment
# nodes_data: {child_id: record_from_json_line}
nodes_data: Dict[int, Dict[str, Any]] = {}
# adj: {parent_id: [child_ids]}
adjacency_list: DefaultDict[int, List[int]] = defaultdict(list)
# compiled_paths: {path_desc_tuple: {"train": [], "test": [], "cumulative_time": []}}
compiled_paths: Dict[Tuple[Any, ...], Dict[str, List[float]]] = {}
experiment_n_val: Optional[int] = None
experiment_k_val: Optional[int] = None
EXPERIMENT_FILES_DIR: str = os.path.join("results", "any")


# --- Data Loading and Processing ---
def get_experiment_files_with_details() -> List[Tuple[str, str]]:
    """Scans the experiment directory for result files and extracts N, K, and Scale info."""
    pattern: str = os.path.join(EXPERIMENT_FILES_DIR, "results_*.json")
    files_basenames: List[str] = sorted([os.path.basename(f) for f in glob.glob(pattern)])
    
    detailed_files_info: List[Tuple[str, str]] = []

    for basename in files_basenames:
        filepath: str = os.path.join(EXPERIMENT_FILES_DIR, basename)
        n_val: Optional[int] = None
        k_val: Optional[int] = None
        epoch_values: List[int] = []
        first_valid_line_processed: bool = False

        try:
            with open(filepath, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record: Dict[str, Any] = json.loads(line)
                        required_keys: List[str] = ["n", "k", "epoch"] # Min keys needed for this summary
                        if not all(key in record for key in required_keys):
                            continue # Skip lines not useful for summary

                        if not first_valid_line_processed:
                            n_val = record.get('n')
                            k_val = record.get('k')
                            first_valid_line_processed = True
                        
                        epoch_val_candidate = record.get('epoch')
                        if isinstance(epoch_val_candidate, int):
                            epoch_values.append(epoch_val_candidate)
                        elif isinstance(epoch_val_candidate, float) and epoch_val_candidate.is_integer():
                            epoch_values.append(int(epoch_val_candidate))
                            
                    except json.JSONDecodeError:
                        # print(f"Warning: JSON decode error in {basename} for line: {line.strip()}")
                        continue # Skip malformed JSON lines
            
            scale_str: str
            if not epoch_values:
                scale_str = "N/A"
            elif len(set(epoch_values)) == 1:
                scale_str = "Lin"
            else:
                scale_str = "Log"

            n_display: str = str(n_val) if n_val is not None else "?"
            k_display: str = str(k_val) if k_val is not None else "?"
            
            display_name: str = f"{basename} (N={n_display}, K={k_display}, Scale={scale_str})"
            detailed_files_info.append((display_name, basename))

        except Exception as e:
            print(f"Error processing file {basename} for details: {e}")
            display_name = f"{basename} (Error reading details)"
            detailed_files_info.append((display_name, basename))
            
    return detailed_files_info


def load_and_process_experiment_data(file_basename: str) -> bool:
    """Loads and processes data from a selected experiment file."""
    global nodes_data, adjacency_list, compiled_paths, experiment_n_val, experiment_k_val
    nodes_data.clear()
    adjacency_list.clear()
    compiled_paths.clear()
    experiment_n_val = None
    experiment_k_val = None

    filepath: str = os.path.join(EXPERIMENT_FILES_DIR, file_basename)

    if not os.path.exists(filepath):
        print(f"Error: File not found {filepath}")
        return False

    try:
        with open(filepath, 'r') as f:
            first_line_processed: bool = False
            for line in f:
                if not line.strip():
                    continue
                record: Dict[str, Any] = json.loads(line)

                # Validate essential keys for execution_tree.py output
                required_keys: List[str] = ["child_id", "parent_id", "factor", "optimizer", "epoch",
                                 "train_losses", "test_losses", "times", "n", "k"]
                if not all(key in record for key in required_keys):
                    print(f"Skipping malformed record (missing keys): {record.get('child_id', 'Unknown ID')}")
                    continue
                
                child_id = record.get('child_id')
                parent_id = record.get('parent_id')

                if not isinstance(child_id, int) or not isinstance(parent_id, int):
                    print(f"Skipping record with non-integer child_id or parent_id: {child_id}, {parent_id}")
                    continue

                nodes_data[child_id] = record
                adjacency_list[parent_id].append(child_id)

                if not first_line_processed:
                    n_candidate = record.get('n')
                    k_candidate = record.get('k')
                    if isinstance(n_candidate, int):
                        experiment_n_val = n_candidate
                    if isinstance(k_candidate, int):
                        experiment_k_val = k_candidate
                    first_line_processed = True
        
        if not nodes_data:
            print("No valid data loaded from the file.")
            return False

    except FileNotFoundError:
        print(f"Error: Experiment file '{filepath}' not found.")
        return False
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{filepath}'.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        return False

    # Reconstruct paths
    # Initial call for DFS: from children of node 0 (root of trained models)
    if 0 in adjacency_list: # Node 0 is the pre-training state
        for start_node_id in adjacency_list[0]:
            _dfs_reconstruct(start_node_id, [], [], [], ())
    else:
        print("Warning: No children found for root node 0. This might be an empty or non-standard file (e.g. not from ExecutionTree).")
        # This might happen if the JSON is empty or doesn't start with parent_id 0.

    return True

def _dfs_reconstruct(current_node_id: int, 
                     path_train_losses_acc: List[float], 
                     path_test_losses_acc: List[float], 
                     path_segment_times_list_acc: List[List[float]], # list of lists (each inner list is raw times for a segment)
                     path_description_acc: Tuple[Any, ...]):
    """
    Recursive helper to reconstruct paths and collate data.
    """
    if current_node_id not in nodes_data:
        # This can happen if a parent_id points to a child_id not in the file (incomplete data)
        print(f"Warning: Node ID {current_node_id} referenced but not found in nodes_data. Path reconstruction might be incomplete.")
        return

    node_info: Dict[str, Any] = nodes_data[current_node_id]

    current_segment_train_losses: List[float] = node_info.get('train_losses', [])
    current_segment_test_losses: List[float] = node_info.get('test_losses', [])
    current_segment_step_times: List[float] = node_info.get('times', []) # These are raw times for each epoch in this segment

    # Extend data for the current path
    new_path_train_losses: List[float] = path_train_losses_acc + current_segment_train_losses
    new_path_test_losses: List[float] = path_test_losses_acc + current_segment_test_losses
    new_path_segment_times_list: List[List[float]] = path_segment_times_list_acc + [current_segment_step_times]
    
    # Make sure factor and optimizer are strings for the description
    factor_str: str = str(node_info.get('factor', 'N/A'))
    optimizer_str: str = str(node_info.get('optimizer', 'N/A'))
    epoch_val: int = node_info.get('epoch', 0)
    new_path_description: Tuple[Any, ...] = path_description_acc + ((factor_str, optimizer_str, epoch_val),)

    if current_node_id not in adjacency_list or not adjacency_list[current_node_id]: # Leaf node for this path
        # Path complete, process accumulated segment times into one final cumulative time list
        final_cumulative_times: List[float] = []
        current_total_time_offset: float = 0.0
        for segment_times_raw in new_path_segment_times_list:
            if not segment_times_raw: continue # Skip empty segments
            segment_cumulative: np.ndarray[Any, Any] = np.cumsum(np.array(segment_times_raw, dtype=float))
            final_cumulative_times.extend( (current_total_time_offset + segment_cumulative).tolist() )
            if final_cumulative_times:
                 current_total_time_offset = final_cumulative_times[-1]
        
        compiled_paths[new_path_description] = {
            "train": new_path_train_losses,
            "test": new_path_test_losses,
            "cumulative_time": final_cumulative_times
        }
    else:
        for child_id in adjacency_list[current_node_id]:
            _dfs_reconstruct(child_id, 
                             new_path_train_losses, 
                             new_path_test_losses, 
                             new_path_segment_times_list, 
                             new_path_description)

# --- Plotting Logic ---
def format_path_description(desc_tuple: Tuple[Any, ...]) -> str:
    parts: List[str] = []
    for factor, optimizer, epochs in desc_tuple:
        parts.append(f"{str(factor)[:1].upper()}({str(optimizer)[:3]},{epochs})")
    return " -> ".join(parts)

def plot_paths_data(show_time_plot: bool, 
                      fig_train: Figure, canvas_train: FigureCanvasTkAgg, 
                      fig_test: Figure, canvas_test: FigureCanvasTkAgg, 
                      fig_time: Figure, canvas_time: FigureCanvasTkAgg):
    fig_train.clear()
    fig_test.clear()
    ax_train: Axes = fig_train.add_subplot(111)
    ax_test: Axes = fig_test.add_subplot(111)
    
    ax_time: Optional[Axes] = None
    if show_time_plot:
        fig_time.clear()
        ax_time = fig_time.add_subplot(111)
    else:
        fig_time.clear() # Clear even if hidden
        # ax_time remains None

    if not compiled_paths:
        for ax, cv, name in [(ax_train, canvas_train, "Train"), (ax_test, canvas_test, "Test")]:
            ax.text(0.5, 0.5, f"No paths to plot for {name} Loss. Select an experiment file or check data.", ha='center', va='center', color='red'); cv.draw()
        if show_time_plot and ax_time:
            ax_time.text(0.5, 0.5, "No paths to plot for Time. Select an experiment file.", ha='center', va='center', color='red'); canvas_time.draw()
        elif not show_time_plot:
             canvas_time.draw() # Ensure canvas is clean if hidden
        return

    # print(f"Plotting {len(compiled_paths)} paths. N={experiment_n_val}, K={experiment_k_val}, ShowTime={show_time_plot}")
    
    has_data_to_plot_train: bool = False
    has_data_to_plot_test: bool = False
    has_data_to_plot_time: bool = False

    for path_desc_tuple, path_data in compiled_paths.items():
        label: str = format_path_description(path_desc_tuple)
        
        train_losses: List[float] = path_data.get("train", [])
        test_losses: List[float] = path_data.get("test", [])
        cumulative_times: List[float] = path_data.get("cumulative_time", [])

        if train_losses:
            epochs_axis: np.ndarray[Any, Any] = np.arange(1, len(train_losses) + 1)
            ax_train.plot(epochs_axis, train_losses, label=label, alpha=0.8)
            has_data_to_plot_train = True
        
        if test_losses:
            epochs_axis_test: np.ndarray[Any, Any] = np.arange(1, len(test_losses) + 1) # Renamed to avoid conflict if types differ
            ax_test.plot(epochs_axis_test, test_losses, label=label, alpha=0.8)
            has_data_to_plot_test = True

        if show_time_plot and ax_time and cumulative_times:
            epochs_axis_time: np.ndarray[Any, Any] = np.arange(1, len(cumulative_times) + 1) # Renamed
            ax_time.plot(epochs_axis_time, cumulative_times, label=label, alpha=0.8)
            has_data_to_plot_time = True

    # --- Setup Plot Axes and Legends ---
    common_title_suffix: str = f"(N={experiment_n_val}, K={experiment_k_val})" if experiment_n_val is not None else ""

    if has_data_to_plot_train:
        ax_train.set_title(f"Train Loss {common_title_suffix}")
        ax_train.set_xlabel("Total Epochs"); ax_train.set_ylabel("Train Loss")
        ax_train.legend(loc='best', fontsize='small'); ax_train.set_yscale('log'); ax_train.grid(True, which="both", ls="--")
    else: ax_train.text(0.5, 0.5, "No Train Loss data for selected paths.", ha='center', va='center')
    
    if has_data_to_plot_test:
        ax_test.set_title(f"Test Loss {common_title_suffix}")
        ax_test.set_xlabel("Total Epochs"); ax_test.set_ylabel("Test Loss")
        ax_test.legend(loc='best', fontsize='small'); ax_test.set_yscale('log'); ax_test.grid(True, which="both", ls="--")
    else: ax_test.text(0.5, 0.5, "No Test Loss data for selected paths.", ha='center', va='center')

    if show_time_plot and ax_time:
        if has_data_to_plot_time:
            ax_time.set_title(f"Cumulative Time {common_title_suffix}")
            ax_time.set_xlabel("Total Epochs"); ax_time.set_ylabel("Cumulative Time (s)")
            ax_time.legend(loc='best', fontsize='small'); ax_time.grid(True, which="both", ls="--", axis='y') # Linear scale for time
        else: ax_time.text(0.5, 0.5, "No Time data for selected paths.", ha='center', va='center')
    
    canvas_train.draw()
    canvas_test.draw()
    canvas_time.draw()


# --- GUI Setup ---
class PlotterApp:
    def __init__(self, root_window: tk.Tk):
        self.root: tk.Tk = root_window
        self.root.title("Execution Tree Experiment Plotter")
        self.root.geometry("1300x900") 

        # Map for display name to actual basename
        self.experiment_file_map: Dict[str, str] = {}

        # --- Controls Frame ---
        controls_frame: ttk.Frame = ttk.Frame(self.root, padding="10")
        controls_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(controls_frame, text="Experiment File:").pack(side=tk.LEFT, padx=(0,5))
        self.file_var: tk.StringVar = tk.StringVar()
        self.file_combobox: ttk.Combobox = ttk.Combobox(controls_frame, textvariable=self.file_var, state="readonly", width=60) # Increased width for longer names
        self.file_combobox.pack(side=tk.LEFT, padx=5)
        self.file_combobox.bind("<<ComboboxSelected>>", self.on_file_select)

        reload_button: ttk.Button = ttk.Button(controls_frame, text="Scan/Reload Files", command=self.reload_experiment_files_action)
        reload_button.pack(side=tk.LEFT, padx=10)
        
        # Info display (N, K)
        self.info_label_var: tk.StringVar = tk.StringVar(value="N: --, K: --")
        ttk.Label(controls_frame, textvariable=self.info_label_var).pack(side=tk.LEFT, padx=10)

        # Checkbox for showing time plot
        self.show_time_var: tk.BooleanVar = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls_frame, text="Show Time Plot", variable=self.show_time_var, command=self.trigger_plot_redraw).pack(side=tk.LEFT, padx=10)

        # --- Plotting Frames ---
        # Top frame for Train and Test Loss plots
        self.top_plots_frame: ttk.Frame = ttk.Frame(self.root, padding="5")
        self.top_plots_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        train_loss_frame: ttk.Frame = ttk.Frame(self.top_plots_frame)
        train_loss_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))
        self.fig_train: Figure = Figure(figsize=(6, 4.5), dpi=100) 
        self.canvas_train: FigureCanvasTkAgg = FigureCanvasTkAgg(self.fig_train, master=train_loss_frame)
        self.canvas_train.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        test_loss_frame: ttk.Frame = ttk.Frame(self.top_plots_frame)
        test_loss_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5,0))
        self.fig_test: Figure = Figure(figsize=(6, 4.5), dpi=100) 
        self.canvas_test: FigureCanvasTkAgg = FigureCanvasTkAgg(self.fig_test, master=test_loss_frame)
        self.canvas_test.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.bottom_plot_frame: ttk.Frame = ttk.Frame(self.root, padding="5")
        self.fig_time: Figure = Figure(figsize=(12, 3.5), dpi=100)
        self.canvas_time: FigureCanvasTkAgg = FigureCanvasTkAgg(self.fig_time, master=self.bottom_plot_frame)
        self.canvas_time.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.reload_experiment_files_action() # Initial scan for files
        self.trigger_plot_redraw() # Initial plot (will show "select file" if none chosen)

    def reload_experiment_files_action(self) -> None:
        print("Scanning for experiment files...")
        self.experiment_file_map.clear()
        detailed_files: List[Tuple[str, str]] = get_experiment_files_with_details()
        
        display_names: List[str] = []
        for display_name, basename in detailed_files:
            self.experiment_file_map[display_name] = basename
            display_names.append(display_name)
            
        self.file_combobox['values'] = display_names
        
        if display_names:
            current_selection_display_name: str = self.file_var.get()
            # Check if current selection is valid among new display names
            if current_selection_display_name not in display_names:
                self.file_var.set(display_names[0]) # Default to first file's display name
            # on_file_select will be triggered if self.file_var.set changes it,
            # or we call it explicitly to ensure data for the (potentially new) default is loaded.
            self.on_file_select() 
        else:
            self.file_var.set("")
            self.info_label_var.set("N: --, K: -- (No files found)")
            nodes_data.clear(); adjacency_list.clear(); compiled_paths.clear()
            global experiment_n_val, experiment_k_val
            experiment_n_val = None; experiment_k_val = None
            self.trigger_plot_redraw()
        print(f"Found {len(display_names)} files.")

    def on_file_select(self, event: Optional[tk.Event] = None) -> None:
        selected_display_name: str = self.file_var.get()
        actual_basename: Optional[str] = self.experiment_file_map.get(selected_display_name)

        if actual_basename:
            print(f"Loading data for: {actual_basename} (Selected as: {selected_display_name})")
            if load_and_process_experiment_data(actual_basename):
                self.info_label_var.set(f"N: {experiment_n_val if experiment_n_val is not None else 'N/A'}, K: {experiment_k_val if experiment_k_val is not None else 'N/A'}")
                print(f"Data loaded. N={experiment_n_val}, K={experiment_k_val}. Found {len(compiled_paths)} paths.")
            else:
                self.info_label_var.set("N: --, K: -- (Error loading)")
                print("Failed to load or process data.")
                # Clear potentially stale plot data on error
                nodes_data.clear(); adjacency_list.clear(); compiled_paths.clear() 
        else:
            if selected_display_name: # If there's a display name but no mapping (e.g. error entry)
                print(f"No actual file mapped for selection: {selected_display_name}")
            self.info_label_var.set("N: --, K: -- (No valid file selected)")
            # Clear global data and plots if no valid file is effectively selected
            nodes_data.clear(); adjacency_list.clear(); compiled_paths.clear()
            # Access global experiment_n_val and experiment_k_val correctly
            global experiment_n_val_global_ref_on_select, experiment_k_val_global_ref_on_select
            experiment_n_val_global_ref_on_select = None 
            experiment_k_val_global_ref_on_select = None
            # Update the actual global variables if this is the intended behavior upon invalid selection
            # This part might need clarification: are we clearing the globally displayed N/K if selection is bad?
            # For now, assume the info_label_var is the primary display and it's already set to "--"

        self.trigger_plot_redraw()

    def trigger_plot_redraw(self) -> None:
        show_time: bool = self.show_time_var.get()

        if show_time:
            self.bottom_plot_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        else:
            self.bottom_plot_frame.pack_forget()
        
        plot_paths_data(show_time, 
                        self.fig_train, self.canvas_train, 
                        self.fig_test, self.canvas_test, 
                        self.fig_time, self.canvas_time)

# Define global references for N and K to be modified by on_file_select if needed
# These are distinct from the module-level experiment_n_val / experiment_k_val used by plotting logic
experiment_n_val_global_ref_on_select: Optional[int] = None
experiment_k_val_global_ref_on_select: Optional[int] = None

if __name__ == "__main__":
    # Ensure the results directory exists, or file scanning might be confusing
    if not os.path.exists(EXPERIMENT_FILES_DIR):
        os.makedirs(EXPERIMENT_FILES_DIR, exist_ok=True)
        print(f"Created directory: {EXPERIMENT_FILES_DIR} as it was missing.")
        
    root = tk.Tk()
    app = PlotterApp(root)
    root.mainloop() 