import json
import os
import glob
from collections import defaultdict
import numpy as np

def average_experiment_results(project_root_path=".", output_filename="averaged_results.json"):
    """
    Averages train_losses, test_losses, and times from multiple results.json files
    found under project_root_path/results/tree/XXX/results.json.

    Each results.json file is expected to contain one JSON object per line,
    representing an experiment step, uniquely identified by 'child_id'.

    The averaged results are written to a new file in project_root_path.
    """
    tree_path = os.path.join(project_root_path, "results", "tree")
    search_pattern = os.path.join(tree_path, "*", "results.json")
    file_paths = glob.glob(search_pattern)

    if not file_paths:
        print(f"No results.json files found under {tree_path}")
        return

    print(f"Found {len(file_paths)} results files to process:")
    for fp in file_paths:
        print(f"  - {fp}")

    # grouped_data stores records by child_id: {child_id: [record1, record2, ...]}
    grouped_data = defaultdict(list)

    for file_path in file_paths:
        try:
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    try:
                        data = json.loads(line)
                        child_id = data.get("child_id")
                        if child_id is not None:
                            grouped_data[child_id].append(data)
                        else:
                            print(f"Warning: Missing 'child_id' in line {i+1} of file {file_path}: {line.strip()}")
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON from line {i+1} of file {file_path}: {line.strip()}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            continue

    if not grouped_data:
        print("No data successfully read from files. Averaged output will be empty.")
        return

    averaged_output_list = []

    # Process records sorted by child_id to maintain order in the output file
    for child_id in sorted(grouped_data.keys()):
        records_for_child = grouped_data[child_id]
        if not records_for_child:
            continue

        output_record = records_for_child[0].copy()

        for field_to_average in ["train_losses", "test_losses", "times"]:
            all_values = [r.get(field_to_average) for r in records_for_child]
            # Filter for non-None list values. Empty lists are kept at this stage.
            valid_values = [v for v in all_values if v is not None and isinstance(v, list)]

            if valid_values:
                try:
                    first_len = len(valid_values[0])
                    # Check if all valid lists have the same length as the first one
                    if not all(len(v_list) == first_len for v_list in valid_values):
                        print(f"Warning: Inconsistent list lengths for field '{field_to_average}' for child_id {child_id}. Setting to None.")
                        output_record[field_to_average] = None
                    else:
                        if first_len == 0: # All valid lists are empty
                            output_record[field_to_average] = []
                        else: # Non-empty lists, proceed with averaging
                            np_array_values = np.array(valid_values, dtype=float)
                            avg_values = np.mean(np_array_values, axis=0).tolist()
                            output_record[field_to_average] = avg_values
                except Exception as e:
                    print(f"Error averaging field '{field_to_average}' for child_id {child_id}: {e}. Setting to None.")
                    output_record[field_to_average] = None
            else: # No valid lists found (e.g., all were None or not lists)
                output_record[field_to_average] = None
        
        averaged_output_list.append(output_record)

    final_output_path = os.path.join(project_root_path, output_filename)
    
    try:
        with open(final_output_path, 'w') as outfile:
            for record in averaged_output_list:
                outfile.write(json.dumps(record) + '\n')
        print(f"Averaged results successfully saved to {final_output_path}")
        if not averaged_output_list and grouped_data : # grouped_data was not empty but output is
             print("Note: The output file is empty, possibly due to issues processing data fields.")
        elif not averaged_output_list:
             print("Note: The output file is empty as no processable data was found.")

    except Exception as e:
        print(f"Error writing averaged results to {final_output_path}: {e}")

if __name__ == "__main__":
    # This script assumes it's located in the project root directory 
    # (e.g., /home/opalien/Documents/other/29_05_25/sparse4pinns)
    # and run from there using `python average_experiment_results.py`.
    # The project_root_path will then be "." (current directory).
    average_experiment_results(project_root_path=".")