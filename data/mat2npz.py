import scipy.io as sio
import numpy as np
import os

def convert_mat_to_npz(mat_file_path, npz_file_path, variable_name='H_bank'):
    """
    Loads a specific variable from a .mat file and saves it as a .npz file.

    Args:
        mat_file_path (str): Path to the input .mat file.
        npz_file_path (str): Path to the output .npz file.
        variable_name (str): The name of the variable to extract from the .mat file.
    """
    if not os.path.exists(mat_file_path):
        print(f"Error: Input file not found at {mat_file_path}")
        return

    print(f"Loading data from '{mat_file_path}'...")
    try:
        mat_contents = sio.loadmat(mat_file_path)
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return

    if variable_name not in mat_contents:
        print(f"Error: Variable '{variable_name}' not found in the .mat file.")
        print(f"Available variables: {list(mat_contents.keys())}")
        return

    data_matrix = mat_contents[variable_name]
    print(f"Successfully loaded variable '{variable_name}' with shape {data_matrix.shape}.")

    print(f"Saving data to '{npz_file_path}'...")
    try:
        # Use the same variable name as the key in the .npz file for compatibility
        np.savez_compressed(npz_file_path, **{variable_name: data_matrix})
        print(f"Conversion successful. File saved to '{npz_file_path}'.")
    except Exception as e:
        print(f"Error saving .npz file: {e}")

if __name__ == '__main__':
    # Define file paths relative to the project root
    # Note: This script assumes it is run from the project root directory
    mat_path = 'data/H_bank_64_original.mat'
    npz_path = 'data/H_bank_64.npz'
    
    convert_mat_to_npz(mat_path, npz_path)
