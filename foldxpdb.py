import pandas as pd
import os
import subprocess
import shutil
from multiprocessing import Pool
import glob


def run_foldx_buildmodel(pdb_base, mutation, id_name, repair_dir, mut_dir, foldx_exe, txt_dir):
    # Create subdir for this task to avoid conflicts
    task_dir = os.path.join(mut_dir, id_name)
    os.makedirs(task_dir, exist_ok=True)
    log_path = os.path.join(task_dir, f"{id_name}_log.txt")  # Log in subdir

    repaired_pdb = os.path.join(repair_dir, f"{pdb_base}.pdb")  # Use direct name without _Repair suffix
    if not os.path.exists(repaired_pdb):
        error_msg = f"Repaired PDB not found: {repaired_pdb}"
        print(error_msg)
        with open(log_path, 'w') as log_file:
            log_file.write(error_msg + '\n')
        return

    # Copy repaired PDB to task_dir
    mut_pdb_copy = os.path.join(task_dir, os.path.basename(repaired_pdb))
    shutil.copy(repaired_pdb, mut_pdb_copy)

    # Load pre-existing mutant file (from txt_dir, with individual_list_ prefix)
    mutant_file_src = os.path.join(txt_dir, f"individual_list_{id_name}.txt")
    if not os.path.exists(mutant_file_src):
        error_msg = f"Pre-existing mutant file not found: {mutant_file_src}"
        print(error_msg)
        with open(log_path, 'w') as log_file:
            log_file.write(error_msg + '\n')
        return

    # Copy mutant file to task_dir (keep original name, as it starts with individual_list_)
    mutant_file = os.path.join(task_dir, os.path.basename(mutant_file_src))
    shutil.copy(mutant_file_src, mutant_file)

    # Run FoldX BuildModel in task_dir
    cmd = [
        foldx_exe,
        '--command=BuildModel',
        f'--pdb={os.path.basename(repaired_pdb)}',  # e.g., HM_2NZ9.pdb
        f'--mutant-file={os.path.basename(mutant_file)}',  # Use local filename in task_dir
        '--numberOfRuns=3',
        '--out-pdb=true',
        '--order=_USERDEFINED'  # Add this to specify mutation order, as per FoldX recommendations
    ]
    try:
        result = subprocess.run(cmd, cwd=task_dir, check=True, capture_output=True, text=True)
        with open(log_path, 'w') as log_file:
            log_file.write(f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\n")
            log_file.write(f"Used mutant file: {mutant_file_src}\n")

        # Find and rename the first output mutated PDB (e.g., HM_2NZ9_1_*.pdb -> id_name.pdb)
        output_pattern = os.path.join(task_dir, f"{pdb_base}_1_*.pdb")
        output_files = glob.glob(output_pattern)
        if output_files:
            first_output = sorted(output_files)[0]  # Take the first run (e.g., _1_0.pdb)
            new_name = os.path.join(mut_dir, f"{id_name}.pdb")  # Move to main mut_dir
            shutil.move(first_output, new_name)
            print(f"Mutated PDB generated and renamed: {new_name}")
            with open(log_path, 'a') as log_file:
                log_file.write(f"Success: Renamed {first_output} to {new_name}\n")
        else:
            error_msg = f"Output mutated PDB not found for {id_name} (pattern: {output_pattern})"
            print(error_msg)
            with open(log_path, 'a') as log_file:
                log_file.write(error_msg + '\n')
    except subprocess.CalledProcessError as e:
        error_msg = f"Error running BuildModel for {id_name}: STDOUT={e.stdout} STDERR={e.stderr}"
        print(error_msg)
        with open(log_path, 'w') as log_file:
            log_file.write(error_msg + '\n')
            log_file.write(f"Used mutant file: {mutant_file_src}\n")


def generate_mutations_batch(csv_path, repair_dir, mut_dir, foldx_exe, txt_dir, num_processes=16, limit=None):
    """
    Batch generate mutated PDB files using FoldX BuildModel in parallel.
    Uses pre-existing txt files in txt_dir based on ID (with individual_list_ prefix).
    Added 'limit' parameter to process only the first N rows for testing.
    """
    os.makedirs(mut_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    # Limit to first 'limit' rows for testing if specified
    if limit is not None:
        df = df.head(limit)

    # No need to process Mutation here since txt files are pre-generated

    # Use multiprocessing Pool to parallelize
    with Pool(processes=num_processes) as pool:
        args = [(row['PDB'], row.get('Mutation', ''), row['ID'], repair_dir, mut_dir, foldx_exe, txt_dir) for _, row in df.iterrows()]
        pool.starmap(run_foldx_buildmodel, args)


# Example usage - adjust paths as needed
if __name__ == "__main__":
    csv_path = '/home/zhao/gwc/NEW3/data/S641/S641.csv'
    repair_dir = '/home/zhao/gwc/NEW3/data/S641/wt'
    mut_dir = '/home/zhao/gwc/NEW3/data/S641/mut'
    foldx_exe = '/home/zhao/gwc/MODEL/FOLDX/foldx'  # Full path to foldx executable
    txt_dir = '/home/zhao/gwc/NEW3/data/S641/txt'  # Directory with pre-generated txt files

    # Process all data with 80 parallel processes
    generate_mutations_batch(csv_path, repair_dir, mut_dir, foldx_exe, txt_dir, num_processes=40, limit=None)