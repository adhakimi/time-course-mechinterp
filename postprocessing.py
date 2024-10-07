import pickle
import numpy as np
from networkx import DiGraph
from seaborn import heatmap
import numpy as np
import matplotlib.pyplot as plt

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # Import tqdm for progress bar


def get_head_contribution_map(gr: DiGraph, attn_contr_maps, THR=0.0):
    contr_map = np.zeros((len(attn_contr_maps), attn_contr_maps[0].shape[-1]), dtype=bool)
    for s, t, w in gr.edges.data():
        if t[:1] != "A":
            continue
        t_splitted = t.split("_")
        s_splitted = s.split("_")
        layer = int(t_splitted[0][1:])
        target_token_idx = int(t_splitted[1])
        source_token_idx = int(s_splitted[1])

        attn_layer_map = np.array(attn_contr_maps[layer][0].cpu())
        contr_map[layer] += attn_layer_map[target_token_idx, source_token_idx] > THR
    return contr_map


def get_ffn_contribution_map(gr: DiGraph, THR=0.0, LAYERS=32):
    contr_map = np.zeros(LAYERS, dtype=bool)
    for s, t, w in gr.edges.data():
        if t[:1] != "M":
            continue
        t_splitted = t.split("_")
        s_splitted = s.split("_")
        layer = int(t_splitted[0][1:])
        contr_map[layer] += w["weight"] > THR
    return contr_map


def get_contribution_maps(example, LAYERS=32, HEADS=32, THR=0.0):
    attnheads_contribution_maps = np.zeros((len(example["token_subgraphs"]), LAYERS, HEADS), dtype=bool)
    for t in range(len(example["token_subgraphs"])):
        attnheads_contribution_maps[t] = get_head_contribution_map(example["token_subgraphs"][t], example["contributions"]["c_attns"], THR=THR)

    ffns_contribution_maps = np.zeros((len(example["token_subgraphs"]), LAYERS), dtype=bool)
    for t in range(len(example["token_subgraphs"])):
        ffns_contribution_maps[t] = get_ffn_contribution_map(example["token_subgraphs"][t], LAYERS=LAYERS, THR=THR)

    return attnheads_contribution_maps, ffns_contribution_maps


def process_pickle_file(input_filepath, output_filepath):
    """
    Processes a single pickle file by extracting ranks and contributions and saving the modified data.
    """
    with open(input_filepath, "rb") as f:
        data = pickle.load(f)

    keys_to_remove = ["logit_lens_result", "neuron_contributions", "contributions", "full_graph"]

    for entry in data:
        resid_subj_ranks = []
        resid_ans_ranks = []
        output_subj_ranks = []
        output_ans_ranks = []

        logit_lens_result = entry.get("logit_lens_result", {})

        # Extract ranks from 'resid'
        for layer, tokens in logit_lens_result.get('resid', {}).items():
            for token_data in tokens:
                resid_subj_ranks.append(token_data["rank_subject"])
                resid_ans_ranks.append(token_data["rank_answer"])

        # Extract ranks from 'output'
        for layer, tokens in logit_lens_result.get('output', {}).items():
            for token_data in tokens:
                output_subj_ranks.append(token_data["rank_subject"])
                output_ans_ranks.append(token_data["rank_answer"])

        entry["resid_subj_ranks"] = resid_subj_ranks
        entry["resid_ans_ranks"] = resid_ans_ranks
        entry["output_subj_ranks"] = output_subj_ranks
        entry["output_ans_ranks"] = output_ans_ranks

        # Compute attnheads_contribution_maps and ffns_contribution_maps and store them
        attnheads_contribution_maps, ffns_contribution_maps = get_contribution_maps(entry)
        entry["attnheads_contribution_maps"] = attnheads_contribution_maps
        entry["ffns_contribution_maps"] = ffns_contribution_maps

        for key in keys_to_remove:
            entry.pop(key, None)

    # Save the processed data back to the output filepath
    with open(output_filepath, "wb") as f:
        pickle.dump(data, f)


def process_and_save_pkl_files_parallel(input_dir, output_dir, max_workers=4):
    """
    Recursively processes pickle files in parallel and saves the modified files
    while maintaining the same directory structure.
    
    Parameters:
    - input_dir: Directory to search for pickle files
    - output_dir: Directory to save processed files
    - max_workers: Maximum number of parallel workers
    """

    # Collect all file paths
    file_pairs = []
    for dirpath, _, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith(".pkl"):
                input_filepath = os.path.join(dirpath, filename)

                # Generate the corresponding output file path
                relative_path = os.path.relpath(dirpath, input_dir)
                output_dirpath = os.path.join(output_dir, relative_path)
                os.makedirs(output_dirpath, exist_ok=True)
                output_filepath = os.path.join(output_dirpath, filename)

                # Append file paths for processing
                file_pairs.append((input_filepath, output_filepath))

    # Use ProcessPoolExecutor to parallelize processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_pickle_file, inp, outp) for inp, outp in file_pairs]

        # Use tqdm to show progress
        with tqdm(total=len(futures), desc="Processing files") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()  # This will re-raise any exception from the worker
                except Exception as e:
                    print(f"Error processing file: {e}")
                pbar.update(1)


# Example usage
input_dir = "/nfs/datz/olmo_models/new_outputs"
output_dir = "/nfs/datz/olmo_models/processed_outputs"
max_workers = 2  # Adjust based on the number of cores on your system

process_and_save_pkl_files_parallel(input_dir, output_dir, max_workers)
