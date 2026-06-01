import pickle
import numpy as np
from networkx import DiGraph
from seaborn import heatmap
import numpy as np
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


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

def process_pickle_file(input_filepath, output_filepath, subgraphs_filepath):
    """
    Processes a single pickle file by extracting ranks and contributions and saving the modified data.
    """
    with open(input_filepath, "rb") as f:
        data = pickle.load(f)

    keys_to_remove = ["logit_lens_result", "neuron_contributions", "contributions", "full_graph", "token_subgraphs"]

    subgraphs_data = []

    for entry in data:
        resid_subj_ranks = []
        resid_ans_ranks = []
        output_subj_ranks = []
        output_ans_ranks = []
        resid_top_tokens = []
        output_top_tokens = []

        logit_lens_result = entry.get("logit_lens_result", {})
        
        # Extract ranks from 'resid' 
        for layer, tokens in logit_lens_result.get('resid', {}).items():
            layer_subj_rank = [token_data["rank_subject"] for token_data in tokens]
            layer_ans_rank = [token_data["rank_answer"] for token_data in tokens]
            layer_top_tokens = [token_data["top_tokens"] for token_data in tokens]

            resid_subj_ranks.append(layer_subj_rank) # why should this be a list in a list?
            resid_ans_ranks.append(layer_ans_rank)
            resid_top_tokens.append(layer_top_tokens)

        # Extract ranks from 'output'
        for layer, tokens in logit_lens_result.get('output', {}).items():
            layer_subj_rank = [token_data["rank_subject"] for token_data in tokens]
            layer_ans_rank = [token_data["rank_answer"] for token_data in tokens]
            layer_top_tokens = [token_data["top_tokens"] for token_data in tokens]
            
            output_subj_ranks.append(layer_subj_rank)
            output_ans_ranks.append(layer_ans_rank)
            output_top_tokens.append(layer_top_tokens)

        entry["resid_subj_ranks"] = np.array(resid_subj_ranks, dtype=np.int32) 
        entry["resid_ans_ranks"] = np.array(resid_ans_ranks, dtype=np.int32) 
        entry["resid_top_tokens"] = np.array(resid_top_tokens, dtype=np.int32) 
        entry["output_subj_ranks"] = np.array(output_subj_ranks, dtype=np.int32) 
        entry["output_ans_ranks"] = np.array(output_ans_ranks, dtype=np.int32) 
        entry["output_top_tokens"] = np.array(output_top_tokens, dtype=np.int32) 

        # Compute attnheads_contribution_maps and ffns_contribution_maps and store them
        attnheads_contribution_maps, ffns_contribution_maps = get_contribution_maps(entry)
        entry["attnheads_contribution_maps"] = attnheads_contribution_maps
        entry["ffns_contribution_maps"] = ffns_contribution_maps
        
        # Extract subgraphs and add to subgraphs_data
        subgraphs = entry.get("token_subgraphs", None)
        
        if subgraphs is not None:
            subgraphs_data.append(subgraphs)
        
        for key in keys_to_remove:
            entry.pop(key, None)

    # Save the processed data back to the output filepath
    with open(output_filepath, "wb") as f:
        pickle.dump(data, f)
        
    with open(subgraphs_filepath, "wb") as f:
        pickle.dump(subgraphs_data, f)

# def process_and_save_pkl_files(input_dir, output_dir):
#     """
#     Recursively processes pickle files and saves the modified files
#     while maintaining the same directory structure.
    
#     Parameters:
#     - input_dir: Directory to search for pickle files
#     - output_dir: Directory to save processed files
#     """

#     # Collect all file paths
#     file_pairs = []
#     for dirpath, _, filenames in os.walk(input_dir):
#         for filename in filenames:
#             if filename.endswith(".pkl"):
#                 input_filepath = os.path.join(dirpath, filename)

#                 # Generate the corresponding output file path
#                 relative_path = os.path.relpath(dirpath, input_dir)
#                 output_dirpath = os.path.join(output_dir, relative_path)
#                 os.makedirs(output_dirpath, exist_ok=True)
#                 output_filepath = os.path.join(output_dirpath, filename)

#                 # Append file paths for processing
#                 file_pairs.append((input_filepath, output_filepath))

#     file_pairs = file_pairs[:28]
#     # Process files sequentially
#     for inp, outp in tqdm(file_pairs, desc="Processing files"):
#     # Modify the output path to create a new subgraphs file path
#         base, ext = os.path.splitext(outp)
#         subgraphs_file = f"{base}_subgraphs{ext}"
        
#         # Call the function with the modified paths
#         process_pickle_file(inp, outp, subgraphs_file)

def process_and_save_pkl_files(input_dir, output_dir, max_workers=4):
    """
    Recursively processes pickle files and saves the modified files
    while maintaining the same directory structure in parallel.
    
    Parameters:
    - input_dir: Directory to search for pickle files
    - output_dir: Directory to save processed files
    - max_workers: Number of parallel workers for processing
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

                # Check if the output file already exists
                if os.path.exists(output_filepath):
                    logging.info(f"File {output_filepath} already exists. Skipping...")
                    continue  # Skip this file
                
                file_pairs.append((input_filepath, output_filepath))

    logging.info(f"Total files to process: {len(file_pairs)}")

    # Use ProcessPoolExecutor to parallelize the processing
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for inp, outp in file_pairs:
            # Modify the output path to create a new subgraphs file path
            base, ext = os.path.splitext(outp)
            subgraphs_file = f"{base}_subgraphs{ext}"
            
            logging.debug(f"Processing file: {inp}")
            
            # Submit tasks to the executor
            futures.append(executor.submit(process_pickle_file, inp, outp, subgraphs_file))

        # Use tqdm to monitor progress
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            # Catch exceptions if any file processing fails
            try:
                future.result()  # Will raise if the process raised an exception
            except Exception as exc:
                logging.error(f"File processing generated an exception: {exc}")


# Example usage
DATA_ROOT = os.environ["DATA_ROOT"]  # set in paths.env; `source paths.env` first
input_dir = os.path.join(DATA_ROOT, "new_outputs/")
output_dir = os.path.join(DATA_ROOT, "new_processed_outputs/")
max_workers = 8  # Adjust based on the number of cores on your system

process_and_save_pkl_files(input_dir, output_dir, max_workers)
