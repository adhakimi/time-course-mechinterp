import os, pickle, json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import pprint
from tqdm import tqdm
import multiprocessing
import re
from seaborn import heatmap
from functools import reduce
import seaborn as sns
import pandas as pd
import torch
from transformer_lens import HookedTransformer
from huggingface_hub import list_repo_refs


def traverse_directory(root_directory):
    """Traverse directory and collect all .pkl file paths."""
    file_paths = []
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith(".pkl") and "subgraph" not in filename:  # Adjust based on your file format
                file_paths.append(os.path.join(dirpath, filename))
    return file_paths

def extract_model_and_relation(file_path):
    """Extract model name and relation name from file path."""
    path_parts = file_path.split(os.sep)
    model_name = path_parts[-2]  # Model name is the second last part of the path
    relation_name = os.path.splitext(path_parts[-1])[0]  # File name without extension
    return model_name, relation_name

def read_single_file(file_path):
    """Load data from a single .pkl file and return the model, relation, and data."""
    model_name, relation_name = extract_model_and_relation(file_path)
    with open(file_path, 'rb') as file:
        data_entries = pickle.load(file)
    return model_name, relation_name, data_entries

def read_all_files(file_paths):
    """Reads all .pkl files and organizes data by model and relation sequentially."""
    models_data = {}
    total_files = len(file_paths)
    
    print(f"Total files to process: {total_files}")

    # Sequentially read each file
    for file_path in tqdm(file_paths, total=total_files, desc="Processing files"):
        model_name, relation_name, data_entries = read_single_file(file_path)
        if model_name not in models_data:
            models_data[model_name] = {}
        models_data[model_name][relation_name] = data_entries

    return models_data


def load_models(model_name, device='cuda', checkpoint=""):
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        dtype=torch.bfloat16,
        center_unembed=True,
        center_writing_weights=True,
        #fold_ln=True,
        device=device,
        trust_remote_code=True,
        cache_dir=os.environ.get("HF_CACHE_DIR") or None,
        checkpoint_value=checkpoint,
    )

    return model