#from utils import *
import pickle
import networkx as nx
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
        cache_dir="/mounts/data/proj/hypersum",
        checkpoint_value=checkpoint,
    )

    return model

def graph_signature(G):
    nodes = {("node", n) for n in G.nodes()}
    edges = {("edge", u, v) for u, v in G.edges()}
    return nodes.union(edges)

def jaccard_similarity(G1, G2):
    sig1 = graph_signature(G1)
    sig2 = graph_signature(G2)
    intersection = sig1.intersection(sig2)
    union = sig1.union(sig2)
    return len(intersection) / len(union) if union else 1.0

def snapshot_sort_key(snapshot_name):
    if snapshot_name == "main":
        return float('inf')
    try:
        step_str = snapshot_name.split("step")[1].split('-')[0]
        return int(step_str)
    except Exception:
        return 0

def merge_subgraphs(subgraphs):
    combined = nx.DiGraph()
    for sg in subgraphs:
        for u, v, data in sg.edges(data=True):
            w = data.get("weight", 0)
            if combined.has_edge(u, v):
                combined[u][v]["weight"] = max(combined[u][v]["weight"], w)
            else:
                combined.add_edge(u, v, **data)
    return combined

def collapse_node(node, subj_span, rel_tokens, ans_span):
    parts = node.split('_')
    if len(parts) == 3:
        prefix, role, _ = parts
        return f"{prefix}_{role}"
    if len(parts) == 2:
        prefix, token_str = parts
        try:
            tok = int(token_str)
        except ValueError:
            return node
        # subject span
        if subj_span[0] is not None and subj_span[0] <= tok <= subj_span[1]:
            return f"{prefix}_{subj_span[0]}-{subj_span[1]}"
        # relation tokens
        if tok in rel_tokens:
            rmin, rmax = (min(rel_tokens), max(rel_tokens)) if rel_tokens else (None, None)
            return f"{prefix}_{rmin}-{rmax}" if rmin is not None else f"{prefix}_relation"
        # answer span
        a_min, a_max = (min(ans_span), max(ans_span)) if ans_span else (None, None)
        if a_min is not None and a_min <= tok <= a_max:
            return f"{prefix}_{a_min}-{a_max}"
        return node
    return node

def collapse_graph_nodes(G, subj_span, rel_tokens, ans_span):
    newG = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        cu = collapse_node(u, subj_span, rel_tokens, ans_span)
        cv = collapse_node(v, subj_span, rel_tokens, ans_span)
        if newG.has_edge(cu, cv):
            newG[cu][cv]['weight'] = max(newG[cu][cv]['weight'], data.get('weight', 0))
        else:
            newG.add_edge(cu, cv, **data)
    return newG

def merge_and_collapse_subgraph_list(subgraph_list, subj_span, rel_tokens, ans_span):
    if isinstance(subgraph_list, nx.DiGraph):
        subgraph_list = [subgraph_list]
    merged = merge_subgraphs(subgraph_list)
    return collapse_graph_nodes(merged, subj_span, rel_tokens, ans_span)

if __name__ == '__main__':
    root_directory = "/dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/di35pop/new_outputs"
    file_paths = traverse_directory(root_directory)
    all_files = [f for f in file_paths if "test" not in f]
    models_data = read_all_files(all_files)

    collapsed_models = {}

    for model_name, relations in models_data.items():
        collapsed_models[model_name] = {}
        for relation_name, entries in relations.items():
            collapsed_entries = []
            for entry in entries:
                tokens = entry['tokens']
                subj_tokens = entry.get('subject_tokens', [])
                raw_ans_tokens = entry.get('answer_tokens', [])

                # Determine subject span
                if entry.get('subj_token_span'):
                    subj_list = entry['subj_token_span']
                else:
                    subj_list = subj_tokens[:3]
                subj_span = (min(subj_list), max(subj_list)) if subj_list else (None, None)

                # Answer span list
                ans_span = entry.get('answer_token_span', [])

                # Relation tokens = all tokens not in subject or answer
                rel_tokens = set(tokens) - set(subj_list) - set(raw_ans_tokens)

                # Now collapse ALL token_subgraphs, preserving the same key structure
                original_subs = entry.get('token_subgraphs', {})
                collapsed_subs = []
                for t_idx, G in enumerate(original_subs):#.items():
                    # each G is a DiGraph for token index t_idx
                    collapsed_subs.insert(t_idx, merge_and_collapse_subgraph_list(
                        G, subj_span, rel_tokens, ans_span
                    ))

                # build new entry
                new_entry = entry.copy()
                new_entry['collapsed_token_subgraphs'] = collapsed_subs

                collapsed_entries.append(new_entry)

            collapsed_models[model_name][relation_name] = collapsed_entries

    # Save collapsed results for later use
    output_path = 'collapsed_all_subgraphs.pkl'
    with open(output_path, 'wb') as fout:
        pickle.dump(collapsed_models, fout)

    print(f"Finished collapsing all token_subgraphs. Saved to {output_path}")
