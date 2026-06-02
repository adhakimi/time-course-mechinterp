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


def load_models(model_name, device='cuda:4', checkpoint=""):
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        dtype=torch.bfloat16,
        center_unembed=True,
        center_writing_weights=True,
        #fold_ln=True,
        device=device,
        trust_remote_code=True,
        cache_dir=os.path.join(os.environ["DATA_ROOT"], "revisions_temp/models--allenai--OLMo-7B-0424-hf/"),
        checkpoint_value=checkpoint,
    )

    return model

def calculate_consistency(heatmaps_dict, relation_name, fact_idx):
    iou_results = {}

    # Retrieve the 'main' model's heatmaps
    main_heatmaps = heatmaps_dict['main']

    # Iterate through each model (excluding 'main') and each heatmap type
    for model_name, model_data in heatmaps_dict.items():
        if model_name == 'main':
            continue

        # Store IoU results for this model
        model_ious = {}

        for heatmap_type in main_heatmaps.keys():
            if heatmap_type == "relation_answer_heatmaps":
                # Index by relation_name for relation_answer_heatmaps
                main_heatmap = main_heatmaps[heatmap_type][relation_name]
                model_heatmap = model_data[heatmap_type][relation_name]
            elif heatmap_type == "relation_answer_with_specific":
                # Index by relation_name and fact_idx for relation_answer_with_specific
                main_heatmap = main_heatmaps[heatmap_type][relation_name][fact_idx]
                model_heatmap = model_data[heatmap_type][relation_name][fact_idx]
            elif heatmap_type == "entity_heatmap":
                main_heatmap = main_heatmaps[heatmap_type]
                model_heatmap = model_data[heatmap_type]
            elif heatmap_type == "general_heatmap":
                main_heatmap = main_heatmaps[heatmap_type]
                model_heatmap = model_data[heatmap_type]
            else:
                continue

            # Calculate IoU
            intersection = np.logical_and(main_heatmap, model_heatmap)
            union = np.logical_or(main_heatmap, model_heatmap)

            area_of_intersection = np.sum(intersection)
            area_of_union = np.sum(union)

            # Avoid division by zero
            iou = area_of_intersection / area_of_union if area_of_union > 0 else 0

            # Store IoU result and statistics
            model_ious[heatmap_type] = {
                'iou': iou,
                'intersection': area_of_intersection,
                'union': area_of_union
            }

        # Add this model's IoU results to the main dictionary
        iou_results[model_name] = model_ious

    return iou_results

def plot_proportion_overlap_multiple(overlap_results, relation_name, fact_idx):
    # Prepare data
    model_names = sorted(
        overlap_results.keys(),
        key=lambda x: int(x.split('step')[1].split('-')[0])
    )
    general_proportions = []
    entity_proportions = []
    answer_all_proportions = []
    answer_proportions = []

    # Populate lists for each heatmap type
    for model_name in model_names:
        general_proportions.append(overlap_results[model_name]['general_heatmap']['iou'])
        entity_proportions.append(overlap_results[model_name]['entity_heatmap']['iou'])
        answer_all_proportions.append(overlap_results[model_name]['relation_answer_heatmaps']['iou'])
        answer_proportions.append(overlap_results[model_name]['relation_answer_with_specific']['iou'])

    # Plotting
    plt.figure(figsize=(20, 6))
    plt.plot(model_names, general_proportions, label="General Heatmap", marker='o', linestyle='-')
    plt.plot(model_names, entity_proportions, label="Entity Heatmap", marker='s', linestyle='--')
    plt.plot(model_names, answer_all_proportions, label="Relation Answer Heatmap", marker='x', linestyle='-.')
    plt.plot(model_names, answer_proportions, label="Answer Specific Heatmap", marker='^', linestyle='-.')

    plt.xticks(ticks=range(len(model_names)), labels=model_names, rotation=45, ha='right')
    plt.xlabel("Model Steps")
    plt.ylabel("IoU")
    plt.title(f"IoU Across Models for Relation: {relation_name}, Fact: {fact_idx}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_filename = f"neuron_iou.png"
    plt.savefig(output_filename, format="png", dpi=300)  # Save as PNG with 300 dpi for good quality
    plt.close()
    

def plot_count_proportion_overlap_multiple(overlap_results, relation_name, fact_idx):
    # Prepare data
    model_names = sorted(
        overlap_results.keys(),
        key=lambda x: int(x.split('step')[1].split('-')[0])
    )
    general_proportions = []
    entity_proportions = []
    answer_all_proportions = []
    answer_proportions = []

    # Populate lists for each heatmap type
    for model_name in model_names:
        general_proportions.append(overlap_results[model_name]['general_heatmap']['intersection'])
        entity_proportions.append(overlap_results[model_name]['entity_heatmap']['intersection'])
        answer_all_proportions.append(overlap_results[model_name]['relation_answer_heatmaps']['intersection'])
        answer_proportions.append(overlap_results[model_name]['relation_answer_with_specific']['intersection'])

    # Plotting
    plt.figure(figsize=(20, 6))
    plt.plot(model_names, general_proportions, label="General Heatmap", marker='o', linestyle='-')
    plt.plot(model_names, entity_proportions, label="Entity Heatmap", marker='s', linestyle='--')
    plt.plot(model_names, answer_all_proportions, label="Relation Answer Heatmap", marker='x', linestyle='-.')
    plt.plot(model_names, answer_proportions, label="Answer Specific Heatmap", marker='^', linestyle='-.')

    plt.xticks(ticks=range(len(model_names)), labels=model_names, rotation=45, ha='right')
    plt.xlabel("Model Steps")
    plt.ylabel("Counts")
    plt.title(f"Counts of overlapping specialized Heads Across Models for Relation: {relation_name}, Fact: {fact_idx}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    output_filename = f"neuron_intersection.png"
    plt.savefig(output_filename, format="png", dpi=300)  # Save as PNG with 300 dpi for good quality
    plt.close() 
    
def plot_all_count_proportion_overlap_multiple(overlap_results, relation_name, fact_idx):
    # Prepare data
    model_names = sorted(
        overlap_results.keys(),
        key=lambda x: int(x.split('step')[1].split('-')[0])
    )
    general_proportions = []
    entity_proportions = []
    answer_all_proportions = []
    answer_proportions = []

    # Populate lists for each heatmap type
    for model_name in model_names:
        general_proportions.append(overlap_results[model_name]['general_heatmap']['union'])
        entity_proportions.append(overlap_results[model_name]['entity_heatmap']['union'])
        answer_all_proportions.append(overlap_results[model_name]['relation_answer_heatmaps']['union'])
        answer_proportions.append(overlap_results[model_name]['relation_answer_with_specific']['union'])

    # model_names.append('main')
    # general_proportions.append(overlap_results[model_name]['general_heatmap']['count_main'])
    # entity_proportions.append(overlap_results[model_name]['entity_heatmap']['count_main'])
    # answer_all_proportions.append(overlap_results[model_name]['relation_answer_heatmaps']['count_main'])
    # answer_proportions.append(overlap_results[model_name]['relation_answer_with_specific']['count_main'])


    # Plotting
    plt.figure(figsize=(20, 6))
    plt.plot(model_names, general_proportions, label="General Heads", marker='o', linestyle='-')
    plt.plot(model_names, entity_proportions, label="Entity Heads", marker='s', linestyle='--')
    plt.plot(model_names, answer_all_proportions, label="Relation Answer Heads", marker='x', linestyle='-.')
    plt.plot(model_names, answer_proportions, label="Answer Specific Heads", marker='^', linestyle='-.')

    plt.xticks(ticks=range(len(model_names)), labels=model_names, rotation=45, ha='right')
    plt.xlabel("Model Steps")
    plt.ylabel("Counts")
    plt.title(f"Counts of All Specialized Heads Across Models for Relation: {relation_name}, Fact: {fact_idx}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_filename = f"neuron_union.png"
    plt.savefig(output_filename, format="png", dpi=300)  # Save as PNG with 300 dpi for good quality
    plt.close()
        
def calculate_proper_heads(heatmaps_dict):
    """
    Calculates proper entity heads, proper relation answer heads, 
    and proper answer specific heads for each model in heatmaps_dict.

    Args:
        heatmaps_dict (dict): A dictionary where each key is a model name and 
                              the value is a dictionary with heatmaps.

    Returns:
        dict: A dictionary containing proper heads for each model.
    """
    proper_heads_dict = {}

    for model_name, heatmaps in heatmaps_dict.items():
        # Extract heatmaps
        general_heads = heatmaps['general_heatmap']
        entity_heads = heatmaps['entity_heatmap']
        relation_answer_heads = heatmaps['relation_answer_heatmaps']
        answer_specific_heads = heatmaps['relation_answer_with_specific']

        # Calculate proper heads
        proper_entity_heads = np.logical_and(entity_heads, np.logical_not(general_heads))
        
        # Initialize dictionaries to store results for relations and specific answers
        proper_relation_answer_heads = {}
        proper_answer_specific_heads = {}

        # Calculate proper relation answer heads per relation
        for relation, relation_heads in relation_answer_heatmaps.items():
            proper_relation_answer_heads[relation] = np.logical_and(
                relation_heads,
                np.logical_not(np.logical_or(entity_heads, general_heads))
            )

        # Calculate proper answer specific heads per relation and specific answer
        for relation, specific_answers in relation_answer_with_specific.items():
            proper_answer_specific_heads[relation] = {}
            for answer_id, specific_heads in specific_answers.items():
                proper_answer_specific_heads[relation][answer_id] = np.logical_and(
                    specific_heads,
                    np.logical_not(
                        np.logical_or(
                            relation_answer_heatmaps[relation],
                            np.logical_or(entity_heads, general_heads)
                        )
                    )
                )

        # Store results in the output dictionary
        proper_heads_dict[model_name] = {
            'proper_general_heads': general_heads,
            'proper_entity_heads': proper_entity_heads,
            'proper_relation_answer_heads': proper_relation_answer_heads,
            'proper_answer_specific_heads': proper_answer_specific_heads
        }

    return proper_heads_dict

def calculate_consistency_proper_heads(heatmaps_dict, relation_name, fact_idx):
    iou_results = {}

    # Retrieve the 'main' model's heatmaps
    main_heatmaps = heatmaps_dict['main']

    # Iterate through each model (excluding 'main') and each heatmap type
    for model_name, model_data in heatmaps_dict.items():
        if model_name == 'main':
            continue

        # Store IoU results for this model
        model_ious = {}

        for heatmap_type in main_heatmaps.keys():
            if heatmap_type == "proper_relation_answer_heads":
                # Index by relation_name for relation_answer_heatmaps
                main_heatmap = main_heatmaps[heatmap_type][relation_name]
                model_heatmap = model_data[heatmap_type][relation_name]
            elif heatmap_type == "proper_answer_specific_heads":
                # Index by relation_name and fact_idx for relation_answer_with_specific
                main_heatmap = main_heatmaps[heatmap_type][relation_name][fact_idx]
                model_heatmap = model_data[heatmap_type][relation_name][fact_idx]
            elif heatmap_type == "proper_entity_heads":
                main_heatmap = main_heatmaps[heatmap_type]
                model_heatmap = model_data[heatmap_type]
            elif heatmap_type == "proper_general_heads":
                main_heatmap = main_heatmaps[heatmap_type]
                model_heatmap = model_data[heatmap_type]
            else:
                continue

            # Calculate IoU
            intersection = np.logical_and(main_heatmap, model_heatmap)
            union = np.logical_or(main_heatmap, model_heatmap)

            area_of_intersection = np.sum(intersection)
            area_of_union = np.sum(union)

            # Avoid division by zero
            iou = area_of_intersection / area_of_union if area_of_union > 0 else 0

            # Store IoU result and statistics
            model_ious[heatmap_type] = {
                'iou': iou,
                'intersection': area_of_intersection,
                'union': area_of_union
            }

        # Add this model's IoU results to the main dictionary
        iou_results[model_name] = model_ious

    return iou_results



def plot_proper_heads_iou_multiple(iou_results, relation_name, fact_idx):
    # Prepare data
    model_names = sorted(
        iou_results.keys(),
        key=lambda x: int(x.split('step')[1].split('-')[0])
    )
    general_ious = []
    entity_ious = []
    answer_all_ious = []
    answer_ious = []

    # Populate lists for each heatmap type
    for model_name in model_names:
        general_ious.append(iou_results[model_name]['proper_general_heads']['iou'])
        entity_ious.append(iou_results[model_name]['proper_entity_heads']['iou'])
        answer_all_ious.append(iou_results[model_name]['proper_relation_answer_heads']['iou'])
        answer_ious.append(iou_results[model_name]['proper_answer_specific_heads']['iou'])

    # Plotting
    plt.figure(figsize=(20, 6))
    plt.plot(model_names, general_ious, label="Proper General Heads IoU", marker='o', linestyle='-')
    plt.plot(model_names, entity_ious, label="Proper Entity Heads IoU", marker='s', linestyle='--')
    plt.plot(model_names, answer_all_ious, label="Proper Relation Answer Heads IoU", marker='x', linestyle='-.')
    plt.plot(model_names, answer_ious, label="Proper Answer Specific Heads IoU", marker='^', linestyle='-.')

    plt.xticks(ticks=range(len(model_names)), labels=model_names, rotation=45, ha='right')
    plt.xlabel("Model Steps")
    plt.ylabel("IoU")
    plt.title(f"Proper Heads IoU Across Models for Relation: {relation_name}, Fact: {fact_idx}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_filename = f"proper_neuron_iou.png"
    plt.savefig(output_filename, format="png", dpi=300)  # Save as PNG with 300 dpi for good quality
    plt.close()
    
def plot_proper_heads_counts_intersection(iou_results, relation_name, fact_idx):
    # Prepare data
    model_names = sorted(
        iou_results.keys(),
        key=lambda x: int(x.split('step')[1].split('-')[0])
    )

    general_intersections = []
    entity_intersections = []
    answer_all_intersections = []
    answer_intersections = []

    # Populate lists for each heatmap type
    for model_name in model_names:
        general_intersections.append(iou_results[model_name]['proper_general_heads']['intersection'])
        entity_intersections.append(iou_results[model_name]['proper_entity_heads']['intersection'])
        answer_all_intersections.append(iou_results[model_name]['proper_relation_answer_heads']['intersection'])
        answer_intersections.append(iou_results[model_name]['proper_answer_specific_heads']['intersection'])

    # Plotting Intersections
    plt.figure(figsize=(20, 6))
    plt.plot(model_names, general_intersections, label="Proper General Heads Intersections", marker='o', linestyle='-')
    plt.plot(model_names, entity_intersections, label="Proper Entity Heads Intersections", marker='s', linestyle='--')
    plt.plot(model_names, answer_all_intersections, label="Proper Relation Answer Heads Intersections", marker='x', linestyle='-.')
    plt.plot(model_names, answer_intersections, label="Proper Answer Specific Heads Intersections", marker='^', linestyle='-.')

    plt.xticks(ticks=range(len(model_names)), labels=model_names, rotation=45, ha='right')
    plt.xlabel("Model Steps")
    plt.ylabel("Intersection Counts")
    plt.title(f"Proper Heads Intersections Across Models for Relation: {relation_name}, Fact: {fact_idx}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_filename = f"proper_neuron_intersection.png"
    plt.savefig(output_filename, format="png", dpi=300)  # Save as PNG with 300 dpi for good quality
    plt.close()

def plot_proper_heads_counts_union(iou_results, relation_name, fact_idx):
    # Prepare data
    model_names = sorted(
        iou_results.keys(),
        key=lambda x: int(x.split('step')[1].split('-')[0])
    )

    general_unions = []
    entity_unions = []
    answer_all_unions = []
    answer_unions = []

    # Populate lists for each heatmap type
    for model_name in model_names:
        general_unions.append(iou_results[model_name]['proper_general_heads']['union'])
        entity_unions.append(iou_results[model_name]['proper_entity_heads']['union'])
        answer_all_unions.append(iou_results[model_name]['proper_relation_answer_heads']['union'])
        answer_unions.append(iou_results[model_name]['proper_answer_specific_heads']['union'])

    # Plotting Intersections
    plt.figure(figsize=(20, 6))
    plt.plot(model_names, general_unions, label="Proper General Heads Unions", marker='o', linestyle='-')
    plt.plot(model_names, entity_unions, label="Proper Entity Heads Unions", marker='s', linestyle='--')
    plt.plot(model_names, answer_all_unions, label="Proper Relation Answer Heads Unions", marker='x', linestyle='-.')
    plt.plot(model_names, answer_unions, label="Proper Answer Specific Heads Unions", marker='^', linestyle='-.')

    plt.xticks(ticks=range(len(model_names)), labels=model_names, rotation=45, ha='right')
    plt.xlabel("Model Steps")
    plt.ylabel("Union Counts")
    plt.title(f"Proper Heads Unions Across Models for Relation: {relation_name}, Fact: {fact_idx}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    output_filename = f"proper_neuron_union.png"
    plt.savefig(output_filename, format="png", dpi=300)  # Save as PNG with 300 dpi for good quality
    plt.close()



root_directory = os.path.join(os.environ["DATA_ROOT"], "neuron_outputs")

file_paths = traverse_directory(root_directory)
#main_files = [path for path in file_paths if ("subgraph" not in path and "test" not in path and "all_country_capital" in path)]

all_files = [file for file in file_paths if "test" not in file]

models_data = read_all_files(all_files)

for model_name, relations in models_data.items():
    print(f"Model: {model_name}")
    for relation_name, data_entries in relations.items():
        print(f"  Relation: {relation_name}, Number of Entries: {len(data_entries)}")
        
sorted_models = sorted(
        models_data.keys(), 
        key=lambda name: int(re.search(r'step(\d+)', name).group(1)) if "main" not in name else float('inf')
    )

model_name = "allenai/OLMo-7B-0424-hf"

model = load_models(model_name, device='cuda', checkpoint="main")


heatmaps_dict = {}

theta = 0.01

for model_name in sorted_models:
    relations_data = models_data[model_name]

    print(model_name)
    g_total = 0
    general_heatmap = np.zeros((32, 11008), dtype=float)
    e_total = 0
    entity_heatmap = np.zeros((32, 11008), dtype=float)

    relation_answer_heatmaps = {}
    relation_answer_with_specific = {}

    for relation_name, entries in relations_data.items():

        relation_answer_heatmap = np.zeros((32, 11008), dtype=float)
        relation_answer_total = 0
        answer_specific_heatmaps = {}

        for idx, entry in enumerate(entries):

            # Ensure subj_token_span is not None
            if entry["subj_token_span"] is None:
                entry["subj_token_span"] = np.arange(0, len(entry["subject_tokens"])).tolist()

            # Update general heatmap
            #entry['neuron_contributions']['vals'][:17].shape
            #general_heatmap += np.mean(entry['neuron_contributions']['vals'][:entry['answer_token_span'][0]], axis=0) #added [:entry['answer_token_span'][0]] because I dont need the contributions from the answer token itself
            general_heatmap += torch.mean(entry['neuron_contributions']['vals'][:entry['answer_token_span'][0]], dim=0).cpu().numpy()
            g_total += 1

            # Update entity heatmap for subject tokens
            one_data_map = np.zeros((32, 11008), dtype=float)
            for t in entry["subj_token_span"]:
                if t == 0:
                    e_total -= 1
                    continue
                one_data_map += entry['neuron_contributions']['vals'][t - 1].cpu().numpy()
            entity_heatmap += one_data_map
            e_total += len(entry["subj_token_span"])

            # Update entity heatmap and relation answer heatmap for answer tokens
            one_data_map = np.zeros((32, 11008), dtype=float)
            for t in entry["answer_token_span"]:
                one_data_map += entry['neuron_contributions']['vals'][t - 1].cpu().numpy()
            entity_heatmap += one_data_map
            relation_answer_heatmap += one_data_map

            e_total += len(entry["answer_token_span"])
            relation_answer_total += len(entry["answer_token_span"])

            # Store answer-specific heatmap
            answer_specific_heatmap = one_data_map / len(entry["answer_token_span"])
            answer_specific_heatmaps[idx] = answer_specific_heatmap  > theta

        relation_answer_heatmap /= relation_answer_total
        relation_answer_heatmaps[f"{relation_name}"] = relation_answer_heatmap > theta
        relation_answer_with_specific[f"{relation_name}"] = answer_specific_heatmaps

    # Normalize heatmaps
    general_heatmap /= g_total
    entity_heatmap /= e_total

    # Store heatmaps in the dictionary
    heatmaps_dict[model_name] = {
        'general_heatmap': general_heatmap > theta,
        'entity_heatmap': entity_heatmap > theta,
        'relation_answer_heatmaps': relation_answer_heatmaps,
        'relation_answer_with_specific': relation_answer_with_specific
    }
    
relation_name = "country_capital_city"
fact_idx = 1
sent = models_data['main'][relation_name][fact_idx]['sentence']
overlap = calculate_consistency(heatmaps_dict, relation_name, fact_idx)
plot_proportion_overlap_multiple(overlap, relation_name, sent)
plot_count_proportion_overlap_multiple(overlap, relation_name, sent)
plot_all_count_proportion_overlap_multiple(overlap, relation_name, sent)


relation_name = "books_written"
fact_idx = 1
sent = models_data['main'][relation_name][fact_idx]['sentence']
overlap = calculate_consistency(heatmaps_dict, relation_name, fact_idx)
plot_proportion_overlap_multiple(overlap, relation_name, sent)
plot_count_proportion_overlap_multiple(overlap, relation_name, sent)
plot_all_count_proportion_overlap_multiple(overlap, relation_name, sent)


relation_name = "country_capital_city"
fact_idx = 1
sent = models_data['main'][relation_name][fact_idx]['sentence']
proper_heads = calculate_proper_heads(heatmaps_dict)
overlap = calculate_consistency_proper_heads(proper_heads, relation_name, fact_idx)
plot_proper_heads_iou_multiple(overlap, relation_name, sent)
plot_proper_heads_counts_intersection(overlap, relation_name, sent)
plot_proper_heads_counts_union(overlap, relation_name, sent)

relation_name = "books_written"
fact_idx = 1
sent = models_data['main'][relation_name][fact_idx]['sentence']
proper_heads = calculate_proper_heads(heatmaps_dict)
overlap = calculate_consistency_proper_heads(proper_heads, relation_name, fact_idx)
plot_proper_heads_iou_multiple(overlap, relation_name, sent)
plot_proper_heads_counts_intersection(overlap, relation_name, sent)
plot_proper_heads_counts_union(overlap, relation_name, sent)