from utils import *
import pickle
import networkx as nx

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
        # if tok in rel_tokens:
        #     rmin, rmax = (min(rel_tokens), max(rel_tokens)) if rel_tokens else (None, None)
        #     return f"{prefix}_{rmin}-{rmax}" if rmin is not None else f"{prefix}_relation"
        # # answer span
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
    root_directory = "/nfs/datz/olmo_models/new_outputs/"
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
                    subj_list = [5,6,7]
                    # subj_l = subj_tokens[:3]
                    # start = entry['tokens'].index(subj_l[0])
                    # # length of the span
                    # length = len(subj_l)                 
                    # subj_list = list(range(start, start + length))
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
    output_path = 'collapsed_all_subgraphs2.pkl'
    with open(output_path, 'wb') as fout:
        pickle.dump(collapsed_models, fout)

    print(f"Finished collapsing all token_subgraphs. Saved to {output_path}")
