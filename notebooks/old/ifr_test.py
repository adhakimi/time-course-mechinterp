import os
import pickle

def snapshot_sort_key(snapshot_name):
    """Sort snapshots by step number; 'main' is considered the final snapshot."""
    if snapshot_name == "main":
        return float('inf')
    try:
        # Extract the numeric part after 'step' before the first dash.
        step_str = snapshot_name.split("step")[1].split('-')[0]
        return int(step_str)
    except Exception:
        return 0

if __name__ == "__main__":
    root_directory = "/nfs/datz/olmo_models/"
    output_file = os.path.join(root_directory, "all_snapshots_reduced.pkl")
    with open(output_file, "rb") as f:
        all_snapshots_reduced = pickle.load(f)
        
    sorted_snapshots = sorted(all_snapshots_reduced.keys(), key=snapshot_sort_key)
    relations = list(all_snapshots_reduced['main'].keys())


    # This prints the token_subgraph (circuit) for the example "Germany has the capital city of Berlin". Selecting the -2 position takes the token_subgraph on the "of" position.
    # The nodes are in this format: M<layer>_<token>, there are four types of nodes:
    #    - X0_<token> original tokens coming into the model
    #    - A<layer>_<token> Attention Nodes
    #    - M<layer>_<token> MLP Nodes
    #    - I<layer>_<token> Node which represents the residual stream after the output of the MLP is added
    token_subgraph = all_snapshots_reduced[sorted_snapshots[-1]][relations[-1]][3]['token_subgraphs'][-2]
    
    print(token_subgraph.nodes())
    print(token_subgraph.edges())

    #This prints out the edges with their weight 
    print(token_subgraph.edges().data())