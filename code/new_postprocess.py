import pickle
import numpy as np
from networkx import DiGraph
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def parse_idx(idx_str: str) -> list[int]:
    # "5" -> [5], "2-4" -> [2,3,4]
    if "-" in idx_str:
        a, b = map(int, idx_str.split("-"))
        return list(range(a, b + 1))
    return [int(idx_str)]


def get_head_contribution_map(gr: DiGraph, attn_contr_maps, THR=0.0) -> np.ndarray:
    n_layers = len(attn_contr_maps)
    n_heads = attn_contr_maps[0].shape[-1]
    cm = np.zeros((n_layers, n_heads), dtype=bool)
    for src, tgt, data in gr.edges.data():
        if not tgt.startswith("A"):
            continue
        layer = int(tgt[1:].split("_")[0])
        tspan = tgt.split("_", 1)[1]
        sspan = src.split("_", 1)[1]
        t_idxs = parse_idx(tspan)
        s_idxs = parse_idx(sspan)
        attn_layer = np.array(attn_contr_maps[layer][0].cpu())
        for ti in t_idxs:
            for si in s_idxs:
                cm[layer] |= (attn_layer[ti, si, :] > THR)
    return cm


def get_ffn_contribution_map(gr: DiGraph, THR=0.0, LAYERS=32) -> np.ndarray:
    cm = np.zeros(LAYERS, dtype=bool)
    for _, tgt, data in gr.edges.data():
        if not tgt.startswith("M"):
            continue
        layer = int(tgt[1:].split("_")[0])
        if data.get("weight", 0.0) > THR:
            cm[layer] = True
    return cm


def get_contribution_maps(entry, LAYERS=32, HEADS=32, THR=0.0):
    T = len(entry["token_subgraphs"])
    attn_maps = np.zeros((T, LAYERS, HEADS), dtype=bool)
    ffn_maps  = np.zeros((T, LAYERS),       dtype=bool)
    for t, subgr in enumerate(entry["token_subgraphs"]):
        attn_maps[t] = get_head_contribution_map(subgr, entry["contributions"]["c_attns"], THR)
        ffn_maps[t]  = get_ffn_contribution_map(subgr, THR, LAYERS)
    return attn_maps, ffn_maps


def process_entry(entry):
    # 1) rank extraction
    lr = entry.get("logit_lens_result", {})
    resid_s, resid_a, resid_t = [], [], []
    out_s, out_a, out_t       = [], [], []

    for layer, toks in lr.get("resid", {}).items():
        resid_s.append([td["rank_subject"] for td in toks])
        resid_a.append([td["rank_answer"]  for td in toks])
        resid_t.append([td["top_tokens"]   for td in toks])
    for layer, toks in lr.get("output", {}).items():
        out_s.append([td["rank_subject"] for td in toks])
        out_a.append([td["rank_answer"]  for td in toks])
        out_t.append([td["top_tokens"]   for td in toks])

    entry["resid_subj_ranks"]  = np.array(resid_s, dtype=np.int32)
    entry["resid_ans_ranks"]   = np.array(resid_a, dtype=np.int32)
    entry["resid_top_tokens"]  = np.array(resid_t, dtype=np.int32)
    entry["output_subj_ranks"] = np.array(out_s,  dtype=np.int32)
    entry["output_ans_ranks"]  = np.array(out_a,  dtype=np.int32)
    entry["output_top_tokens"] = np.array(out_t,  dtype=np.int32)

    # 2) contributions
    a_map, f_map = get_contribution_maps(entry)
    entry["attnheads_contribution_maps"] = a_map
    entry["ffns_contribution_maps"]      = f_map

    # 3) collect subgraphs, drop heavies
    subgs = entry.get("token_subgraphs", None)
    for k in ["logit_lens_result", "neuron_contributions",
              "contributions", "full_graph", "token_subgraphs"]:
        entry.pop(k, None)

    return entry, subgs


def main():
    INPUT    = "collapsed_all_subgraphs_full.pkl"
    OUT_PP   = "collapsed_all_subgraphs_postprocessed.pkl"
    OUT_SUBG = "collapsed_all_subgraphs_subgraphs.pkl"

    with open(INPUT, "rb") as f:
        data: dict = pickle.load(f)

    processed = {}
    all_subgraphs = {}

    for snap, task_dict in data.items():
        processed[snap] = {}
        all_subgraphs[snap] = {}
        logging.info(f"Processing snapshot {snap}…")

        for task, entries in task_dict.items():
            proc_list = []
            subg_list = []
            for entry in entries:
                p, sg = process_entry(entry)
                proc_list.append(p)
                if sg is not None:
                    subg_list.append(sg)
            processed[snap][task]   = proc_list
            all_subgraphs[snap][task] = subg_list

    # write out
    with open(OUT_PP, "wb") as f:
        pickle.dump(processed, f)
    logging.info(f"Wrote postprocessed: {OUT_PP}")

    with open(OUT_SUBG, "wb") as f:
        pickle.dump(all_subgraphs, f)
    logging.info(f"Wrote subgraphs:    {OUT_SUBG}")


if __name__ == "__main__":
    main()
