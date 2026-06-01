# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os
from tqdm import tqdm
import time
from copy import deepcopy
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
#import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))) # Why is it not working without this?

import networkx as nx
# import pandas as pd
# import plotly.express
# import plotly.graph_objects as go
# import streamlit as st
# import streamlit_extras.row as st_row
import torch
from jaxtyping import Float
from torch.amp import autocast
from transformers import HfArgumentParser

# import llm_transparency_tool.components
# from llm_transparency_tool.models.tlens_model import TransformerLensTransparentLlm
import llm_transparency_tool.routes.contributions as contributions
import llm_transparency_tool.routes.graph
from llm_transparency_tool.models.transparent_llm import TransparentLlm
from llm_transparency_tool.routes.graph_node import NodeType
from llm_transparency_tool.server.graph_selection import (
    GraphSelection,
    UiGraphEdge,
    UiGraphNode,
)
from llm_transparency_tool.server.utils import (
    B0,
    get_contribution_graph,
    load_dataset,
    load_model,
    get_val,
    possible_devices
)
from networkx.classes.digraph import DiGraph


from dataclasses import dataclass
from typing import List
import json
import pickle


@dataclass
class LogLensResult:
    token_idx: int = 0
    top_tokens: List[str] = field(default_factory=list)
    rank_subject: int = 0
    rank_answer: int = 0
    logit_subject: float = 0.0
    logit_answer: float = 0.0
    prob_subject: float = 0.0
    prob_answer: float = 0.0
    max_logit: float = 0.0
    max_prob: float = 0.0
    entropy: float = 0.0



# @st.cache_resource(
#     hash_funcs={
#         nx.Graph: id,
#         DiGraph: id
#     }
# )
def cached_build_paths_to_predictions(
    graph: nx.Graph,
    n_layers: int,
    n_tokens: int,
    starting_tokens: List[int],
    threshold: float,
):
    return llm_transparency_tool.routes.graph.build_paths_to_predictions(
        graph, n_layers, n_tokens, starting_tokens, threshold
    )

# @st.cache_resource(
#     hash_funcs={
#         TransformerLensTransparentLlm: id
#     }
# )
def cached_run_inference_and_populate_state(
    stateless_model,
    sentences,
    subject,
    object
):
    # stateful_model = stateless_model.copy()
    stateless_model.run(sentences, subject, object)
    return stateless_model



def load_json_files(directory: str, selected_category: str) -> Dict[str, List[Dict]]:
    data = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                with open(path, 'r') as f:
                    category = os.path.basename(root)
                    if category == selected_category or not selected_category:
                        relation = os.path.basename(file).replace('.json', '')
                        #if relation == "country_capital_city":
                        if relation not in data:
                            data[relation] = []
                        data[relation].append(json.load(f))
                    
    return data

def find_subject_token_position(template: str, subject: str) -> int:
    # Format the template with the subject
    fact = template.format(subject)

    # Tokenize the sentence by splitting on whitespace
    tokens = fact.split()

    # Find the position of the subject in the tokenized list (if contained in a token)
    for i, token in enumerate(tokens):
        if subject in token:
            return i
    
    # If subject is not found, raise an error
    raise ValueError(f"Subject '{subject}' not found in the tokenized string.")
    

def parse_samples(data: Dict[str, List[Dict]]) -> Tuple[List[str], List[str], List[str], List[str], List[int]]:
    facts = []
    subjects = []
    targets = []
    sentences = []
    all_subject_positions = []
    indices = []
    subject_object_to_index = {}
    current_index = 0

    for entries in data:
        prompt_templates = entries['prompt_templates']
        samples = entries['samples']
        
        for sample in samples:
            subject = sample['subject']
            obj = " " + sample['object']
            
            # Use the subject-object pair as the unique identifier for indexing
            subject_object_pair = (subject, obj)

            if subject_object_pair not in subject_object_to_index:
                subject_object_to_index[subject_object_pair] = current_index
                current_index += 1
            
            # Get the index for this subject-object pair
            fact_index = subject_object_to_index[subject_object_pair]
            
            for template in prompt_templates:  # Iterate through multiple templates
                subject_pos = find_subject_token_position(template, "{}")
                
                fact = template.format(subject)
                facts.append(fact)
                if subject_pos != 0: #This is needed to find the proper indices if subject at first position
                    subjects.append(" " + subject) 
                else:
                    subjects.append(subject)
                targets.append(obj)
                sentences.append(fact + obj)
                all_subject_positions.append(subject_pos)
                indices.append(fact_index)  # Append the index corresponding to this subject-object pair
    
    return sentences, facts, subjects, targets, all_subject_positions, indices

def save_analysis_per_relation(data, relation, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a filename for each relation
    filename = os.path.join(output_dir, f"{relation}.pkl")
    
    # Save the relation analysis to a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Analysis for relation '{relation}' saved to {filename}")
    
# Function to find the span indices in a tensor

def find_span_indices(tokens, subj_tokens):
    # Get the length of the span to match
    span_len = len(subj_tokens)
    # Sliding window over the tokens to find where the subj_tokens match
    for i in range(len(tokens) - span_len + 1):
        if torch.equal(tokens[i:i+span_len], subj_tokens):
            return list(range(i, i + span_len))
    return None
    

class App:
    _stateful_model: TransparentLlm = None
    _graph: Optional[nx.Graph] = None
    _contribution_threshold: float = 0.0
    _renormalize_after_threshold: bool = False
    _normalize_before_unembedding: bool = True

    @property
    def stateful_model(self) -> TransparentLlm:
        return self._stateful_model

    def __init__(self):
        pass
        # self._config = config
        # st.set_page_config(layout="wide")
        # st.markdown(margins_css, unsafe_allow_html=True)

    def _get_representation(self, node: Optional[UiGraphNode]) -> Optional[Float[torch.Tensor, "d_model"]]:
        if node is None:
            return None
        fn = {
            NodeType.AFTER_ATTN: self.stateful_model.residual_after_attn,
            NodeType.AFTER_FFN: self.stateful_model.residual_out,
            NodeType.FFN: None,
            NodeType.ORIGINAL: self.stateful_model.residual_in,
        }
        return fn[node.type](node.layer)[B0][node.token]    
    
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)

        self.model_name = config.get("model_name", "allenai/OLMo-7B-0424-hf")
        self.device = config.get("device", "gpu")
        _model_path = config.get("_model_path", None)
        self._model_path = os.path.expandvars(_model_path) if _model_path else None
        dtype_str = config.get("dtype", "torch.bfloat16")
        self.dtype = getattr(torch, dtype_str, torch.bfloat16)
        self.amp_enabled = config.get("amp_enabled", True)
        self._renormalize_after_threshold = config.get("renormalize_after_threshold", True)
        self._normalize_before_unembedding = config.get("normalize_before_unembedding", True)
        self._prepend_bos = config.get("prepend_bos", False)
        self._do_neuron_level = config.get("do_neuron_level", False)
        self._do_head_level = config.get("do_head_level", True)
        self._contribution_threshold = config.get("contribution_threshold", 0.01)
        self._logit_lens_topK = config.get("logit_lens_topK", 10)
        self._logit_lens_topK_neurons = config.get("logit_lens_topK_neurons", 10)

    def _unembed(
        self,
        representation: torch.Tensor,
    ) -> torch.Tensor:
        return self.stateful_model.unembed(representation, normalize=self._normalize_before_unembedding)

    def draw_graph(self, contribution_threshold: float) -> Optional[GraphSelection]:
        tokens = self.stateful_model.tokens()[B0]
        n_tokens = tokens.shape[0]
        model_info = self.stateful_model.model_info()

        graphs = cached_build_paths_to_predictions(
            self._graph,
            model_info.n_layers,
            n_tokens,
            range(n_tokens),
            contribution_threshold,
        )

        return llm_transparency_tool.components.contribution_graph(
            model_info,
            self.stateful_model.tokens_to_strings(tokens),
            graphs,
            key=f"graph_{hash(self.sentence)}",
        )


    def run_inference(self):

        with autocast(enabled=self.amp_enabled, device_type="cuda", dtype=self.dtype):
            self._stateful_model = cached_run_inference_and_populate_state(self.stateful_model, [self.sentence], self.subj_token, self.obj_token)

        with autocast(enabled=self.amp_enabled, device_type="cuda", dtype=self.dtype):
            self._graph, self._contributions_dict = get_contribution_graph(
                self.stateful_model,
                "",
                self.stateful_model.tokens()[B0].tolist(),
                (self._contribution_threshold if self._renormalize_after_threshold else 0.0),
            )

    #@profile
    def process_logits(self, logit_scores, subj_voc_id=None, answer_voc_id=None):
        tokens = self.stateful_model.tokens()[B0]
        #s2 = time.time()
        #tokens = self.stateful_model.tokens()[B0]
        probs = torch.nn.functional.softmax(logit_scores, dim=-1).cpu()
        entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=-1).cpu()
        sorted_indices = torch.argsort(logit_scores, dim=-1, descending=True)
        rank_subj = (sorted_indices == subj_voc_id).nonzero(as_tuple=True)[1].cpu() # probably one?!
        rank_ans = (sorted_indices == answer_voc_id).nonzero(as_tuple=True)[1].cpu()
        results = []
        
        for token_idx in range(len(entropy)):
            result = {}
            result["token_idx"] = token_idx
            # tok = self.stateful_model.tokens_to_strings(sorted_indices[token_idx][:self._logit_lens_topK].cpu())
            result["top_tokens"] = [int(i) for i in sorted_indices[token_idx][:self._logit_lens_topK].cpu()]
            # result["rank_subject"] = get_val((sorted_indices[token_idx] == subj_voc_id).nonzero(as_tuple=True)[0])
            # result["rank_answer"] = get_val((sorted_indices[token_idx] == answer_voc_id).nonzero(as_tuple=True)[0])
            result["rank_subject"] = get_val(rank_subj[token_idx])
            result["rank_answer"] = get_val(rank_ans[token_idx])
            result["logit_subject"] = get_val(logit_scores[token_idx][subj_voc_id].cpu())
            result["logit_answer"] = get_val(logit_scores[token_idx][answer_voc_id].cpu())
            result["prob_subject"] = get_val(probs[token_idx][subj_voc_id])
            result["prob_answer"] = get_val(probs[token_idx][answer_voc_id])
            result["max_logit"] = get_val(logit_scores[token_idx].max().cpu())
            result["max_prob"] = get_val(probs[token_idx].max())
            result["entropy"] = get_val(entropy[token_idx])
            results.append(result)
        return results
        

    @torch.no_grad()
    def run_logit_lens_on_resid(self, n_layers, subj_voc_id=None, answer_voc_id=None):
        full_results = {}
        for layer in range(n_layers):
            for resid_pos in ["pre", "mid"]:
                if resid_pos == "pre":
                    representations = self.stateful_model.residual_in(layer)
                else:
                    representations = self.stateful_model.residual_after_attn(layer)
                logit_scores = self._unembed(representations)[B0]
                hook_name = f"{layer}_{resid_pos}"
                full_results[hook_name] = self.process_logits(logit_scores, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
        representations = self.stateful_model.residual_out(n_layers-1)
        logit_scores = self._unembed(representations)[B0]
        full_results["final_post"] = self.process_logits(logit_scores, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
        return full_results
    

    @torch.no_grad()
    def run_logit_lens_on_outputs(self, n_layers, subj_voc_id=None, answer_voc_id=None):
        full_results = {}
        representations = self.stateful_model.residual_in(0)
        logit_scores = self._unembed(representations)[B0]
        full_results["embed"] = self.process_logits(logit_scores, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
        for layer in range(n_layers):
            for resid_pos in ["attn", "mlp"]:
                if resid_pos == "attn":
                    representations = self.stateful_model._get_block(layer, "hook_attn_out")
                else:
                    representations = self.stateful_model.ffn_out(layer)
                logit_scores = self._unembed(representations)[B0]
                hook_name = f"{layer}_{resid_pos}_out"
                full_results[hook_name] = self.process_logits(logit_scores, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
        return full_results
    

    @torch.no_grad()
    def run_logit_lens_on_heads(self, n_layers, n_heads, subj_voc_id=None, answer_voc_id=None):
        full_results = {}
        for layer in range(n_layers):
            representations = self.stateful_model._get_block(layer, "attn.hook_result")
            for head in range(n_heads):
                hook_name = f"L{layer}H{head}"
                logit_scores = self._unembed(representations[:, :, head])[B0]
                full_results[hook_name] = self.process_logits(logit_scores, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
        return full_results
    
    @torch.no_grad()
    def compute_neuron_contributions(self, n_layers):
        tokens = self.stateful_model.tokens()[B0]

        ffn_contributions = []
        for layer in range(n_layers):
            hook_name = f"L{layer}"
            resid_mid = self.stateful_model.residual_after_attn(layer)[B0]
            resid_post = self.stateful_model.residual_out(layer)[B0]
            decomposed_ffn = self.stateful_model.decomposed_ffn_out(B0, layer, -1)
            results = []
            for token in range(len(tokens)):
                c_ffn, _ = contributions.get_decomposed_mlp_contributions(resid_mid[token], resid_post[token], decomposed_ffn[token], renormalizing_threshold=0.01)
                results.append(c_ffn)
            ffn_contributions.append(torch.stack(results))
        return torch.stack(ffn_contributions).transpose(1, 0)


    @torch.no_grad()
    def run_logit_lens_on_neurons(self, n_layers, sel_neurons_layerwise, subj_voc_id=None, answer_voc_id=None):
        full_results = {}
        for layer in range(n_layers):
            #sel_neurons = sel_neurons_layerwise[0]
            for neuron in sel_neurons_layerwise:
                hook_name = f"L{layer}N{neuron}"
                representations = self.stateful_model.neuron_output(layer, int(neuron)).unsqueeze(0).unsqueeze(1)
                logit_scores = self._unembed(representations)[B0][0]
                probs = torch.nn.functional.softmax(logit_scores, dim=-1).cpu()
                entropy = (-probs * torch.log(probs + 1e-10)).sum(dim=-1).cpu()
                sorted_indices = torch.argsort(logit_scores, dim=-1, descending=True).cpu()

                result = {}
                #result["top_tokens"] = [w for w in self.stateful_model.tokens_to_strings(sorted_indices[:self._logit_lens_topK])]
                result["top_tokens"] = [int(i) for i in sorted_indices[:self._logit_lens_topK].cpu()]
                result["rank_subject"] = get_val((sorted_indices == subj_voc_id).nonzero(as_tuple=True)[0])
                result["rank_answer"] = get_val((sorted_indices == answer_voc_id).nonzero(as_tuple=True)[0])
                result["logit_subject"] = get_val(logit_scores[subj_voc_id].cpu())
                result["logit_answer"] = get_val(logit_scores[answer_voc_id].cpu())
                result["prob_subject"] = get_val(probs[subj_voc_id])
                result["prob_answer"] = get_val(probs[answer_voc_id])
                result["max_logit"] = get_val(logit_scores.max().cpu())
                result["max_prob"] = get_val(probs.max().cpu())
                result["entropy"] = get_val(entropy)

                full_results[hook_name] = result
        return full_results


    def process_sentences(self, sentences, all_subject_positions, facts, subjects, targets, relation, indices):
        relation_sentence_analyses = []  # List to hold analyses for the current relation
        print(f"Relation {relation}")
        for idx, sent in tqdm(enumerate(sentences)):
            start_time = time.time()
            self.sentence = sent
            self.subj_token = subjects[idx] 
            self.obj_token = targets[idx]

            # Run inference
            self.run_inference()

            tokens = self.stateful_model.tokens()[B0]
            subj_tokens = self.stateful_model.subj_tokens()[B0]
            obj_tokens = self.stateful_model.obj_token()[B0]
            n_tokens = tokens.shape[0]
            #print(f"Number of Tokens: {n_tokens}")
            
            subj_token_span = find_span_indices(tokens, subj_tokens)
            obj_token_span = find_span_indices(tokens, obj_tokens)

            subj_voc_id = subj_tokens[0].cpu() #tokens[all_subject_positions[idx]].cpu()
            answer_voc_id = obj_tokens[0].cpu() #tokens[-1].cpu()

            model_info = self.stateful_model.model_info()

            # Build contribution graphs
            graphs = cached_build_paths_to_predictions(
                self._graph,
                model_info.n_layers,
                n_tokens,
                range(n_tokens),
                self._contribution_threshold,
            )
            #print(f"Done with Graph {time.time() - start_time}")

            # Run logit lens on various outputs
            resid_logit_lens_results = self.run_logit_lens_on_resid(model_info.n_layers, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
            output_logit_lens_results = self.run_logit_lens_on_outputs(model_info.n_layers, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)
            
            # Create sentence analysis dictionary
            sentence_analysis = {
                "sentence": self.sentence,
                "tokens": tokens.tolist(),
                "subject_tokens": subj_tokens.tolist(),
                "answer_tokens": obj_tokens.tolist(),
                "data_idx": indices[idx],
                "subj_token_span": subj_token_span,   
                "answer_token_span": obj_token_span,               
                "contributions": self._contributions_dict,
                "full_graph": self._graph.copy(),
                "token_subgraphs": graphs,
                "logit_lens_result": {
                   "resid": resid_logit_lens_results,
                   "output": output_logit_lens_results,
                }
            }
            if self._do_head_level:
                heads_logit_lens_results = self.run_logit_lens_on_heads(model_info.n_layers, model_info.n_heads, subj_voc_id=subj_voc_id, answer_voc_id=answer_voc_id)#, subj_token_span=subj_token_span, obj_token_span=obj_token_span)
                sentence_analysis["logit_lens_result"]["heads"] = heads_logit_lens_results

            # If neuron level analysis is enabled
            if self._do_neuron_level:
                neuron_contributions = self.compute_neuron_contributions(model_info.n_layers)
                # top_neuron_contvals, top_neuron_indices = torch.topk(neuron_contributions, k=self._logit_lens_topK_neurons)
                # top_neuron_indices = top_neuron_indices.cpu().numpy()
                # top_neuron_contvals = top_neuron_contvals.cpu().to(torch.float16).numpy()
                
                n_tokens, n_heads, n_neurons = neuron_contributions.shape
                all_neuron_indices = torch.arange(n_neurons, device=neuron_contributions.device)#.unsqueeze(0).unsqueeze(0)
                #all_neuron_indices = all_neuron_indices.expand(n_tokens, n_heads, -1)

                # Convert to numpy if needed
                all_neuron_indices = all_neuron_indices.cpu().numpy()  # Shape: (n_tokens, 32, 11008)

                # sel_neurons_layerwise = []
                # for layer in range(model_info.n_layers):
                #     sel_neurons_layerwise.append([])
                #     for token in range(n_tokens):
                #         sel_neurons_layerwise[-1].extend(top_neuron_indices[token, layer].tolist())
                #     sel_neurons_layerwise[-1] = list(set(sel_neurons_layerwise[-1]))

                run_logit_lens_on_neurons = self.run_logit_lens_on_neurons(
                    model_info.n_layers,
                    sel_neurons_layerwise=all_neuron_indices,
                    subj_voc_id=subj_voc_id,
                    answer_voc_id=answer_voc_id
                )
                #print(f"Done with Neurons {time.time() - start_time}")
                sentence_analysis["neuron_contributions"] = {
                    "vals": neuron_contributions,
                    "ind": all_neuron_indices
                }
                sentence_analysis["logit_lens_result"]["neurons"] = run_logit_lens_on_neurons

            # Append sentence analysis for this sentence to the relation list
            relation_sentence_analyses.append(sentence_analysis)

        return relation_sentence_analyses


    def run(self, args):

        self.load_config(args.config)
        
        self._stateful_model = load_model(
            model_name=self.model_name,
            revision = args.revision,
            prepend_bos=self._prepend_bos,
            _model_path=self._model_path,
            _device=self.device,
            _dtype=self.dtype,
        )
        
        relations = load_json_files(args.dataset_path, args.category)
        all_sentence_analyses = {}
        
        # Create the base output directory (based on model name)
        base_output_dir = args.output_path
        
        # Create the revision-specific directory
        revision_output_dir = os.path.join(base_output_dir, str(args.revision))

        with torch.inference_mode():
            if args.relation:
                # Check if file already exists before processing
                if not os.path.exists(os.path.join(revision_output_dir, f"{args.relation}.pkl")):
                    sentences, facts, subjects, targets, all_subject_positions, indices = parse_samples(relations[args.relation])
                    relation_sentence_analyses = self.process_sentences(
                        sentences, all_subject_positions, facts, subjects, targets, args.relation, indices
                    )
                    all_sentence_analyses[args.relation] = relation_sentence_analyses
                    save_analysis_per_relation(relation_sentence_analyses, args.relation, revision_output_dir)
                else:
                    print(f"Skipping processing for relation '{args.relation}' as it is already saved.")

                # sentences, facts, subjects, targets, all_subject_positions, indices = parse_samples(relations[args.relation])
                # #print(len(sentences))
                # relation_sentence_analyses = self.process_sentences(
                #     sentences, all_subject_positions, facts, subjects, targets, args.relation, indices
                # )
                # all_sentence_analyses[args.relation] = relation_sentence_analyses
                # save_analysis_per_relation(relation_sentence_analyses, args.relation, revision_output_dir)
            else:
                
                for relation, relation_data in tqdm(relations.items(), desc="Processing Relations"):
                    # Skip if the analysis has already been saved
                    if os.path.exists(os.path.join(revision_output_dir, f"{relation}.pkl")):
                        print(f"Skipping processing for relation '{relation}' as it is already saved.")
                        continue
                    
                    sentences, facts, subjects, targets, all_subject_positions, indices = parse_samples(relation_data)
                    # Uncomment the line below if needed to see the count of sentences
                    # print(len(sentences))
                    
                    relation_sentence_analyses = self.process_sentences(
                        sentences, all_subject_positions, facts, subjects, targets, relation, indices
                    )
                    all_sentence_analyses[relation] = relation_sentence_analyses
                    
                    save_analysis_per_relation(relation_sentence_analyses, relation, revision_output_dir)
                
                # # Process all relations
                # for relation, relation_data in tqdm(relations.items(), desc="Processing Relations"):
                #     #print(f"Relation {relation}")
                #     sentences, facts, subjects, targets, all_subject_positions, indices = parse_samples(relation_data)
                #     print(len(sentences))
                #     relation_sentence_analyses = self.process_sentences(
                #         sentences, all_subject_positions, facts, subjects, targets, relation, indices
                #     )
                #     all_sentence_analyses[relation] = relation_sentence_analyses
                    
                #     save_analysis_per_relation(relation_sentence_analyses, relation, revision_output_dir)

        # Return or save all_sentence_analyses as needed
        return all_sentence_analyses
        
        print("Done")
            # If a specific relation is provided, process only that relation
        #     if args.relation:
        #         sentences, facts, subjects, targets, all_subject_positions, indices = parse_samples(relations[args.relation])
        #         #print(len(sentences))
        #         relation_sentence_analyses = self.process_sentences(
        #             sentences, all_subject_positions, facts, subjects, targets, args.relation, indices
        #         )
        #         all_sentence_analyses[args.relation] = relation_sentence_analyses
        #         save_analysis_per_relation(relation_sentence_analyses, args.relation, revision_output_dir)
        #     else:
        #         # Process all relations
        #         for relation, relation_data in tqdm(relations.items(), desc="Processing Relations"):
        #             #print(f"Relation {relation}")
        #             sentences, facts, subjects, targets, all_subject_positions, indices = parse_samples(relation_data)
        #             print(len(sentences))
        #             relation_sentence_analyses = self.process_sentences(
        #                 sentences, all_subject_positions, facts, subjects, targets, relation, indices
        #             )
        #             all_sentence_analyses[relation] = relation_sentence_analyses
                    
        #             save_analysis_per_relation(relation_sentence_analyses, relation, revision_output_dir)

        # # Return or save all_sentence_analyses as needed
        # return all_sentence_analyses
        
        # print("Done")

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=str, required=True, help='Model revision to use')
    parser.add_argument('--dataset_path', type=str, required=True, help='Dataset to use')
    parser.add_argument('--category', type=str, required=False, help='Select from Main Categories: Factual, Commonsense, Bias, Linguistic')
    parser.add_argument('--relation', type=str, required=False, help='Specific relation from Category')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for saved pickle file')
    parser.add_argument(
        '--config', type=str, required=False,
        default=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "exp_olmo_config.json")),
        help='Path to the analysis config JSON (default: config/exp_olmo_config.json next to this script)',
    )
    args = parser.parse_args()
    
    
    app = App()
    app.run(args)

