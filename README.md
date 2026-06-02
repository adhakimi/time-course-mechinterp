# Time Course MechInterp — Pretraining-Step Analysis of OLMo-7B

Code release for [**Time Course MechInterp: Analyzing the Evolution of Components and Knowledge in Large Language Models**](https://aclanthology.org/2025.findings-acl.654/) (Hakimi, Modarressi, Wicke, Schütze; Findings of ACL 2025). This repository contains the pipeline used to trace how internal mechanisms in a large language model develop over the course of pretraining, by running mechanistic-interpretability probes on every available checkpoint of `allenai/OLMo-7B-0424-hf`.

The analysis is built on top of Meta's [llm-transparency-tool](https://github.com/facebookresearch/llm-transparency-tool) and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), both vendored under `code/` with project-specific modifications (notably a non-Streamlit batch driver in `app.py`).

> The pipeline was also adapted for `EleutherAI/pythia-6.9b`, but the information-flow / contribution-graph routes in `llm-transparency-tool` proved incompatible with that architecture, so all reported results are on OLMo. `download_pythia.py` is kept as a record of that attempt.

---

## Pipeline overview

```
 ┌──────────────────┐    ┌────────────────────────┐    ┌──────────────────┐    ┌───────────────────────┐
 │ download model   │ →  │ per-checkpoint analysis │ →  │ postprocess pkl  │ →  │ cross-checkpoint      │
 │ revisions        │    │ (logit lens, contrib    │    │ (strip heavy     │    │ analysis (IoU of      │
 │ (HF snapshots)   │    │  graph, head probes)    │    │  fields, ranks)  │    │  specialized heads)   │
 └──────────────────┘    └────────────────────────┘    └──────────────────┘    └───────────────────────┘
```

1. **Checkpoint acquisition** — snapshots of every OLMo revision (`step5000-tokens20B` … `step651581-tokens2731B`, plus `main`).
2. **Per-checkpoint, per-relation analysis** — `code/llm-transparency-tool/llm_transparency_tool/server/app.py` runs each relation's prompt set through the model and records, for every example:
   - tokenization + subject/answer token spans,
   - contribution graph and per-token subgraphs,
   - logit-lens probes on residual stream, attn/MLP outputs, attention heads (optionally neurons),
   - subject/answer ranks and top-K token IDs.
3. **Postprocessing** — `code/postprocessing.py` strips the heavy raw fields (`full_graph`, `token_subgraphs`, `logit_lens_result`, `neuron_contributions`) and writes compact arrays plus attn-head / FFN contribution maps. Subgraphs are split into a separate `*_subgraphs.pkl`.
4. **Cross-checkpoint analysis** — `code/neuron_logic_analysis.py` aggregates "general / entity / relation-answer / answer-specific" specialized-head heatmaps for every checkpoint, then computes IoU against `main` to plot how specialized circuits emerge and stabilize during training. `code/metrics.py` computes top-1 / top-10 logit-lens accuracy.
5. **Figures** — the notebooks in `notebooks/` read the postprocessed pickles and produce the paper's figures (head specialization & transitions, FFN switches, knowledge accuracy). They are the interactive counterparts to the batch scripts in `code/` and are what generated the published plots.

---

## Repository layout

```
.
├── README.md
├── run.sh                                  ← SLURM wrapper that invokes download_olmo.py
├── download_olmo.py                         ← HF snapshot download for OLMo-7B-0424-hf
├── download_pythia.py                       ← preserved attempt at Pythia-6.9b (unused; see note above)
│
├── dataset/
│   └── relations_data/                      ← prompt sets (JSON) per category
│       ├── semantic/                        ← **Factual Knowledge Dataset from the paper** (10 relations, LOC + NAME)
│       ├── factual/                         ← 27 auxiliary LRE-style relations (not used in the paper's main analysis)
│       ├── commonsense/
│       ├── bias/
│       ├── linguistic/
│       └── syn_data/
│
├── code/
│   ├── run.sh                               ← SLURM: per-revision analysis job
│   ├── master_run.sh                        ← dispatches one sbatch per revision
│   ├── postprocess.sh, slurm_run.sh         ← SLURM wrappers for postprocessing / cross-checkpoint
│   │
│   ├── postprocessing.py                    ← strips heavy fields, computes contribution maps
│   ├── new_postprocess.py                   ← variant postprocessor
│   ├── collapse_graphs.py                   ← collapses subgraphs across examples
│   ├── neuron_logic_analysis.py             ← cross-checkpoint specialized-head IoU plots
│   ├── metrics.py                           ← top-1 / top-10 logit-lens accuracy
│   ├── check_files.py                       ← integrity check over output pickles
│   │
│   ├── llm-transparency-tool/               ← vendored upstream + project-specific changes
│   │   ├── config/exp_olmo_config.json      ← model + analysis hyperparameters
│   │   └── llm_transparency_tool/
│   │       ├── server/app.py                ← non-Streamlit batch analysis entry point (custom)
│   │       └── routes/, models/             ← upstream code (lightly modified)
│   └── TransformerLens/                     ← vendored upstream
│
└── notebooks/                              ← interactive analysis → the paper's figures
    ├── spec_head_logic_analysis.ipynb      ← attention-head specialization / transition figures
    ├── spec_ffn_logic_analysis.ipynb       ← FFN component-switch figures
    ├── new_metrics.ipynb                   ← top-1 / top-10 knowledge-accuracy figures
    ├── clean_heads_final.ipynb             ← aggregated component-switch summary plot
    ├── utils.py                            ← postprocessed-pkl loader + plotting helpers
    └── old/                                ← superseded draft notebooks (outputs stripped)
```

---

## Setup

The pipeline targets a SLURM cluster with at least one A100 (≥40 GB) per analysis job.

```bash
# 1. Create the conda env (mirrors llm-transparency-tool's env.yaml).
conda env create -f code/llm-transparency-tool/env.yaml
conda activate llmtt

# 2. Install the vendored tools in editable mode.
pip install -e code/TransformerLens
pip install -e code/llm-transparency-tool
```

### Paths to update for your environment

All cluster-specific paths are centralized in **`paths.env`** at the repo root. Edit the three variables to match your environment:

```bash
# paths.env
export REPO_ROOT=/path/to/this/repo
export DATA_ROOT=/path/to/large-data/storage          # holds model snapshots, raw + processed outputs
export CONDA_ENV=/path/to/conda/envs/llmtt
```

`paths.env` lays out the subdirectories that the pipeline expects under `$DATA_ROOT` (`revisions_temp/`, `new_outputs/`, `new_processed_outputs/`, `neuron_outputs/`); the scripts will create them on first write.

#### Loading `paths.env`

- **SLURM jobs** (`sbatch run.sh`, `sbatch code/run.sh`, etc.) source `paths.env` automatically and `conda activate "$CONDA_ENV"` — nothing else needed.
- **Direct Python invocations** (`python code/postprocessing.py`, `python code/neuron_logic_analysis.py`, `python download_olmo.py`) require the environment to be loaded first. Run `source paths.env` once per shell session, or add it to your `~/.bashrc` if you use this repo often:
  ```bash
  source paths.env
  python code/postprocessing.py
  ```
  Python scripts read `DATA_ROOT` via `os.environ["DATA_ROOT"]` and will raise a `KeyError` with that exact name if you forget — that's the signal to `source paths.env`.

The JSON config (`code/llm-transparency-tool/config/exp_olmo_config.json`) uses the `${DATA_ROOT}` placeholder, which `app.py` expands via `os.path.expandvars` when loading.

---

## Usage

### 1. Download OLMo checkpoints

`download_olmo.py` pulls revisions of `allenai/OLMo-7B-0424-hf` into `DOWNLOAD_DIR` (set near the top of the file). By default every branch is fetched in parallel; set `STEP_INCREMENT` to e.g. `5000` to subsample.

```bash
sbatch run.sh                  # via SLURM (recommended for the full sweep)
python download_olmo.py        # directly
```

### 2. Run per-checkpoint analysis

Edit the `revisions=(…)` array in `code/master_run.sh` to the list of OLMo revisions you want to analyze, then:

```bash
cd code
bash master_run.sh
```

This submits one SLURM job per revision (via `code/run.sh`). Each job runs `app.py` against every relation in the selected category and writes one pickle per relation under `<output_path>/<revision>/<relation>.pkl`.

Direct invocation (one revision, one category):

```bash
python -u code/llm-transparency-tool/llm_transparency_tool/server/app.py \
    --revision step651581-tokens2731B \
    --dataset_path dataset/relations_data \
    --category semantic \
    --output_path /path/to/outputs
```

To reproduce the paper, use `--category semantic` (this is the Factual Knowledge Dataset; see the Dataset section below). Analysis hyperparameters (dtype, AMP, contribution threshold, head/neuron toggles, logit-lens top-K) live in `code/llm-transparency-tool/config/exp_olmo_config.json`.

### 3. Postprocess

Set `input_dir` and `output_dir` at the bottom of `code/postprocessing.py`, then:

```bash
python code/postprocessing.py
# or, on SLURM:
sbatch code/postprocess.sh
```

This strips heavy fields, emits compact rank arrays and per-head/FFN contribution maps, and writes a sibling `<relation>_subgraphs.pkl` per file.

### 4. Cross-checkpoint analysis and metrics

```bash
# IoU / intersection / union plots across checkpoints (writes PNGs under code/).
python code/neuron_logic_analysis.py

# Top-1 / top-10 logit-lens accuracy for a given revision.
# Note: reads logit_lens_result, so it must run on raw outputs (step 2), not the postprocessed tree.
python code/metrics.py --output_path <raw_outputs_dir> --revision main [--relation <name>]
```

`neuron_logic_analysis.py` walks the directory set in its `root_directory` constant. It expects raw outputs that include `neuron_contributions` — produced by an `app.py` run with `do_neuron_level: true` in `exp_olmo_config.json`. It emits `neuron_iou.png`, `neuron_intersection.png`, `neuron_union.png` and their `proper_*` variants (general / entity / relation-answer / answer-specific heads) under `code/`.

### 5. Figures (analysis notebooks)

The notebooks in `notebooks/` generate the paper's figures from the postprocessed pickles. They read the data directory from the `PROCESSED_OUTPUTS_DIR` environment variable, defaulting to `../new_processed_outputs/` (i.e. they assume Jupyter is launched from `notebooks/` and the data sits at the repo root). `notebooks/utils.py` provides the loader (`read_all_files`) and plotting helpers.

```bash
# Get the data first (see "Postprocessed outputs" below), then:
cd notebooks
export PROCESSED_OUTPUTS_DIR=../new_processed_outputs   # only needed if the data lives elsewhere
jupyter lab        # or: jupyter notebook
```

| Notebook | Produces |
|---|---|
| `spec_head_logic_analysis.ipynb` | attention-head specialization heatmaps, IoU/count, head transition probabilities |
| `spec_ffn_logic_analysis.ipynb`  | FFN component-switch figures (aggregated + per-relation) |
| `new_metrics.ipynb`              | top-1 / top-10 logit-lens knowledge accuracy over training |
| `clean_heads_final.ipynb`        | aggregated component-switch summary plot |

PDFs are written to `notebooks/final_figures2/` (git-ignored). The notebooks under `notebooks/old/` are earlier drafts kept for reference (outputs stripped); they are not needed to reproduce the paper.

---

## Postprocessed outputs (for the notebooks)

The figure notebooks consume the postprocessed circuit pickles — one `<relation>.pkl` per checkpoint, laid out as `new_processed_outputs/<revision>/<relation>.pkl` (40 OLMo revisions × 10 relations, ~2.5 GB). You can either:

1. **Regenerate** them from scratch by running the pipeline (steps 1–3 above), or
2. **Download** the prepared archive and drop it at the repo root:

   ```bash
   # Hugging Face Hub (recommended)
   pip install -U "huggingface_hub[cli]"
   hf download adhakimi/time-course-mechinterp-data \
       --repo-type dataset --local-dir new_processed_outputs

   # (alternative) Zenodo archive
   # curl -L -o new_processed_outputs.tar.gz <ZENODO_URL>
   # tar -xzf new_processed_outputs.tar.gz
   ```

After download, `new_processed_outputs/` should contain the `step*-tokens*/` (and `main/`) checkpoint directories directly. The `*_subgraphs.pkl` files are not required by the notebooks and can be omitted from the archive to save space.

> **Note:** the pickles store Python objects (NetworkX graphs, NumPy arrays), so loading them requires the same `networkx` / `numpy` versions used to create them — install the repo env (see Setup) before opening the notebooks.

---

## Dataset

The paper's **Factual Knowledge Dataset** (Section 3.1 of the paper) lives in `dataset/relations_data/semantic/`. It contains 10 manually curated relations grouped into Location-Based (LOC) and Name-Based (NAME), with prompt templates designed to minimize syntactic ambiguity (e.g. avoiding cases where the subject already contains the answer, or where multiple answers are valid). The dataset extends LRE (Hernandez et al., 2024), CounterFact (Meng et al., 2022), ParaRel (Elazar et al., 2021), and Summing Up The Facts (Chughtai et al., 2024); the two relations sourced from Goodreads and IMDb are new contributions of this work. Prompts and facts went through the multi-step validation pipeline described in Appendix B of the paper.

| Group | Relation | Prompt template | # Facts |
|---|---|---|---|
| LOC | `city_in_country` | `{}` is part of the country of | 14 |
| LOC | `company_hq` | The headquarters of `{}` are in the city of | 20 |
| LOC | `country_capital_city` | `{}` has the capital city of | 19 |
| LOC | `food_from_country` | `{}` is from the country of | 17 |
| LOC | `official_language` | In `{}`, the official language is | 14 |
| LOC | `plays_sport` | `{}` plays professionally in the sport of | 12 |
| LOC | `sights_in_city` | `{}` is a landmark in the city of | 17 |
| NAME | `books_written` | The Book `{}` was written by the author with the name of | 13 |
| NAME | `company_ceo` | Who is the CEO of `{}`? Their name is | 17 |
| NAME | `movie_directed` | The Movie `{}` was directed by the director with the name of | 17 |

**New relations introduced by this work** (rest are adapted from prior resources):

- `books_written` — sourced from Goodreads
- `movie_directed` — sourced from IMDb's Top Favorites list

The other sibling directories (`factual/`, `commonsense/`, `bias/`, `linguistic/`, `syn_data/`) hold auxiliary relations from the LRE family that the paper's main analysis does not use; they are kept here to make exploratory follow-ups easier.

### File schema

```json
{
  "name": "country capital city",
  "prompt_templates": ["{} has the capital city of"],
  "properties": { "relation_type": "factual", "domain_name": "country", "range_name": "city", "symmetric": false },
  "samples": [
    {"subject": "United States", "object": "Washington D.C."},
    {"subject": "France",        "object": "Paris"}
  ]
}
```

`app.py` only consumes `prompt_templates` and `samples`: it materializes one prompt per (template, subject) pair via `template.format(subject)` and runs the analysis pipeline on each.

---

## Acknowledgements

This work builds on:

- [llm-transparency-tool](https://github.com/facebookresearch/llm-transparency-tool) (Meta) — contribution-graph and logit-lens infrastructure
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — hooked transformer abstractions

Both libraries are vendored under `code/` (see their respective `LICENSE` files).

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{hakimi-etal-2025-time,
    title     = "Time Course {M}ech{I}nterp: Analyzing the Evolution of Components and Knowledge in Large Language Models",
    author    = "Hakimi, Ahmad Dawar  and
                 Modarressi, Ali  and
                 Wicke, Philipp  and
                 Schuetze, Hinrich",
    editor    = "Che, Wanxiang  and
                 Nabende, Joyce  and
                 Shutova, Ekaterina  and
                 Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month     = jul,
    year      = "2025",
    address   = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2025.findings-acl.654/",
    doi       = "10.18653/v1/2025.findings-acl.654",
    pages     = "12633--12653"
}
```
