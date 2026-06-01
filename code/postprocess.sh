#!/bin/bash
#SBATCH --output=./logs/postprocess_%j.out
#SBATCH --error=./logs/postprocess_%j.err
#SBATCH --time=0-01:00:00
#SBATCH --partition=mcml-hgx-a100-80x4-mig
#SBATCH --qos=mcml
#SBATCH --cpus-per-task=90
#SBATCH --mem=500G
#SBATCH --gres=gpu:3g.40gb:1

source "$(dirname "$(readlink -f "$0")")/../paths.env"
source ~/.bashrc
conda activate "$CONDA_ENV"

cd "$REPO_ROOT/code"
python -u new_postprocess.py #postprocessing.py