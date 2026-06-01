#!/bin/bash
#SBATCH --output=./logs/neuron_logic_analysis.py_%j.out       # Adjusted output log name for clarity
#SBATCH --error=./logs/neuron_logic_analysis.py_%j.err        # Adjusted error log name for clarity
#SBATCH --time=0-04:00:00                          # Increased time limit for large downloads
#SBATCH --partition=mcml-hgx-a100-80x4-mig         # Partition remains the same
#SBATCH --qos=mcml                                 # QoS remains the same
#SBATCH --mem=520G                                 # Memory allocation remains the same
#SBATCH --cpus-per-task=16                         # Specify the number of CPUs for multiprocessing
#SBATCH --job-name=clean_shards
#SBATCH --gres=gpu:1

source "$(dirname "$(readlink -f "$0")")/../paths.env"
source ~/.bashrc
conda activate "$CONDA_ENV"

cd "$REPO_ROOT/code"
python -u neuron_logic_analysis.py