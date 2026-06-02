#!/bin/bash
#SBATCH --output=./logs/llm_pt_analysis_%j.out
#SBATCH --error=./logs/llm_pt_analysis_%j.err
#SBATCH --time=0-04:00:00
#SBATCH --partition=mcml-hgx-a100-80x4-mig
#SBATCH --qos=mcml
#SBATCH --mem=250G
#SBATCH --gres=gpu:3g.40gb:1

source "$(dirname "$(readlink -f "$0")")/../paths.env"
source ~/.bashrc
conda activate "$CONDA_ENV"

dataset_path="$REPO_ROOT/dataset/relations_data"
output_path="$DATA_ROOT/new_outputs"

#export HF_HOME="$REPO_ROOT/../shared-hf-cache"
#echo $HF_HOME
echo $output_path
local_snapshot_dir="$DATA_ROOT/revisions_temp/models--allenai--OLMo-7B-0424-hf/snapshots/"
#export HF_HUB_OFFLINE=1

#revisions=$(ssh adhakimi@pi.cis.lmu.de 'ls /mounts/data/proj/olmo_models/models--allenai--OLMo-7B-0424-hf/refs')

#echo $revisions

# 1- Copy the specified revision
# 1.1- Figure out the revision location based on the snapshot hash (copy the snapshot)

echo "Processing revision: $revision"

#revision_hash=$(ssh adhakimi@pi.cis.lmu.de "cat /mounts/data/proj/olmo_models/models--allenai--OLMo-7B-0424-hf/refs/$revision")

#echo "Resolved hash for $revision: $revision_hash"
#scp -r adhakimi@pi.cis.lmu.de:/mounts/data/proj/olmo_models/models--allenai--OLMo-7B-0424-hf/snapshots/$revision_hash $local_snapshot_dir

# if [ ! -d "$local_snapshot_dir/$revision_hash" ]; then
#     echo "Snapshot not found locally, downloading..."
#     scp -r adhakimi@pi.cis.lmu.de:/mounts/data/proj/olmo_models/models--allenai--OLMo-7B-0424-hf/snapshots/$revision_hash $local_snapshot_dir
# else
#     echo "Snapshot already exists locally, skipping download."
# fi


# 2- Run the analysis
# 2.1 For loop on a specified relations (category is factual) - the list would be specified in this file
python -u "$REPO_ROOT/code/llm-transparency-tool/llm_transparency_tool/server/app.py" \
        --revision $revision \
        --dataset_path $dataset_path \
        --category semantic \
        --output_path $output_path

#done

# 2.2 specify the arguments based on the script args

# 3- Remove the revision
# echo "Removing snapshot for $revision ($revision_hash) to prevent memory overflow"
# rm -r $local_snapshots_dir/$revision_hash
# remove the snapshot copied in first step
# rm -r  