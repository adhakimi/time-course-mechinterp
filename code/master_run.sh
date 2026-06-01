# # revisions=$(ssh adhakimi@pi.cis.lmu.de 'ls /mounts/data/proj/olmo_models/models--allenai--OLMo-7B-0424-hf/refs')
# revision='step651581-tokens2731B'
# # for revision in $revisions; do

# #     step=$(echo $revision | sed -E 's/step([0-9]+)-.*/\1/')

# #     # Check if the revision is "step115000-tokens482B" (exclude it)
# #     if [[ $revision == "step115000-tokens482B" ]]; then
# #         continue
# #     fi

#     # Check if the step is divisible by 5000 and within the desired range  (( $step % 5000 == 0 )) && (( step <= 200000 )) || 
# if [[ $revision == "step651581-tokens2731B" ]]; then
#     echo "Processing revision: $revision with step: $step"

#     sbatch --export=revision=$revision run.sh
# fi

# #done

#!/bin/bash

# Define the five revisions as an array
# revisions=(
#   "step651581-tokens2731B"
#   "step650650-tokens2728B"
#   "step649650-tokens2723B"
#   "step648650-tokens2719B"
#   "step647650-tokens2715B"
# )

revisions=(
  "step651581-tokens2731B"
  "step650650-tokens2728B"
  "step649650-tokens2723B"
  "step648650-tokens2719B"
  "step647650-tokens2715B"
)

# Loop over each revision
for revision in "${revisions[@]}"; do
    # Extract the step number from the revision name
    step=$(echo "$revision" | sed -E 's/step([0-9]+)-.*/\1/')

    echo "Processing revision: $revision with step: $step"

    # Submit the job with the current revision as an environment variable
    sbatch --export=revision="$revision" run.sh
done