#!/bin/bash

N_CLIENTS=512
SAMPLE_RATE=0.015625  # Sample and aggregate 8 clients each iteration
EPOCHS=960            # 1000 iterations of 8 sampled clients
LOCAL_EPOCHS=4        # Each client iterate 4 times over all his local data
DATASET="CIFAR10"
DELTA=0.001953125     # 1/512
PUBLIC_CLIENTS=10     # clients used for subspace compute

# Check if an argument is passed
if [ -z "$1" ]; then
  echo "No argument provided!"
  echo "Usage: $0 <argument>"
  exit 1
fi

# Store the argument in a variable
GPU="$1"
CLIPPING_BOUND="$2"
SEED=42

# Print the argument
echo "GPU $GPU CLIPPING_BOUND: $CLIPPING_BOUND  EPS $EPS"

LOG_FOLDER="logs/${DATASET}_${PUBLIC_CLIENTS}_public_${EPOCHS}_epochs__${SAMPLE_RATE}_sample_rate_total_sweep"
echo "$LOG_FOLDER"
# Check if the folder exists
if [ ! -d "$LOG_FOLDER" ]; then
    # If the folder does not exist, create it
    mkdir -p "$LOG_FOLDER"
    echo "Folder created at $LOG_FOLDER"
else
    # If the folder exists, output a message
    echo "Folder already exists at $LOG_FOLDER"
fi

noise_multiplers=(1.7 1.8 1.9 2.0)
for np in "${noise_multiplers[@]}"; do

  EXP_NAME="${DATASET}"_noise_multipler_"${np}"_clip_"${CLIPPING_BOUND}"_seed_"${SEED}"
  ARGUMENTS=(
         --dataset "${DATASET}"
         --num_clients "${N_CLIENTS}"
         --user_sample_rate "${SAMPLE_RATE}"
         --global_epoch "${EPOCHS}"
         --local_epoch "${LOCAL_EPOCHS}"
#         --target_epsilon "${EPS}"
         --target_delta "${DELTA}"
         --seed "${SEED}"
         --clipping_bound "${CLIPPING_BOUND}"
         --noise_multiplier "${np}"
         --exp-name "${EXP_NAME}"
        )
  echo "${ARGUMENTS[@]}"

  echo Run FedAvg SGD_DP with arguments "${ARGUMENTS[@]}"
  CUDA_VISIBLE_DEVICES=$GPU python main_base.py "${ARGUMENTS[@]}" >> "${LOG_FOLDER}"/FedAvgSgdDP_"${EXP_NAME}".txt

  BASIS_SIZE=10
  HISTORY_SIZE=10
  GEP_ARGUMENTS=(--num_public_clients "${PUBLIC_CLIENTS}" --basis_size "${BASIS_SIZE}" --history_size "${HISTORY_SIZE}")
  EXP_NAME=GEP_"${EXP_NAME}"

  ARGUMENTS=(
         --dataset "${DATASET}"
         --num_clients "${N_CLIENTS}"
         --user_sample_rate "${SAMPLE_RATE}"
         --global_epoch "${EPOCHS}"
         --local_epoch "${LOCAL_EPOCHS}"
#         --target_epsilon "${EPS}"
         --target_delta "${DELTA}"
         --seed "${SEED}"
         --clipping_bound "${CLIPPING_BOUND}"
         --noise_multiplier "${np}"
         --exp-name "${EXP_NAME}"
        )
  echo "${ARGUMENTS[@]}"


  echo  ${DATASET} FedAvg GEP with arguments "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"
  CUDA_VISIBLE_DEVICES=$GPU python main_base_gep.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"  >> "${LOG_FOLDER}"/FedAvgGep_"${EXP_NAME}".txt
done



