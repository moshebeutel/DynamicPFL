#!/bin/bash

N_CLIENTS=512
SAMPLE_RATE=0.015625  # Sample and aggregate 8 clients each iteration
EPOCHS=128            # 1000 iterations of 8 sampled clients
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
EPS="$3"
SEED=42
EXP_NAME="${DATASET}"_epsilon_"${EPS}"_clip_"${CLIPPING_BOUND}"_seed_"${SEED}"
ARGUMENTS=(--dataset "${DATASET}" --num_clients "${N_CLIENTS}" --user_sample_rate "${SAMPLE_RATE}" --global_epoch "${EPOCHS}" --local_epoch "${LOCAL_EPOCHS}" --target_epsilon "${EPS}" --target_delta "${DELTA}")

echo "GPU $GPU CLIPPING_BOUND: $CLIPPING_BOUND  EPS $EPS"

LOG_FOLDER="logs/${DATASET}_${PUBLIC_CLIENTS}_public_${EPOCHS}_epochs__${SAMPLE_RATE}_sample_rate_history_sweep"
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


#echo ${DATASET} FedAvg SGD_DP
#python main_base.py "${ARGUMENTS[@]}"  >> ${LOG_FOLDER}/FedAvgSgdDP.txt

#echo "$DATASET" DynamicPFL
#python ours.py "${ARGUMENTS[@]}"  >> ${LOG_FOLDER}/DynamicPFL.txt

history_sizes=(200  800)
basis_sizes=(50 100 200)
for hs in "${history_sizes[@]}"; do
  for bz in "${basis_sizes[@]}"; do

    BASIS_SIZE="${bz}"
    HISTORY_SIZE="${hs}"

    EXP_NAME=GEP_"${DATASET}"_history_size_"${HISTORY_SIZE}"_basis_size_"${BASIS_SIZE}"_clip_"${CLIPPING_BOUND}"_seed_"${SEED}"
    ARGUMENTS=(
           --dataset "${DATASET}"
           --num_clients "${N_CLIENTS}"
           --user_sample_rate "${SAMPLE_RATE}"
           --global_epoch "${EPOCHS}"
           --local_epoch "${LOCAL_EPOCHS}"
           --target_epsilon "${EPS}"
           --target_delta "${DELTA}"
           --seed "${SEED}"
           --clipping_bound "${CLIPPING_BOUND}"
           --exp-name "${EXP_NAME}"
          )
    echo "${ARGUMENTS[@]}"
    GEP_ARGUMENTS=(--num_public_clients "${PUBLIC_CLIENTS}" --basis_size "${BASIS_SIZE}" --history_size "${HISTORY_SIZE}")

    echo  ${DATASET} FedAvg GEP history size "${HISTORY_SIZE}"
    python main_base_gep.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"  >> ${LOG_FOLDER}/FedAvgGep_history_"${HISTORY_SIZE}".txt


    #  echo "$DATASET" DynamicPFL GEP history size ${HISTORY_SIZE}
    #  python ours_gep.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"  >> ${LOG_FOLDER}/DynamicPFL_GEP_history_${HISTORY_SIZE}.txt
  done
done
