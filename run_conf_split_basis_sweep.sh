#!/bin/bash

N_CLIENTS=500
SAMPLE_RATE=0.02       # Sample and aggregate 10 clients each iteration
#SAMPLE_RATE=0.03      # Sample and aggregate 15 clients each iteration
#SAMPLE_RATE=0.04      # Sample and aggregate 20 clients each iteration
#SAMPLE_RATE=0.06      # Sample and aggregate 30 clients each iteration
EPOCHS=100             # number of federated iterations
DATASET="CIFAR10"
DELTA=0.002           # 1/500

# Check if an argument is passed
if [ -z "$1" ]; then
  echo "No argument provided!"
  echo "Usage: $0 <argument>"
  exit 1
fi

# Store the argument in a variable
GPU="$1"
NOISE_MULTIPLIER="$2"
SEED=43

LOCAL_EPOCHS=5         # Each client iterate  times over all his local data

# Print the argument
echo GPU "$GPU" SEED "$SEED" LOCAL_EPOCHS "${LOCAL_EPOCHS}"


EVAL_AFTER=10
EVAL_EVERY=5
LOG_EVERY=10
#BASIS_SIZE=400
PUBLIC_CLIENTS=10                    # clients used for subspace compute
#VIRTUAL_PUBLICS=400
#HISTORY_SIZE=1000
#GEP_ARGUMENTS=(
#          --num_public_clients "${PUBLIC_CLIENTS}"
#          --basis_size "${BASIS_SIZE}"
#          --history_size "${HISTORY_SIZE}"
#          --virtual_publics "${VIRTUAL_PUBLICS}"
#          )

#noise_multipliers=(0.5 1.0 2.0 2.5)
#clips=(0.1  0.001)
#clips_residual=(0.1  0.001)
#virtual_publics_list=(50)

#for virtual_publics in "${virtual_publics_list[@]}"; do
#  max_history_size=$(echo "$virtual_publics * 20" | bc)
#  min_history_size=$(echo "$virtual_publics * 2" | bc)

CLIPPING_BOUND=0.001
CLIPPING_BOUND_RESIDUAL=0.1
NOISE_MULTIPLIER_RESIDUAL=0.1
NOISE_MULTIPLIER_GRAD="${NOISE_MULTIPLIER}"

echo NOISE_MULTIPLIER_GRAD "${NOISE_MULTIPLIER_GRAD}"
echo NOISE_MULTIPLIER_RESIDUAL "${NOISE_MULTIPLIER_RESIDUAL}"
echo CLIPPING_BOUND "${CLIPPING_BOUND}"
echo CLIPPING_BOUND_RESIDUAL "${CLIPPING_BOUND_RESIDUAL}"


basis_size_list=(50 1100)
history_size_list=(1200)
virtual_publics_list=(10)


#=
for virtual_publics in "${virtual_publics_list[@]}"; do
  for history_size in "${history_size_list[@]}"; do
    for basis_size in "${basis_size_list[@]}"; do

      HISTORY_SIZE=$history_size
      BASIS_SIZE=$basis_size
      VIRTUAL_PUBLICS=$virtual_publics

  #    for NOISE_MULTIPLIER_RESIDUAL in $(seq 0.5 0.5 "$NOISE_MULTIPLIER"); do
  #      for grad_clip in "${clips[@]}"; do
  #        for res_clip in "${clips_residual[@]}"; do

      GEP_ARGUMENTS=(
      --num_public_clients "${PUBLIC_CLIENTS}"
      --basis_size "${BASIS_SIZE}"
      --history_size "${HISTORY_SIZE}"
      --virtual_publics "${VIRTUAL_PUBLICS}"
      )

      echo  GEP_ARGUMENTS  "${GEP_ARGUMENTS[@]}"


  #    TOTAL_NOISE_SQUARED="$(echo "scale=4; $NOISE_MULTIPLIER^2" | bc -l)"
  #    echo TOTAL_NOISE_SQUARED "${TOTAL_NOISE_SQUARED}"
  #    NOISE_MULTIPLIER_RESIDUAL_SQUARED="$(echo "scale=4; $NOISE_MULTIPLIER_RESIDUAL^2" | bc -l)"
  #    echo NOISE_MULTIPLIER_RESIDUAL_SQUARED "${NOISE_MULTIPLIER_RESIDUAL_SQUARED}"
  #    NOISE_MULTIPLIER_GRAD_SQUARED="$(echo "${TOTAL_NOISE_SQUARED} - ${NOISE_MULTIPLIER_RESIDUAL_SQUARED}" | bc)"
  #    echo NOISE_MULTIPLIER_GRAD_SQUARED "${NOISE_MULTIPLIER_GRAD_SQUARED}"
  #    NOISE_MULTIPLIER_GRAD="$(echo "scale=4; sqrt($NOISE_MULTIPLIER_GRAD_SQUARED)" | bc)"



      EXP_NAME=virtual_"${VIRTUAL_PUBLICS}"_basis_"${BASIS_SIZE}"_history_"${HISTORY_SIZE}"_noise_grad"${NOISE_MULTIPLIER_GRAD}"_noise_res_"${NOISE_MULTIPLIER_RESIDUAL}"_clip_"${CLIPPING_BOUND}"_clip_res_"${CLIPPING_BOUND_RESIDUAL}"_seed_"${SEED}"_epochs_"${EPOCHS}"_local_epochs"${LOCAL_EPOCHS}"
      EXP_NAME=GEP_RES_"${EXP_NAME}"

      LOG_FOLDER="logs/${DATASET}_${EXP_NAME}"
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
             --clipping_bound_residual "${CLIPPING_BOUND_RESIDUAL}"
             --noise_multiplier "${NOISE_MULTIPLIER_GRAD}"
             --noise_multiplier_residual "${NOISE_MULTIPLIER_RESIDUAL}"
             --eval-after "${EVAL_AFTER}"
             --eval-every "${EVAL_EVERY}"
             --log-every "${LOG_EVERY}"
             --exp-name "${EXP_NAME}"
            )




      echo  ${DATASET} FedAvg GEP RESIDUAL VIRTUAL SINGLE LOADER  with arguments "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"
      CUDA_VISIBLE_DEVICES=$GPU python main_base_gep_gp.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"
      #   CUDA_VISIBLE_DEVICES=$GPU python main_base_gep_gp.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"  >> "${LOG_FOLDER}"/FedAvgGep_"${EXP_NAME}".txt
    done
  done
done


