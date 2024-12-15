#!/bin/bash

N_CLIENTS=500
#SAMPLE_RATE=0.02      # Sample and aggregate 10 clients each iteration
#SAMPLE_RATE=0.03      # Sample and aggregate 15 clients each iteration
#SAMPLE_RATE=0.04      # Sample and aggregate 20 clients each iteration
#SAMPLE_RATE=0.06      # Sample and aggregate 30 clients each iteration
SAMPLE_RATE=0.1        # Sample and aggregate 50 clients each iteration
EPOCHS=99              # number of federated iterations
DATASET="CIFAR10"
DELTA=0.002            # 1/500

# Check if an argument is passed
if [ -z "$1" ]; then
  echo "No argument provided!"
  echo "Usage: $0 <argument>"
  exit 1
fi

# Store the argument in a variable
GPU="$1"
NOISE_MULTIPLIER="$2"
SEED="$3"

#LOCAL_EPOCHS=5         # Each client iterate  times over all his local data

# Print the argument
echo GPU "$GPU" SEED "$SEED" LOCAL_EPOCHS "${LOCAL_EPOCHS}"


EVAL_AFTER=10
EVAL_EVERY=5
LOG_EVERY=5
#BASIS_SIZE=400
#PUBLIC_CLIENTS=50                    # clients used for subspace compute
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
#
#CLIPPING_BOUND=0.001
#CLIPPING_BOUND_RESIDUAL=0.1
NOISE_MULTIPLIER_RESIDUAL=0.1
NOISE_MULTIPLIER_GRAD="${NOISE_MULTIPLIER}"

echo NOISE_MULTIPLIER_GRAD "${NOISE_MULTIPLIER_GRAD}"
echo NOISE_MULTIPLIER_RESIDUAL "${NOISE_MULTIPLIER_RESIDUAL}"
echo CLIPPING_BOUND "${CLIPPING_BOUND}"
echo CLIPPING_BOUND_RESIDUAL "${CLIPPING_BOUND_RESIDUAL}"


global_lr_list=(0.9)
lr_list=(0.0001)
clipping_bound_list=(0.01)
clipping_bound_residual_list=(0.1)
public_clients_list=(10 100 200)
#public_clients_list=(10)
basis_size_list=(100)
history_size_list=(2000)
virtual_publics_list=(200 400)
local_epoch_list=(5)

#global_lr_list=(0.5 0.001)
#lr_list=(0.001 0.0001)
#clipping_bound_list=(0.001 0.0001)
#clipping_bound_residual_list=(0.1 0.001)
#public_clients_list=(10 200)
#basis_size_list=(10 200)
#history_size_list=(2000)
#virtual_publics_list=(200 400)
#local_epoch_list=(3)

for global_lr in "${global_lr_list[@]}"; do
  for lr in "${lr_list[@]}"; do
    for clipping_bound in "${clipping_bound_list[@]}"; do
      for clipping_bound_residual in "${clipping_bound_residual_list[@]}"; do
        for public_clients in "${public_clients_list[@]}"; do
          for virtual_publics in "${virtual_publics_list[@]}"; do
            for history_size in "${history_size_list[@]}"; do
              for basis_size in "${basis_size_list[@]}"; do
                for local_epoch in "${local_epoch_list[@]}"; do

                  GLOBAL_LR=$global_lr
                  LR=$lr
                  CLIPPING_BOUND=$clipping_bound
                  CLIPPING_BOUND_RESIDUAL=$clipping_bound_residual
                  LOCAL_EPOCHS=$local_epoch
                  HISTORY_SIZE=$history_size
                  BASIS_SIZE=$basis_size
                  VIRTUAL_PUBLICS=$virtual_publics
                  PUBLIC_CLIENTS=$public_clients


                  GEP_ARGUMENTS=(
                  --num_public_clients "${PUBLIC_CLIENTS}"
                  --basis_size "${BASIS_SIZE}"
                  --history_size "${HISTORY_SIZE}"
                  --virtual_publics "${VIRTUAL_PUBLICS}"
                  )

                  echo  GEP_ARGUMENTS  "${GEP_ARGUMENTS[@]}"


                  EXP_NAME=glr_"${GLOBAL_LR}"_lr_"${LR}"_pub_"${PUBLIC_CLIENTS}"_virt_"${VIRTUAL_PUBLICS}"_basis_"${BASIS_SIZE}"_history_"${HISTORY_SIZE}"_mul_"${NOISE_MULTIPLIER_GRAD}"_"${NOISE_MULTIPLIER_RESIDUAL}"_clip_"${CLIPPING_BOUND}"_"${CLIPPING_BOUND_RESIDUAL}"_seed_"${SEED}"_ep_"${EPOCHS}"_loc_ep_"${LOCAL_EPOCHS}"
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
                         --global-lr "${GLOBAL_LR}"
                         --lr "${LR}"
                         --log-every "${LOG_EVERY}"
                         --exp-name "${EXP_NAME}"
                        )




                  echo  ${DATASET} FedAvg GEP RESIDUAL VIRTUAL SINGLE LOADER  with arguments "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"
                  CUDA_VISIBLE_DEVICES=$GPU python main_base_gep_gp.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"
                  #   CUDA_VISIBLE_DEVICES=$GPU python main_base_gep_virtual_public_single_loader_groups.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"  >> "${LOG_FOLDER}"/FedAvgGep_"${EXP_NAME}".txt
                done
              done
            done
          done
        done
      done
    done
  done
done


