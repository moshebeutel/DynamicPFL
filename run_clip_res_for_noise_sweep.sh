#!/bin/bash

N_CLIENTS=500
SAMPLE_RATE=0.02       # Sample and aggregate 10 clients each iteration
#SAMPLE_RATE=0.03      # Sample and aggregate 15 clients each iteration
#SAMPLE_RATE=0.04      # Sample and aggregate 20 clients each iteration
#SAMPLE_RATE=0.06      # Sample and aggregate 30 clients each iteration
EPOCHS=400             # number of federated iterations
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
NOISE_MULTIPLIER_GRAD="$2"
NOISE_MULTIPLIER_RESIDUAL="$3"
CLIPPING_BOUND="$4"
CLIPPING_BOUND_RESIDUAL_MIN="$5"
CLIPPING_BOUND_RESIDUAL_MAX="$6"



# Print the arguments
echo ARGUMENTS FOR "$0"
echo "-------------------------------------------------------"
echo Arg1: GPU "$GPU"
echo Arg2: NOISE_MULTIPLIER_GRAD "${NOISE_MULTIPLIER_GRAD}"
echo Arg3: NOISE_MULTIPLIER_RESIDUAL "${NOISE_MULTIPLIER_RESIDUAL}"
echo Arg4: CLIPPING_BOUND "${CLIPPING_BOUND}"
echo Arg5: CLIPPING_BOUND_RESIDUAL_MIN "${CLIPPING_BOUND_RESIDUAL_MIN}"
echo Arg6: CLIPPING_BOUND_RESIDUAL_MAX "${CLIPPING_BOUND_RESIDUAL_MAX}"
echo "-------------------------------------------------------"

SEED=43
LOCAL_EPOCHS=5         # Each client iterate  times over all his local data


EVAL_AFTER=50
EVAL_EVERY=10
LOG_EVERY=5
BASIS_SIZE=400
PUBLIC_CLIENTS=50                    # clients used for subspace compute
VIRTUAL_PUBLICS=400
HISTORY_SIZE=1000
GEP_ARGUMENTS=(
          --num_public_clients "${PUBLIC_CLIENTS}"
          --basis_size "${BASIS_SIZE}"
          --history_size "${HISTORY_SIZE}"
          --virtual_publics "${VIRTUAL_PUBLICS}"
          )

INTERVAL="$(echo "${CLIPPING_BOUND_RESIDUAL_MAX} - ${CLIPPING_BOUND_RESIDUAL_MIN}" | bc)"
echo SWEEP INTERVAL "${INTERVAL}"
STEP="$(echo "scale=4; $INTERVAL / 3" | bc)"

for res_clip in $(seq "${CLIPPING_BOUND_RESIDUAL_MIN}" "${STEP}" "${CLIPPING_BOUND_RESIDUAL_MAX}"); do

  CLIPPING_BOUND_RESIDUAL=$res_clip

  echo " >>>>>    Noising Parameters   ****************"
  echo NOISE_MULTIPLIER_GRAD "${NOISE_MULTIPLIER_GRAD}"
  echo NOISE_MULTIPLIER_RESIDUAL "${NOISE_MULTIPLIER_RESIDUAL}"
  echo CLIPPING_BOUND "${CLIPPING_BOUND}"
  echo CLIPPING_BOUND_RESIDUAL "${CLIPPING_BOUND_RESIDUAL}"
  echo " <<<<<    Noising Parameters   ****************"

  EXP_NAME=virtual_"${VIRTUAL_PUBLICS}"_basis_"${BASIS_SIZE}"_history_"${HISTORY_SIZE}"_noise_grad"${NOISE_MULTIPLIER_GRAD}"_noise_res_"${NOISE_MULTIPLIER_RESIDUAL}"_clip_"${CLIPPING_BOUND}"_clip_res_"${CLIPPING_BOUND_RESIDUAL}"_seed_"${SEED}"_epochs_"${EPOCHS}"_local_epochs"${LOCAL_EPOCHS}"
  EXP_NAME=GEP_RES_"${EXP_NAME}"

  echo Start Experiment: "${EXP_NAME}"

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
       --min-acc-save "20.0"
       --log-every "${LOG_EVERY}"
       --exp-name "${EXP_NAME}"
      )




  echo  ${DATASET} FedAvg GEP RESIDUAL VIRTUAL SINGLE LOADER  with arguments "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"
  CUDA_VISIBLE_DEVICES=$GPU python main_base_gep_virtual_public_single_loader_groups.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"
#      CUDA_VISIBLE_DEVICES=$GPU python main_base_gep_virtual_public_single_loader_groups.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"  >> "${LOG_FOLDER}"/FedAvgGep_"${EXP_NAME}".txt
#    done
#  done
done



