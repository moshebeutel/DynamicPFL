#!/bin/bash

N_CLIENTS=512
SAMPLE_RATE=0.0312
EPOCHS=5
LOCAL_EPOCHS=2
DATASET="CIFAR10"
N_PUBLIC=10
N_BASIS=5
EPS=2
DELTA=0.001953125
PUBLIC_CLIENTS=10
BASIS_SIZE=5
HISTORY_SIZE=10
LOG_FOLDER="logs/${DATASET}_${N_PUBLIC}_public_${BASIS_SIZE}_basis_${HISTORY_SIZE}_history"

# Check if the folder exists
if [ ! -d "$LOG_FOLDER" ]; then
    # If the folder does not exist, create it
    mkdir -p "$LOG_FOLDER"
    echo "Folder created at $LOG_FOLDER"
else
    # If the folder exists, output a message
    echo "Folder already exists at $LOG_FOLDER"
fi
ARGUMENTS=(--dataset "${DATASET}" --num_clients "${N_CLIENTS}" --user_sample_rate "${SAMPLE_RATE}" --global_epoch "${EPOCHS}" --local_epoch "${LOCAL_EPOCHS}" --target_epsilon "${EPS}" --target_delta "${DELTA}")
GEP_ARGUMENTS=(--num_public_clients "${PUBLIC_CLIENTS}" --basis_size "${BASIS_SIZE}" --history_size "${HISTORY_SIZE}")

echo ${DATASET} FedAvg SGD_DP
python main_base.py "${ARGUMENTS[@]}"  >> ${LOG_FOLDER}/FedAvgSgdDP.txt
# echo "CIFA10 base ended pushing results"
# git add logs/CIFAR10_base.txt
# CURRENT_TIME=$(date +"%Y-%m-%d %H:%M:%S")
# git commit -m "CIFAR10_base results $CURRENT_TIME"
# bash push.sh

echo  ${DATASET} FedAvg GEP
python main_base_gep.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"  >> ${LOG_FOLDER}/FedAvgGep.txt
# echo "CIFA10 ours ended pushing results"
# git add logs/CIFAR10_ours_gep.txt
# CURRENT_TIME=$(date +"%Y-%m-%d %H:%M:%S")
# git commit -m "CIFAR10_ours_gep results $CURRENT_TIME clients_${N_CLIENTS}_rate_${SAMPLE_RATE}_epochs_${EPOCHS}"
# bash push.sh


