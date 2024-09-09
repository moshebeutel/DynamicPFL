#!/bin/bash

N_CLIENTS=500
SAMPLE_RATE=0.1
EPOCHS=50
LOCAL_EPOCHS=4
DATASET="CIFAR10"
N_PUBLIC=10
N_BASIS=5
EPS=2
DELTA=0.002
PUBLIC_CLIENTS=10
BASIS_SIZE=5
HISTORY_SIZE=40
LOG_FOLDER="logs/${DATASET}_${N_PUBLIC}_public_${BASIS_SIZE}_basis_${HISTORY_SIZE}_history"

ARGUMENTS=(--dataset ${DATASET} --num_clients ${N_CLIENTS} --user_sample_rate ${SAMPLE_RATE} --global_epoch ${EPOCHS} --local_epoch ${LOCAL_EPOCHS} --target_epsilon ${EPS} --target_delta ${DELTA})
GEP_ARGUMENTS=(--num_public_clients ${PUBLIC_CLIENTS} --basis_size ${BASIS_SIZE} --history_size ${HISTORY_SIZE})

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


