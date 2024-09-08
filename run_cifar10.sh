#!/bin/bash

N_CLIENTS=500
SAMPLE_RATE=0.1
EPOCHS=40
LOCAL_EPOCHS=4
DATASET="CIFAR10"
N_PUBLIC=10
N_BASIS=5
EPS=2
DELTA=0.002
PUBLIC_CLIENTS=10
BASIS_SIZE=5
HISTORY_SIZE=30
LOG_FOLDER="logs/${DATASET}_${N_PUBLIC}_public_${BASIS_SIZE}_basis_${HISTORY_SIZE}_history"
#echo Create Log Folder $LOG_FOLDER
#mkdir "$LOG_FOLDER"
ARGUMENTS=(--dataset ${DATASET} --num_clients ${N_CLIENTS} --user_sample_rate ${SAMPLE_RATE} --global_epoch ${EPOCHS} --local_epoch ${LOCAL_EPOCHS} --target_epsilon ${EPS} --target_delta ${DELTA})
GEP_ARGUMENTS=(--num_public_clients ${PUBLIC_CLIENTS} --basis_size ${BASIS_SIZE} --history_size ${HISTORY_SIZE})


echo "$DATASET" DynamicPFL
python ours.py "${ARGUMENTS[@]}"  >> ${LOG_FOLDER}/DynamicPFL.txt
# echo "CIFA10 ours ended pushing results"
# git add logs/CIFAR10_ours.txt
# CURRENT_TIME=$(date +"%Y-%m-%d %H:%M:%S")
# git commit -m "CIFAR10_ours results $CURRENT_TIME clients_${N_CLIENTS}_rate_${SAMPLE_RATE}_epochs_${EPOCHS}"
# bash push.sh

echo "$DATASET" DynamicPFL GEP
python ours.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"  >> ${LOG_FOLDER}/DynamicPFL_GEP.txt
# echo "CIFA10 ours ended pushing results"
# git add logs/CIFAR10_ours_gep.txt
# CURRENT_TIME=$(date +"%Y-%m-%d %H:%M:%S")
# git commit -m "CIFAR10_ours_gep results $CURRENT_TIME clients_${N_CLIENTS}_rate_${SAMPLE_RATE}_epochs_${EPOCHS}"
# bash push.sh


