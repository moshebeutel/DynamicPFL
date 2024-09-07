#!/bin/bash

N_CLIENTS=500
SAMPLE_RATE=0.1
EPOCHS=40
LOCAL_EPOCHS=4
LOG_FOLDER="logs/gep_all_4_epochs_eps_2"
DATASET="CIFAR10"
N_PUBLIC=10
N_BASIS=5
EPS=2
DELTA=0.002

# echo ${DATASET} main base
# python main_base.py --dataset $DATASET --num_clients $N_CLIENTS --user_sample_rate $SAMPLE_RATE --global_epoch $EPOCHS --local_epoch $LOCAL_EPOCHS --target_epsilon $EPS --target_delta $DELTA  >> ${LOG_FOLDER}/CIFAR10_base_clients_${N_CLIENTS}_rate_${SAMPLE_RATE}_epochs_${EPOCHS}.txt
# echo "CIFA10 base ended pushing results"
# git add logs/CIFAR10_base.txt
# CURRENT_TIME=$(date +"%Y-%m-%d %H:%M:%S")
# git commit -m "CIFAR10_base results $CURRENT_TIME"
# bash push.sh

# echo  ${DATASET} main base gep
# python main_base_gep.py --dataset $DATASET --num_clients $N_CLIENTS --user_sample_rate $SAMPLE_RATE --global_epoch $EPOCHS --local_epoch $LOCAL_EPOCHS --target_epsilon $EPS --target_delta $DELTA >> ${LOG_FOLDER}/CIFAR10_base_gep_clients_${N_CLIENTS}_rate_${SAMPLE_RATE}_epochs_${EPOCHS}_10_public_5_basis.txt
# echo "CIFA10 ours ended pushing results"
# git add logs/CIFAR10_ours_gep.txt
# CURRENT_TIME=$(date +"%Y-%m-%d %H:%M:%S")
# git commit -m "CIFAR10_ours_gep results $CURRENT_TIME clients_${N_CLIENTS}_rate_${SAMPLE_RATE}_epochs_${EPOCHS}"
# bash push.sh

echo "CIFAR10 ours"
python ours.py --dataset $DATASET --num_clients $N_CLIENTS --user_sample_rate $SAMPLE_RATE --global_epoch $EPOCHS --local_epoch $LOCAL_EPOCHS --target_epsilon $EPS --target_delta $DELTA >> ${LOG_FOLDER}/CIFAR10_ours_clients_${N_CLIENTS}_rate_${SAMPLE_RATE}_epochs_${EPOCHS}.txt
# echo "CIFA10 ours ended pushing results"
# git add logs/CIFAR10_ours.txt
# CURRENT_TIME=$(date +"%Y-%m-%d %H:%M:%S")
# git commit -m "CIFAR10_ours results $CURRENT_TIME clients_${N_CLIENTS}_rate_${SAMPLE_RATE}_epochs_${EPOCHS}"
# bash push.sh

echo "CIFAR10 ours gep"
python ours_gep.py --dataset $DATASET --num_clients $N_CLIENTS --user_sample_rate $SAMPLE_RATE --global_epoch $EPOCHS --local_epoch $LOCAL_EPOCHS --target_epsilon $EPS --target_delta $DELTA >> ${LOG_FOLDER}/CIFAR10_ours_gep_clients_${N_CLIENTS}_rate_${SAMPLE_RATE}_epochs_${EPOCHS}_10_public_5_basis.txt
# echo "CIFA10 ours ended pushing results"
# git add logs/CIFAR10_ours_gep.txt
# CURRENT_TIME=$(date +"%Y-%m-%d %H:%M:%S")
# git commit -m "CIFAR10_ours_gep results $CURRENT_TIME clients_${N_CLIENTS}_rate_${SAMPLE_RATE}_epochs_${EPOCHS}"
# bash push.sh


