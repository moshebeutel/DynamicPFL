#!/bin/bash

N_CLIENTS=44
SAMPLE_RATE=0.18182
EPOCHS=50
LOCAL_EPOCHS=2
DATASET="putEMG"
EPS=2
DELTA=0.022727272727272728  # 1/44
PUBLIC_CLIENTS=8


LOG_FOLDER="logs/${DATASET}_${N_PUBLIC}_public_${EPOCHS}_epochs_${EPS}_eps_${SAMPLE_RATE}_sample_rate_sweep_basis_size"
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



echo ${DATASET} FedAvg SGD_DP
python main_base.py "${ARGUMENTS[@]}"  >> ${LOG_FOLDER}/FedAvgSgdDP.txt

echo "$DATASET" DynamicPFL
python ours.py "${ARGUMENTS[@]}"  >> ${LOG_FOLDER}/DynamicPFL.txt

BASIS_SIZE=5
HISTORY_SIZE=16
GEP_ARGUMENTS=(--num_public_clients "${PUBLIC_CLIENTS}" --basis_size "${BASIS_SIZE}" --history_size "${HISTORY_SIZE}")

echo  ${DATASET} FedAvg GEP history size ${HISTORY_SIZE}
python main_base_gep.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"  >> ${LOG_FOLDER}/FedAvgGep_history_${HISTORY_SIZE}.txt


echo "$DATASET" DynamicPFL GEP history size ${HISTORY_SIZE}
python ours.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"  >> ${LOG_FOLDER}/DynamicPFL_GEP_history_${HISTORY_SIZE}.txt

BASIS_SIZE=10
HISTORY_SIZE=16
GEP_ARGUMENTS=(--num_public_clients "${PUBLIC_CLIENTS}" --basis_size "${BASIS_SIZE}" --history_size "${HISTORY_SIZE}")

echo  ${DATASET} FedAvg GEP basis size ${BASIS_SIZE}
python main_base_gep.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"  >> ${LOG_FOLDER}/FedAvgGep_basis_${BASIS_SIZE}.txt


echo "$DATASET" DynamicPFL GEP basis size ${BASIS_SIZE}
python ours.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"  >> ${LOG_FOLDER}/DynamicPFL_GEP_basis_${BASIS_SIZE}.txt

BASIS_SIZE=15
HISTORY_SIZE=16
GEP_ARGUMENTS=(--num_public_clients "${PUBLIC_CLIENTS}" --basis_size "${BASIS_SIZE}" --history_size "${HISTORY_SIZE}")

echo  ${DATASET} FedAvg GEP basis size ${BASIS_SIZE}
python main_base_gep.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"  >> ${LOG_FOLDER}/FedAvgGep_basis_${BASIS_SIZE}.txt


echo "$DATASET" DynamicPFL GEP basis size ${BASIS_SIZE}
python ours.py "${ARGUMENTS[@]}" "${GEP_ARGUMENTS[@]}"  >> ${LOG_FOLDER}/DynamicPFL_GEP_basis_${HISTORY_SIZE}.txt

