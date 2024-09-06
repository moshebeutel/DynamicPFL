#!/bin/bash

echo "CIFAR10 main base"
python main_base.py --dataset CIFAR10 --num_clients 500 --user_sample_rate 0.02 --global_epoch 300 >> logs/CIFAR10_base.txt
echo "CIFA10 base ended pushing results"
git add logs/CIFAR10_base.txt
CURRENT_TIME=$(date +"%Y-%m-%d %H:%M:%S")
git commit -m "CIFAR10_base results $CURRENT_TIME"
bash push.sh

echo "CIFAR10 ours"
python ours.py --dataset CIFAR10 --num_clients 500 --user_sample_rate 0.02 --global_epoch 300 >> logs/CIFAR10_ours.txt
echo "CIFA10 ours ended pushing results"
git add logs/CIFAR10_ours.txt
CURRENT_TIME=$(date +"%Y-%m-%d %H:%M:%S")
git commit -m "CIFAR10_ours results $CURRENT_TIME"
bash push.sh

