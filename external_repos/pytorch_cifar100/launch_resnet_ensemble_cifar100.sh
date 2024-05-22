#!/bin/bash

declare -a loss_functions=("cross_entropy" "brier_score" "spherical_score")

# Iterate over each loss function
for loss_function in "${loss_functions[@]}"
do
    # Iterate over each model_id from 0 to 20
    for model_id in $(seq 0 20)
    do
        # Run the Python command with the current model_id and loss_function
        CUDA_VISIBLE_DEVICES=1 python train.py --model_id $model_id --architecture vgg --loss $loss_function -gpu --dataset noisy_cifar100
    done
done