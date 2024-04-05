#!/bin/bash

# Define an array of loss function types
# declare -a loss_functions=("brier_score")

# Outer loop over loss function types
# for loss_function in "${loss_functions[@]}"
# do
    # Inner loop from 0 to 20 for model_id
loss_function="brier_score"
    for model_id in $(seq 2 20)
    do
        # Run the Python command with the current model_id and loss_function
        CUDA_VISIBLE_DEVICES=1 python main.py --model_id $model_id --architecture resnet18 --loss $loss_function
    done
# done
