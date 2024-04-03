#!/bin/bash

# Define an array of loss function types
#declare -a loss_functions=("cross_entropy")

# Outer loop over loss function types
#for loss_function in "${loss_functions[@]}"
#do
 loss_function="cross_entropy"
 # Inner loop from 6 to 50 for model_id
    for model_id in $(seq 6 50)
    do
        # Run the Python command with the current model_id and loss_function
        CUDA_VISIBLE_DEVICES=0 python main.py --model_id $model_id --architecture resnet18 --loss $loss_function
    done
#done
