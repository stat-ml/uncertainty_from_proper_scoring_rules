#!/bin/bash

# Define an array of loss function types
#declare -a loss_functions=("cross_entropy")

# Outer loop over loss function types
#for loss_function in "${loss_functions[@]}"
#do
loss_function="cross_entropy"
 # Inner loop from 10 to 20 for model_id
    for model_id in $(seq 1 3)
    do
        # Run the Python command with the current model_id and loss_function
        CUDA_VISIBLE_DEVICES=1 python main.py --model_id $model_id --architecture vgg --loss $loss_function
    done
#done
