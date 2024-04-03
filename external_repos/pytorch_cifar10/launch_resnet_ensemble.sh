#!/bin/bash

# Define an array of loss function types
loss_functions=("cross_entropy" "brier_score")

# Outer loop over loss function types
for loss_function in "${loss_functions[@]}"
do
    # Inner loop from 5 to 50 for model_id
    for model_id in $(seq 5 50)
    do
        # Run the Python command with the current model_id and loss_function
        python main.py --model_id $model_id --architecture resnet18 --loss $loss_function
    done
done
