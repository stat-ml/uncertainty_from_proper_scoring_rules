#!/bin/bash

# Loop from 4 to 50
for model_id in $(seq 4 50)
do
    # Run the Python command with the current model_id
    python main.py --model_id $model_id --architecture resnet18 --loss cross_entropy
done

