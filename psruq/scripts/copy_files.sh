#!/bin/bash

# Define the new prefix
prefix="not_vectorized_"

# Find the files and copy them with a new name
for file in external_repos/pytorch_cifar*/checkpoints*/*/extracted_information_for_notebook.pkl; do
    # Get the directory and the base name of the file
    dir=$(dirname "$file")
    base=$(basename "$file" .pkl)
    
    # Define the new file name
    newfile="$dir/${prefix}${base}.pkl"
    
    # Copy the file to the new file name
    cp "$file" "$newfile"
done
