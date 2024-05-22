# Predictive Uncertainty Quantification via Risk Decompositions for Strictly Proper Scoring Rules

This repository contains code for the paper "Predictive Uncertainty Quantification via Risk Decompositions for Strictly Proper Scoring Rules," submitted to NeurIPS 2024.

## Repository Structure

The repository is organized as follows:

### external_repos
In this folder there are the following subfolders with code for training:
- pytorch_cifar10: Contains training scripts for all models using various loss functions.
- pytorch_cifar100: Contains training scripts for all models using various loss functions.
### Ensemble Training
This script is located in each of the subfolders (pytorch_cifar10 and pytorch_cifar100)
- launch_ensemble.sh: A bash script to initiate the training procedure for the entire ensemble for a specific architecture. Modify the vgg or resnet18 parameter to train ensembles with different architectures.
### Embedding Extraction
- src/evaluation_utils.py: Script to extract embeddings after the ensembles are trained.
### Data Analysis Notebooks
1. joint_tables.ipynb: Prepares comprehensive pandas tables for both problems.
2. ood_analysis.ipynb: Generates plots for the OOD problem.
3. ood_visualizations.ipynb: Creates visualizations for the OOD problem.
4. mis_analysis.ipynb: Generates plots for the misclassification problem.
5. mis_visualizations.ipynb: Creates visualizations for the misclassification problem.

## Notes on Trained Models

Due to a supplementary material size limit of 100MB, we cannot include the trained models in this repository. However, we are ready to provide them through any resource that maintains anonymity upon request.