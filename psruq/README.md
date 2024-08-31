# Predictive Uncertainty Quantification via Risk Decompositions for Strictly Proper Scoring Rules

This repository contains code for the paper "Predictive Uncertainty Quantification via Risk Decompositions for Strictly Proper Scoring Rules," submitted to NeurIPS 2024.

### Poetry
We use poetry to manage dependecies. You can find an installation guide and documentation [here](https://python-poetry.org/docs/#installing-with-the-official-installer).

**Install poetry**

For Linux/MacOS the following line will install poetry on your system (assuming you have python3):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**Install dependencies**

**All the commands should be executed from the folder with pyproject.toml file**

To install dependencies from pyproject.toml file you need to first create a poetry.lock file using
```bash
poetry lock
```
Then you can install necessary dependencies with
```bash
poetry install
```
It will create a .venv folder, that will contain a virtual environment and all the needed dependencies.

**Run python script**

Poetry installs dependencies in special virutal env, that is located in .venv/. To run any python code snippet with it you should run:
```bash
poetry run python ...
```
e.g.
```bash
cd experiments/laplace
poetry run python main.py -f checkpoints/resnet18_ce.pth -d cifar10_one_batch -v
```

### Repository Structure

The repository is organized as follows:

### Experiments
This folder contains different experiments related to the paper.
- cifar10: Contains training scripts for cifar10 dataset.
- laplace: Contains inference scripts for Laplace Redux – Effortless Bayesian Deep Learning model.

### Embedding Extraction
- source/source/evaluation_utils.py: Script to extract embeddings after the ensembles are trained.

### Notebooks
Folder notebooks/ have notebooks with data analysis of experiments.
- joint_tables.ipynb: Prepares comprehensive pandas tables for both problems.
- ood_analysis.ipynb: Generates plots for the OOD problem.
- ood_visualizations.ipynb: Creates visualizations for the OOD problem.
- mis_analysis.ipynb: Generates plots for the misclassification problem.
- mis_visualizations.ipynb: Creates visualizations for the misclassification problem.

### Notes on Trained Models
Due to a supplementary material size limit of 100MB, we cannot include the trained models in this repository. However, we are ready to provide them through any resource that maintains anonymity upon request.