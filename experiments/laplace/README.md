Experiments with Laplace Redux – Effortless Bayesian Deep Learning

## Check model accuracy

```
usage: main.py [-h] [-f FILE_PATH] [-l LOSS] [-d DATASET] [-v] [-m MODEL_NAME] [-o OUTPUT_PATH] [-c CUDA]

options:
  -h, --help            show this help message and exit
  -f FILE_PATH, --file_path FILE_PATH
                        Path to the model weights. It will look, using location where the main file is located as root. (default: None)
  -l LOSS, --loss LOSS  Loss function type. Available options are: ['CrossEntropy', 'BrierScore', 'SphericalScore', 'NegLog'] (default: CrossEntropy)
  -d DATASET, --dataset DATASET
                        Which dataset to use. Available options are: ['cifar10_one_batch', 'cifar10', 'cifar100', 'cifar10_missed_label', 'cifar10_noisy_label', 'svhn'] (default: cifar10)
  -v, --verbose         Wether to show additional information or not. (default: False)
  -m MODEL_NAME, --model_name MODEL_NAME
                        Which model to use. Available options are: ['resnet18', 'vgg11', 'vgg13', 'vgg16', 'vgg19'] (default: resnet18)
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path to save results of the experiment. (default: results/experiment.pth)
  -c CUDA, --cuda CUDA  Which cuda device to use. If set to -1 cpu will be used. Default value is -1. (default: -1)
```

### How to run code
First install all the needed dependencies with [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer). 

For the following example make sure that  you have saved model weights for model `resnet18` in `checkpoints/resnet18_ce.pth` and that you have gpu availible on your machine. Then to test last layer diagonal laplace you can run:
```bash
uv run python main.py -m resnet18 -f checkpoints/resnet18_ce.pth -d cifar10_one_batch -c 0 -o results.pth -v
```
## Monte Carle Sample

`KronLLLaplace` have a special function `_nn_predictive_samples`. It can be used to sample predictions from different models, sampled from normal prior distribution. 

Minimal working example:
```python
from laplace import KronLLLaplace
from source.datasets import get_dataloaders
from source.models import get_model

model = get_model(model_name='resnet18')
trainloader, testloader = get_dataloaders(dataset='cifar10_onebatch')

laplace_model = KronLLLaplace(
    model=model,
    likelihood="classification"
)

laplace_model.fit(train_loader=trainloader)

for X, _ in testloader:
    # Samples are at the last dimension.
    print(laplace_model._nn_predictive_samples(
        X=X,
        n_samples=15,
    ).shape)
# It should give [128, 10, 15] - [batch_size, number_of_classes, number_of_monte_carlo_samples]
```

We have also wrote a script to run logits samples, that will create a folder on path `logits_results/checkpoints_{in_distribution_dataset_name}/{model_name}/{arguments.loss}/` with logits and model weights.

```
usage: sample_logits.py [-h] [-f FILE_PATH] [-o OUT_OF_DISTRIBUTION_DATASET] [-d IN_DISTRIBUTION_DATASET] [-l LOSS] [-m MODEL_NAME] [-u NUMBER_OF_CLASSES] [-v] [-c CUDA] [-n NUMBER_OF_WEIGHT_SAMPLES]

options:
  -h, --help            show this help message and exit
  -f FILE_PATH, --file_path FILE_PATH
                        Path to the model weights. The script will look for the file, using location where the main file is located as root. e.g. chekpoint/model.pth will look for the file at PATH_TO_MAINPY/checkpoint/model.pth (default: None)
  -o OUT_OF_DISTRIBUTION_DATASET, --out_of_distribution_dataset OUT_OF_DISTRIBUTION_DATASET
                        Which type of OOD data to use to evaluate logits. Available options are: ['cifar10_one_batch', 'cifar10', 'cifar100', 'cifar10_missed_label', 'cifar10_noisy_label', 'svhn'] (default: None)
  -d IN_DISTRIBUTION_DATASET, --in_distribution_dataset IN_DISTRIBUTION_DATASET
                        Which type of dataset to use to evaluate laplace approximation. Available options are: ['cifar10_one_batch', 'cifar10', 'cifar100', 'cifar10_missed_label', 'cifar10_noisy_label', 'svhn'] (default: cifar10)
  -l LOSS, --loss LOSS  Loss function type. Available options are: ['CrossEntropy', 'BrierScore', 'SphericalScore', 'NegLog'] (default: CrossEntropy)
  -m MODEL_NAME, --model_name MODEL_NAME
                        Which model to use. Available options are: ['resnet18', 'vgg11', 'vgg13', 'vgg16', 'vgg19'] (default: resnet18)
  -u NUMBER_OF_CLASSES, --number_of_classes NUMBER_OF_CLASSES
                        Number of classes to use for prediction. (default: 10)
  -v, --verbose         Wether to show additional information or not. (default: False)
  -c CUDA, --cuda CUDA  Which cuda device to use. If set to -1 cpu will be used. Default value is -1. (default: -1)
  -n NUMBER_OF_WEIGHT_SAMPLES, --number_of_weight_samples NUMBER_OF_WEIGHT_SAMPLES
                        This parameter sets the amount of times the weights are going to be sample from the model distribution. (default: 20)
```

### How to run code

First install all the needed dependencies with [uv](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer). 

For the following example make sure that  you have saved model weights for model `resnet18` in `checkpoints/resnet18_ce.pth` and that you have gpu availible on your machine. Then to test last layer diagonal laplace you can run:

```bash
uv run python sample_logits.py -f checkpoints/resnet18_ce.pth -o cifar10_one_batch -d cifar10_one_batch -l CrossEntropy -m resnet18 -v
```