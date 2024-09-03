Experiments with Laplace Redux – Effortless Bayesian Deep Learning

```bash
usage: main.py [-h] [-f FILE_PATH] [-l LOSS] [-d DATASET] [-v] [-m MODEL_NAME] [-o OUTPUT_PATH] [-c CUDA]

options:
  -h, --help            show this help message and exit
  -f FILE_PATH, --file_path FILE_PATH
                        Path to the model weights. It will look, using location where the main file is located as root. (default: None)
  -l LOSS, --loss LOSS  Loss function type. (default: CrossEntropy)
  -d DATASET, --dataset DATASET
                        Which dataset to use. (default: cifar10)
  -v, --verbose         Wether to show additional information or not. (default: False)
  -m MODEL_NAME, --model_name MODEL_NAME
                        Which model to use. (default: resnet18)
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Which model to use. (default: results/experiment.pth)
  -c CUDA, --cuda CUDA  Which cuda device to use. If set to -1 cpu will be used. Default value is -1. (default: -1)
```

### How to run code
First install all the needed dependencies with [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer). 

For the following example make sure that  you have saved model weights for model `resnet18` in `checkpoints/resnet18_ce.pth` and that you have gpu availible on your machine. Then to test last layer diagonal laplace you can run:
```bash
poetry run python main.py -m resnet18 -f checkpoints/resnet18_ce.pth -d cifar10_one_batch -c 0 -o results.pth -v
```
### Monte Carle Sample

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