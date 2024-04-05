import sys
sys.path.insert(0, './')
sys.path.insert(0, '.')
import numpy as np
import torch
import torch.nn as nn
import pickle
from external_repos.pytorch_cifar10.utils import get_dataloaders, get_model
from tqdm.auto import tqdm


def load_model_checkpoint(architecture: str, path: str, device) -> nn.Module:
    checkpoint = torch.load(path, map_location=device)
    model = get_model(architecture=architecture)
    model.load_state_dict(checkpoint['net'])
    return model


def make_load_path(
        architecture: str,
        loss_function: str,
        model_id: int,
):
    return (f'./external_repos/'
            'pytorch_cifar10/'
            'checkpoints/'
            f'{architecture}/{loss_function}/{model_id}/')


def extract_embeddings(
        architecture: str,
        loss_function: str,
        model_id: int,
):
    load_path = make_load_path(
        architecture=architecture,
        loss_function=loss_function,
        model_id=model_id
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_checkpoint(
        architecture=architecture,
        path=load_path + 'ckpt.pth',
        device=device)
    model = model.to(device)
    _, testloader = get_dataloaders()
    model.eval()

    output_embeddings = {}
    output_embeddings['embeddings'] = []
    output_embeddings['labels'] = []

    with torch.no_grad():
        for _, (inputs, targets) in tqdm(enumerate(testloader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            output_embeddings['embeddings'].append(outputs.cpu().numpy())
            output_embeddings['labels'].append(targets.cpu().numpy())
    output_embeddings['embeddings'] = np.vstack(
        output_embeddings['embeddings'])
    output_embeddings['labels'] = np.hstack(
        output_embeddings['labels'])

    # Saving the dictionary to a file using pickle
    with open((load_path + 'embeddings.pkl'), 'wb') as file:
        pickle.dump(output_embeddings, file)


if __name__ == '__main__':
    extract_embeddings(
        architecture='resnet18',
        model_id=0,
        loss_function='brier_score'
    )
