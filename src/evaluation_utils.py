import sys
from collections import defaultdict
sys.path.insert(0, './')
sys.path.insert(0, '.')
from tqdm.auto import tqdm
from external_repos.pytorch_cifar10.utils import (
    get_model,
)
import pickle
import json
import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import classification_report
from data_utils import load_dataloader_for_extraction
import os


def save_dict(save_path: str, dict_to_save: dict) -> None:
    """The function saves dict to a specific file

    Args:
        save_path (str): _description_
        dict_to_save (dict): _description_
    """
    with open(save_path, 'wb') as file:
        pickle.dump(dict_to_save, file)


def load_dict(load_path: str) -> dict:
    """The function loads dict from a specific file

    Args:
        load_path (str): _description_

    Returns:
        dict: _description_
    """
    with open(load_path, 'rb') as file:
        loaded_dict = pickle.load(file)
    return loaded_dict


def load_embeddings_dict(
    architecture: str,
    dataset_name: str,
    model_id: int,
    loss_function_name: str,
):
    """The function loads dict with extracted embeddings

    Args:
        architecture (str): _description_
        dataset_name (str): _description_
        model_id (int): _description_
        loss_function_name (str): _description_

    Returns:
        _type_: _description_
    """
    file_path = make_load_path(
        architecture=architecture,
        dataset_name=dataset_name,
        model_id=model_id,
        loss_function_name=loss_function_name
    )

    # Loading the dictionary from the file
    loaded_dict = load_dict(
        load_path=os.path.join(file_path, 'embeddings.pkl'))

    return loaded_dict


def get_additional_evaluation_metrics(embeddings_dict: dict) -> dict:
    embeddings = embeddings_dict['embeddings']
    y_true = embeddings_dict['labels']
    y_pred = np.argmax(embeddings, axis=-1)
    results_dict = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        output_dict=True
    )
    return results_dict


def save_additional_stats(
        architecture: str,
        dataset_name: str,
        loss_function_name: str,
        model_id: int,
):
    load_path = make_load_path(
        dataset_name=dataset_name,
        architecture=architecture,
        loss_function_name=loss_function_name,
        model_id=model_id
    )

    embeddings_dict = load_embeddings_dict(
        architecture=architecture,
        loss_function_name=loss_function_name,
        dataset_name=dataset_name,
        model_id=model_id
    )

    checkpoint_path = os.path.join(load_path, 'ckpt.pth')
    last_acc = torch.load(checkpoint_path, map_location='cpu')['acc']
    actual_acc = get_additional_evaluation_metrics(
        embeddings_dict=embeddings_dict
    )
    actual_acc.update({'last_acc': last_acc / 100})

    with open(os.path.join(load_path, 'results_dict.json'), 'w') as file:
        json.dump(fp=file, obj=actual_acc, indent=4,)


def load_model_checkpoint(architecture: str, path: str, device) -> nn.Module:
    """Load trained model checkpoint

    Args:
        architecture (str): _description_
        path (str): _description_
        device (_type_): _description_

    Returns:
        nn.Module: _description_
    """
    checkpoint = torch.load(path, map_location=device)
    model = get_model(architecture=architecture)
    model.load_state_dict(checkpoint['net'])
    return model


def make_load_path(
        architecture: str,
        loss_function_name: str,
        dataset_name: str,
        model_id: int,
):
    """Create load path for specific model

    Args:
        architecture (str): _description_
        loss_function_name (str): _description_
        dataset_name (str): _description_
        model_id (int): _description_

    Returns:
        _type_: _description_
    """
    if dataset_name == 'cifar10':
        return (f'./external_repos/'
                'pytorch_cifar10/'
                'checkpoints/'
                f'{architecture}/{loss_function_name}/{model_id}/')
    elif dataset_name == 'cifar100':
        return (f'./external_repos/'
                'pytorch_cifar100/'
                'checkpoints/'
                f'{architecture}/{loss_function_name}/{model_id}/')
    else:
        raise ValueError('No such dataset name supported.')


def extract_embeddings(
        architecture: str,
        loss_function_name: str,
        training_dataset_name: str,
        extraction_dataset_name: str,
        model_id: int,
):
    """The function extracts and save embeddings for a specific model

    Args:
        architecture (str): _description_
        loss_function_name (str): _description_
        training_dataset_name (str): _description_
        extraction_dataset_name (str): _description_
        model_id (int): _description_
    """
    load_path = make_load_path(
        architecture=architecture,
        dataset_name=training_dataset_name,
        loss_function_name=loss_function_name,
        model_id=model_id
    )
    checkpoint_path = os.path.join(load_path, 'ckpt.pth')
    embeddings_path = os.path.join(
        load_path,
        f'embeddings_{extraction_dataset_name}.pkl'
    )
    if os.path.exists(embeddings_path):
        print('Embeddings are already extracted! Skipping...')
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model_checkpoint(
        architecture=architecture,
        path=checkpoint_path,
        device=device)
    model = model.to(device)

    model.eval()

    loader = load_dataloader_for_extraction(
        training_dataset_name=training_dataset_name,
        extraction_dataset_name=extraction_dataset_name)

    output_embeddings = {}
    output_embeddings['embeddings'] = []
    output_embeddings['labels'] = []

    with torch.no_grad():
        for _, (inputs, targets) in tqdm(enumerate(loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            output_embeddings['embeddings'].append(outputs.cpu().numpy())
            output_embeddings['labels'].append(targets.cpu().numpy())
    output_embeddings['embeddings'] = np.vstack(
        output_embeddings['embeddings'])
    output_embeddings['labels'] = np.hstack(
        output_embeddings['labels'])

    # Saving the dictionary to a file using pickle
    save_dict(save_path=embeddings_path,
              dict_to_save=output_embeddings)


def collect_stats(
        dataset_name: str,
        architecture: str,
        loss_function_name,
        model_ids: list | np.ndarray,
) -> dict:
    stats_dict = defaultdict(list)
    for model_id in model_ids:
        load_path = make_load_path(
            architecture=architecture,
            loss_function_name=loss_function_name,
            dataset_name=dataset_name,
            model_id=model_id)
        with open(os.path.join(load_path, 'results_dict.json'), 'r') as file:
            current_dict_ = json.load(file)
            stats_dict['accuracy'].append(current_dict_['accuracy'])

            stats_dict['macro_avg_precision'].append(
                current_dict_['macro avg']['precision'])

            stats_dict['macro_avg_recall'].append(
                current_dict_['macro avg']['recall'])

            stats_dict['macro_avg_f1-score'].append(
                current_dict_['macro avg']['f1-score'])
    return stats_dict


if __name__ == '__main__':
    architecture = 'resnet18'
    # model_id = 0
    # loss_function_name = 'brier_score'
    # dataset_name = 'cifar10'
    model_ids = np.arange(20)

    for training_dataset_name in ['cifar10']:  # iterate over training datasets
        for extraction_dataset_name in ['cifar10', 'cifar100', 'svhn', 'lsun']:
            # iterate over datasets from which we want get embeddings
            for loss_function_name in [
                'brier_score',
                'cross_entropy',
                'spherical_score'
            ]:
                # different loss functions
                for model_id in model_ids:
                    # and different ensemble members
                    print((
                        f'Training dataset: {training_dataset_name} ...'
                        f'Extraction dataset: {extraction_dataset_name} '
                        f'Loading {architecture}, '
                        f'model_id={model_id} '
                        f'and loss {loss_function_name}'
                    ))
                    print('Extracting embeddings....')
                    extract_embeddings(
                        training_dataset_name=training_dataset_name,
                        extraction_dataset_name=extraction_dataset_name,
                        architecture=architecture,
                        model_id=model_id,
                        loss_function_name=loss_function_name
                    )
                    print('Finished embeddings extraction!')

                    if extraction_dataset_name == training_dataset_name:
                        print('Saving additional evaluation params...')
                        save_additional_stats(
                            dataset_name=training_dataset_name,
                            architecture=architecture,
                            model_id=model_id,
                            loss_function_name=loss_function_name
                        )

        # stats_dict = collect_stats(
        #     architecture=architecture,
        #     dataset_name=dataset_name,
        #     loss_function_name=loss_function_name,
        #     model_ids=model_ids,
        # )
        # print(stats_dict)

    print('Finished!')
