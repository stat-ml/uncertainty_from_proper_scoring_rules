import json
import os
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import classification_report
from source.source.data_utils import (
    load_dataloader_for_extraction,
    load_dict,
    load_embeddings_dict,
    load_model_checkpoint,
    make_load_path,
    save_dict,
)
from tqdm.auto import tqdm


def get_additional_evaluation_metrics(embeddings_dict: Dict) -> Dict | str:
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
    if isinstance(last_acc, torch.Tensor):
        last_acc = last_acc.cpu().detach().numpy()
    actual_acc = get_additional_evaluation_metrics(
        embeddings_dict=embeddings_dict
    )
    actual_acc.update({'last_acc': last_acc / 100})

    try:
        with open(os.path.join(load_path, 'results_dict.json'), 'w') as file:
            json.dump(fp=file, obj=actual_acc, indent=4,)
    except:
        import pdb
        pdb.set_trace()
        print('oh')


def extract_embeddings(
        architecture: str,
        loss_function_name: str,
        training_dataset_name: str,
        extraction_dataset_name: str,
        model_id: int,
        n_classes: int,
):
    """The function extracts and save embeddings for a specific model

    Args:
        architecture (str): _description_
        loss_function_name (str): _description_
        training_dataset_name (str): _description_
        extraction_dataset_name (str): _description_
        model_id (int): _description_
        n_classes (int): _description_
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
        device=device,
        n_classes=n_classes
    )
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


def collect_embeddings(
        model_ids: list | np.ndarray,
        architecture: str,
        loss_function_name: str,
        training_dataset_name: str,
        list_extraction_datasets: list = [
            'cifar10',
            'cifar100',
            'svhn',
            'blurred_cifar100',
            'blurred_cifar10'
        ],
        temperature: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """The function collects embeddings for different members of ensembles
      and different datasets

    Args:
        model_ids (list | np.ndarray): IDs of ensemble memebers we take
        into account
        architecture (str): model architecture name
        loss_function_name (str): loss function name
        training_dataset_name (str): dataset name used in training
        list_extraction_datasets (list, optional): datasets for which
        we will used embeddings. Defaults to ['cifar10', 'cifar100', 'svhn'].
        temperature: (float,): Temperature to scale logits. Default 1.0

    Returns:
        tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        Key -- dataset name; Value -- stacked logits.
        Key -- dataset name; Value -- stacked targets.
    """
    embeddings_per_dataset = defaultdict(list)
    targets_per_dataset = defaultdict(list)
    for extraction_dataset_name in list_extraction_datasets:
        for model_id in model_ids:

            path_to_model_folder = make_load_path(
                architecture=architecture,
                loss_function_name=loss_function_name,
                dataset_name=training_dataset_name,
                model_id=model_id
            )

            loaded_dict = load_dict(
                os.path.join(
                    path_to_model_folder,
                    f'embeddings_{extraction_dataset_name}.pkl'
                )
            )

            loaded_embeddings = loaded_dict['embeddings'] / temperature
            loaded_targets = loaded_dict['labels']

            embeddings_per_dataset[extraction_dataset_name].append(
                loaded_embeddings[None])
            targets_per_dataset[extraction_dataset_name].append(loaded_targets)

        embeddings_per_dataset[extraction_dataset_name] = np.vstack(
            embeddings_per_dataset[extraction_dataset_name])
        targets_per_dataset[extraction_dataset_name] = np.hstack(
            targets_per_dataset[extraction_dataset_name])

    return embeddings_per_dataset, targets_per_dataset


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
    architecture = 'resnet18'  # 'resnet18' 'vgg'
    training_datasets = [
        'missed_class_cifar10',
        # 'noisy_cifar10',
        # 'noisy_cifar100',
    ]  # ['cifar10', 'cifar100']
    model_ids = np.arange(6)

    # iterate over training datasets
    for training_dataset_name in training_datasets:
        if training_dataset_name in ['cifar100', 'noisy_cifar100']:
            n_classes = 100
        else:
            n_classes = 10

        for extraction_dataset_name in [
            'cifar10',
            'cifar100',
            'svhn',
            'blurred_cifar100',
            'blurred_cifar10',
        ]:
            # iterate over datasets from which we want get embeddings
            for loss_function_name in [
                'cross_entropy',
                'brier_score',
                'spherical_score',
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
                        n_classes=n_classes,
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
