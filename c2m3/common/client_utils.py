# Copyright 2025 Lorenzo Sani & Alexandru-Andrei Iacob
# SPDX-License-Identifier: Apache-2.0

"""Client utilities for the FEMNIST dataset."""

from datetime import datetime, timezone
import json
import logging
import numbers
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, cast
from collections.abc import Callable, Sized

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from flwr.common.typing import NDArrays, Scalar
from flwr.server import History
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from c2m3.models.tiny_resnet import BasicBlock, TinyResNet

from .femnist_dataset import FEMNIST
from flwr.common.logger import log
from c2m3.models.cnn import Net
from c2m3.models.mlp import MLP


class DPAdaptiveClipFlagError(BaseException):
    """Exception in case adaptive clip flag is not boolean-valued."""


class DPClientNormBitError(BaseException):
    """Exception for the case when the client norm bit is not in metrics."""


class IntentionalDropoutError(BaseException):
    """For clients to intentionally drop out of the federated learning process."""


class ModelSizeNotFoundError(BaseException):
    """Exception raised when model size is not found in config."""


def convert(o: Any) -> int | float:
    """Convert numpy types to Python types."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    raise TypeError


def save_history(home_dir: Path, hist: History, name: str) -> None:
    """Save history from simulation to file."""
    time = int(datetime.now(timezone.utc).timestamp())
    path = home_dir / "histories"
    path.mkdir(exist_ok=True)
    path = path / f"hist_{time}_{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(hist.__dict__, f, ensure_ascii=False, indent=4, default=convert)


def get_device() -> str:
    """
    Get the device (CPU, CUDA, or MPS) available for computation.

    Returns
    -------
        str: The device available for computation.
    """
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    return device


# Load with appropriate transforms
def to_tensor_transform(p: Any) -> torch.Tensor:
    """Transform the object given to a PyTorch Tensor.

    Parameters
    ----------
        p (Any): object to transform

    Returns
    -------
        torch.Tensor: resulting PyTorch Tensor
    """
    return torch.tensor(p)


def load_femnist_dataset(data_dir: Path, mapping: Path, name: str) -> FEMNIST:
    """Load the FEMNIST dataset given the mapping .csv file.

    The relevant transforms are automatically applied.

    Parameters
    ----------
        data_dir (Path): path to the dataset folder.
        mapping (Path): path to the mapping .csv file chosen.
        name (str): name of the dataset to load, train or test.

    Returns
    -------
        Dataset: FEMNIST dataset object, ready-to-use.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return FEMNIST(
        mapping=mapping,
        name=name,
        data_dir=data_dir,
        transform=transform,
        target_transform=to_tensor_transform,
    )


class CIFAR10FromSamples(Dataset):
    """CIFAR10 dataset from samples.
    
    Parameters
    ----------
        samples (list): List of (image_path, label) tuples
        data_dir (Path): Base directory for image paths
        transform (callable, optional): Transform to apply to images
    """
    def __init__(self, samples, data_dir, transform=None):
        self.samples = samples
        self.data_dir = data_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Load image as PIL
        from PIL import Image
        img = Image.open(self.data_dir / img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def load_cifar10_dataset(data_dir: Path, mapping: Path, name: str) -> CIFAR10FromSamples:
    """Load the CIFAR10 dataset given the mapping file.

    The relevant transforms are automatically applied.

    Parameters
    ----------
        data_dir (Path): path to the dataset folder.
        mapping (Path): path to the mapping file chosen.
        name (str): name of the dataset to load, train or test.

    Returns
    -------
        Dataset: CIFAR10FromSamples dataset object, ready-to-use.
    """
    # For CIFAR10, use standard normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load samples from the mapping file
    with open(mapping, "r") as f:
        mapping_data = json.load(f)
    
    if name not in mapping_data:
        raise ValueError(f"'{name}' data not found in mapping file")
    
    # Get samples for this name (train or test)
    samples = mapping_data[name]
    
    return CIFAR10FromSamples(
        samples=samples,
        data_dir=data_dir,
        transform=transform
    )


def train_femnist(
    net: Module,
    train_loader: DataLoader,
    epochs: int,
    device: str,
    optimizer: torch.optim.Optimizer,
    criterion: Module,
    max_batches: int | None = None,
    **kwargs: dict[str, Any],
) -> float:
    """Trains the network on the training set.

    Parameters
    ----------
        net (Module): generic module object describing the network to train.
        train_loader (DataLoader): dataloader to iterate during the training.
        epochs (int): number of epochs of training.
        device (str): device name onto which perform the computation.
        optimizer (torch.optim.Optimizer): optimizer object.
        criterion (Module): generic module describing the loss function.

    Returns
    -------
        float: the final epoch mean train loss.
    """
    net.train()
    running_loss, total = 0.0, 0
    for epoch in range(epochs):
        running_loss = 0.0
        total = 0
        # Calculate total batches
        total_batches = len(train_loader) if max_batches is None else min(max_batches, len(train_loader))
        desc = f"Training [Epoch {epoch+1}/{epochs}]"
        
        for i, (data, labels) in enumerate(tqdm(train_loader, desc=desc, total=total_batches, leave=True)):
            if max_batches is not None and i >= max_batches:
                break
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(data), labels)
            running_loss += loss.item()
            total += labels.size(0)
            loss.backward()
            optimizer.step()
    
    # Check for division by zero
    if total == 0:
        print("Warning: No data was processed during training. Dataset might be empty.")
        return 0.0  # Return 0 loss instead of causing division by zero
    
    return running_loss / total


def test_femnist(
    net: Module,
    test_loader: DataLoader,
    device: str,
    criterion: Module,
    max_batches: int | None = None,
    **kwargs: dict[str, Any],
) -> tuple[float, float]:
    """Validate the network on a test set.

    Parameters
    ----------
        net (Module): generic module object describing the network to test.
        test_loader (DataLoader): dataloader to iterate during the testing.
        device (str):  device name onto which perform the computation.
        criterion (Module): generic module describing the loss function.

    Returns
    -------
        tuple[float, float]:
            couple of average test loss and average accuracy on the test set.
    """
    batch_cnt = 0
    correct, total, loss = 0, 0, 0.0
    net.eval()
    
    # Calculate total batches
    total_batches = len(test_loader) if max_batches is None else min(max_batches, len(test_loader))
    desc = f"Testing [{len(test_loader.dataset)} samples]"

    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc=desc, total=total_batches, leave=True):

            if max_batches is not None and batch_cnt >= max_batches:
                break
            batch_cnt += 1

            data, labels = data.to(device), labels.to(device)
            outputs = net(data)

            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Check for division by zero (empty dataset)
    if total == 0:
        print("Warning: No data was processed during testing. Test dataset might be empty.")
        return 0.0, 0.0  # Return zeros for both loss and accuracy
        
    accuracy = correct / total
    return loss, accuracy


def get_activations_from_random_input(
    net: Module,
    device: str,
    n_samples: int = 100,
    seed: int = 1337,
) -> np.ndarray:
    """Return the activations of the network on random input."""
    # Get a random input
    prng = torch.random.manual_seed(seed)
    random_input = torch.rand((n_samples, 1, 28, 28), generator=prng)
    random_input = random_input.to(device)
    # Get the activations
    net.to(device)
    net.eval()
    with torch.no_grad():
        outputs: torch.Tensor = torch.softmax(net(random_input), dim=1)
    average_activations = torch.mean(outputs, dim=0)
    return average_activations.cpu().numpy()


def set_model_parameters(net: Module, parameters: NDArrays) -> Module:
    """Put a set of parameters into the model object.

    Parameters
    ----------
        net (Module): model object.
        parameters (NDArrays): set of parameters to put into the model.

    Returns
    -------
        Module: updated model object.
    """
    weights = parameters
    params_dict = zip(net.state_dict().keys(), weights, strict=False)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net


def get_model_parameters(net: Module) -> NDArrays:
    """Get the current model parameters as NDArrays.

    Parameters
    ----------
        net (Module): current model object.

    Returns
    -------
        NDArrays: set of parameters from the current model.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

# All experiments will have the exact same initialization.
# All differences in performance will come from training
def get_network_generator_cnn() -> Callable[[], Net]:
    """Get function to generate a new CNN model."""
    untrained_net: Net = Net()

    def generated_net() -> Net:
        return deepcopy(untrained_net)

    return generated_net

# Implementation with different initialization
def get_network_generator_cnn_random() -> Callable[[], Net]:
    """Get function to generate a new CNN model."""
    def generated_net() -> Net:
        return Net()

    return generated_net

def get_network_generator_tiny_resnet(num_classes=10):
    """
    Create a generator function that returns a TinyResNet model configured for CIFAR10.
    
    Args:
        input_channels (int): Number of input channels (3 for RGB images)
        num_classes (int): Number of output classes (10 for CIFAR10)
        
    Returns:
        Callable that returns a TinyResNet model
    """
    untrained_net: TinyResNet = TinyResNet(
            block=BasicBlock, 
            num_classes=num_classes, 
            norm_layer=nn.BatchNorm2d
        )

    def network_generator() -> TinyResNet:
        return deepcopy(untrained_net)
    
    return network_generator

def get_network_generator_tiny_resnet_random() -> Callable[[], TinyResNet]:
    """Get function to generate a new TinyResNet model with random initialization."""
    def network_generator() -> TinyResNet:
        return TinyResNet(
            block=BasicBlock,
            num_classes=10,
            norm_layer=nn.BatchNorm2d
        )

    return network_generator


# All experiments will have the exact same initialization.
# All differences in performance will come from training
def get_network_generator_mlp() -> Callable[[], MLP]:
    """Get function to generate a new MLP model."""
    untrained_net: MLP = MLP()

    def generated_net() -> MLP:
        return deepcopy(untrained_net)

    return generated_net


def set_model_parameters(net: Module, parameters: NDArrays) -> Module:
    """Get function to put a set of parameters into the model object.

    Parameters
    ----------
        net (Module): model object.
        parameters (NDArrays): set of parameters to put into the model.

    Returns
    -------
        Module: updated model object.
    """
    weights = parameters
    params_dict = zip(net.state_dict().keys(), weights, strict=False)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    return net


def get_model_parameters(net: Module) -> NDArrays:
    """Get function to get the current model parameters as NDArrays.

    Parameters
    ----------
        net (Module): current model object.

    Returns
    -------
        NDArrays: set of parameters from the current model.
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def aggregate_weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Combine results from multiple clients.

    Parameters
    ----------
        metrics (list[tuple[int, dict]]): collected clients metrics

    Returns
    -------
        dict: result dictionary containing the aggregate of the metrics passed.
    """
    average_dict: dict = defaultdict(list)
    total_examples: int = 0
    for num_examples, metrics_dict in metrics:
        for key, val in metrics_dict.items():
            if isinstance(val, numbers.Number):
                average_dict[key].append((num_examples, val))
        total_examples += num_examples
    return {
        key: {
            "avg": float(
                sum([num_examples * m for num_examples, m in val])
                / float(total_examples)
            ),
            "all": val,
        }
        for key, val in average_dict.items()
    }


def get_federated_evaluation_function(
    data_dir: Path,
    centralized_mapping: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    model_generator: Callable[[], Module],
    criterion: Module,
) -> Callable[[int, NDArrays, dict[str, Any]], tuple[float, dict[str, Scalar]]]:
    """Wrap function for the external federated evaluation function.

    It provides the external federated evaluation function with some
    parameters for the dataloader, the model generator function, and
    the criterion used in the evaluation.

    Parameters
    ----------
        data_dir (Path): path to the dataset folder.
        centralized_mapping (Path): path to the mapping .csv file chosen.
        device (str):  device name onto which perform the computation.
        batch_size (int): batch size of the test set to use.
        num_workers (int): correspond to `num_workers` param in the Dataloader object.
        model_generator (Callable[[], Module]):  model generator function.
        criterion (Module): PyTorch Module containing the criterion.

    Returns
    -------
        Callable[[int, NDArrays, dict[str, Any]], tuple[float, dict[str, Scalar]]]:
            external federated evaluation function.
    """
    full_file: Path = centralized_mapping
    dataset: Dataset = load_femnist_dataset(data_dir, full_file, "val")
    num_samples = len(cast(Sized, dataset))
    index_list = list(range(num_samples))
    prng = np.random.RandomState(1337)
    prng.shuffle(index_list)
    index_list = index_list[:1500]
    dataset = torch.utils.data.Subset(dataset, index_list)  # type: ignore[reportAttributeAccessIssue]

    log(
        logging.INFO,
        "Reduced federated test_set size from %s to a size of %s mean index: %s",
        num_samples,
        len(cast(Sized, dataset)),
        np.mean(index_list),
    )

    def federated_evaluation_function(
        server_round: int,
        parameters: NDArrays,
        fed_eval_config: dict[
            str, Any
        ],  # mandatory argument, even if it's not being used
    ) -> tuple[float, dict[str, Scalar]]:
        """Evaluate on a centralized test set.

        It uses the centralized val set for sake of simplicity.

        Parameters
        ----------
            server_round (int): current federated round.
            parameters (NDArrays): current model parameters.
            fed_eval_config (dict[str, Any]): mandatory argument in Flower,
                                              can contain some configuration info

        Returns
        -------
            tuple[float, dict[str, Scalar]]: evaluation results
        """
        net: Module = set_model_parameters(model_generator(), parameters)
        net.to(device)

        valid_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        loss, acc = test_femnist(
            net=net,
            test_loader=valid_loader,
            device=device,
            criterion=criterion,
        )
        return loss, {"accuracy": acc}

    return federated_evaluation_function


def get_default_train_config() -> dict[str, Any]:
    """Get default training configuration."""
    return {
        "epochs": 8,
        "batch_size": 32,
        "client_learning_rate": 0.01,
        "weight_decay": 0.001,
        "num_workers": 0,
        "max_batches": 100,
    }


def get_default_test_config() -> dict[str, Any]:
    """Get default testing configuration."""
    return {
        "batch_size": 32,
        "num_workers": 0,
        "max_batches": 100,
    }
