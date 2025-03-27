from copy import deepcopy
import os
import json
import time
import logging
from typing import Dict, List, Union, Tuple, Optional, Any, Callable, Set
from pathlib import Path
from dataclasses import dataclass, asdict, field
import random
import argparse
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
import pandas as pd

# Flower imports
import flwr
from flwr.common import NDArrays, Scalar, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.client.client import Client
from flwr.server import History, ServerConfig
from flwr.server.strategy import FedAvgM as FedAvg

# C2M3 imports
from c2m3.utils.utils import set_seed
from c2m3.common.client_utils import (
    get_network_generator_tiny_resnet,
    get_network_generator_tiny_resnet_random,
    load_femnist_dataset, 
    get_network_generator_cnn,
    get_network_generator_cnn_random,
    train_femnist,
    test_femnist,
    save_history,
    get_model_parameters,
    set_model_parameters,
    load_cifar10_dataset
)

from c2m3.experiments.corrected_partition_data import partition_data as corrected_partition_data

from c2m3.match.merger import FrankWolfeSynchronizedMerger
from c2m3.match.permutation_spec import (
    CNNPermutationSpecBuilder,
    TinyResNetPermutationSpecBuilder,
    AutoPermutationSpecBuilder
)
from c2m3.models.utils import get_model_class
from c2m3.models.tiny_resnet import TinyResNet, BasicBlock
from c2m3.modules.pl_module import MyLightningModule
from c2m3.match.ties_merging import merge_models_ties
from flwr.server.strategy.aggregate import aggregate as flwr_aggregate

# Import our custom strategies
from c2m3.flower.fed_frank_wolfe_strategy import FrankWolfeSync
from c2m3.flower.fed_frank_wolfe_cifar_strategy import FrankWolfeSyncTinyResNet
from c2m3.flower.dataset_fed_frank_wolfe_strategy import DatasetFrankWolfeSync

# Setup logging - Configure root logger to capture both application and Flower logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# Ensure Flower logger is properly configured 
flwr_logger = logging.getLogger('flwr')
flwr_logger.setLevel(logging.INFO)

# Get our own logger
logger = logging.getLogger(__name__)

@dataclass
class FederatedExperimentConfig:
    """Configuration for federated learning experiments"""
    # Required parameters (no defaults)
    experiment_name: str
    dataset_name: str  # 'femnist', 'cifar10', 'cifar100'
    model_name: str  # 'resnet18', 'cnn'
    
    # Parameters with default values
    seed: int = 42
    data_dir: str = str(Path(__file__).parent.parent / "data")
    
    # Training configuration
    num_rounds: int = 10  # Number of federation rounds
    local_epochs: int = 1  # Number of local epochs per round
    batch_size: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    
    # Client configuration
    num_total_clients: int = 20  # Total number of available clients
    num_clients_per_round: int = 5  # Number of clients sampled each round
    min_train_samples: int = 10  # Minimum number of training samples per client 
    # Client Initialization
    initialization_type: str = "identical"  # 'identical' or 'random'
    
    # Aggregation configuration
    server_learning_rate: float = 1.0  # Learning rate for server aggregation (FedAvg)
    server_momentum: float = 0.0  # Momentum for server aggregation (FedAvg)
    aggregation_frequency: int = 1  # How often to aggregate (every X rounds)
    
    # Data distribution
    data_distribution: str = "natural"  # 'iid', 'dirichlet', 'pathological', 'natural'
    non_iid_alpha: float = 0.5  # Dirichlet alpha parameter for non-IID distribution
    classes_per_partition: int = 1  # For pathological partitioning, how many classes per client
    samples_per_partition: Optional[int] = None  # Number of samples per partition, None for equal division
    
    # Evaluation
    eval_batch_size: int = 128
    num_evaluation_clients: int = 0  # Number of clients used for validation (0 means centralized evaluation only)
                                    # If > 0, client-side evaluation will also be performed using clients' validation sets
                                    # This enables federated evaluation in addition to centralized evaluation
    
    # Output
    output_dir: str = "./results"
    save_models: bool = False
    save_results: bool = True
    
    def __post_init__(self):
        """Validate config and set derived parameters"""
        # Validate number of clients
        if self.num_clients_per_round > self.num_total_clients:
            raise ValueError(f"num_clients_per_round ({self.num_clients_per_round}) cannot be greater than num_total_clients ({self.num_total_clients})")
        
        # Validate aggregation frequency
        if self.aggregation_frequency <= 0:
            raise ValueError(f"aggregation_frequency must be positive, got {self.aggregation_frequency}")
        
        # Validate server learning rate
        if self.server_learning_rate <= 0:
            raise ValueError(f"server_learning_rate must be positive, got {self.server_learning_rate}")


class FlowerDatasetClient(flwr.client.NumPyClient):
    """Flower client for federated learning with direct dataset objects"""

    def __init__(
        self,
        cid: int,
        train_dataset: Dataset,
        test_dataset: Dataset,
        model_generator: Callable[[], nn.Module],
        properties: Dict[str, Scalar] = None
    ) -> None:
        """Initialize the client with its unique id and datasets.

        Parameters
        ----------
            cid (int): Unique client id for a client
            train_dataset (Dataset): Training dataset for this client
            test_dataset (Dataset): Test/validation dataset for this client. Used for local evaluation 
                                    when the server calls the client's evaluate() method.
                                    Note: This will only be used if num_evaluation_clients > 0 in the config.
            model_generator (Callable[[], Module]): The model generator function
            properties (Dict[str, Scalar], optional): Additional client properties
        """
        self.cid = cid
        logger.info(f"Initializing dataset client {self.cid} with dedicated partition (train samples: {len(train_dataset)}, validation samples: {len(test_dataset)})")
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model_generator = model_generator
        
        # Set properties
        self.properties: dict[str, Scalar] = {
            "tensor_type": "numpy.ndarray",
            "cid": self.cid,
            "partition": "in_memory"
        }
        
        # Add any additional properties
        if properties:
            self.properties.update(properties)

    def set_parameters(self, parameters: NDArrays) -> nn.Module:
        """Load weights inside the network.

        Parameters
        ----------
            parameters (NDArrays): set of weights to be loaded.

        Returns
        -------
            [Module]: Network with new set of weights.
        """
        net = self.model_generator()
        return set_model_parameters(net, parameters)

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return weights from a given model.

        Parameters
        ----------
            config (dict[int, Scalar]): dictionary containing configuration info.

        Returns
        -------
            NDArrays: weights from the model.
        """
        net = self.model_generator()
        return get_model_parameters(net)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict]:
        """Receive and train a model on the local client data.

        Parameters
        ----------
            parameters (NDArrays): Pytorch model parameters
            config (dict[str, Scalar]): dictionary describing the training parameters

        Returns
        -------
            tuple[NDArrays, int, dict]: Returns the updated model, the size of the local
                dataset and other metrics
        """
        # Only create model right before training/testing to lower memory usage when idle
        net = self.set_parameters(parameters)
        net.to(self.device)

        train_loader: DataLoader = self._create_data_loader(self.train_dataset, config, is_train=True)
        train_loss = self._train(net, train_loader=train_loader, config=config)
        return get_model_parameters(net), len(train_loader.dataset), {"train_loss": train_loss}

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict]:
        """Receive and test a model on the local client data.

        Parameters
        ----------
            parameters (NDArrays): Pytorch model parameters
            config (dict[str, Scalar]): dictionary describing the testing parameters

        Returns
        -------
            tuple[float, int, dict]: Returns the loss accumulated during testing, the
                size of the local dataset and other metrics such as accuracy
        """
        net = self.set_parameters(parameters)
        net.to(self.device)

        test_loader: DataLoader = self._create_data_loader(self.test_dataset, config, is_train=False)
        loss, accuracy = self._test(net, test_loader=test_loader, config=config)
        return loss, len(test_loader.dataset), {"local_accuracy": accuracy}

    def _create_data_loader(self, dataset: Dataset, config: dict[str, Scalar], is_train: bool = True) -> DataLoader:
        """Create the data loader using the specified config parameters.

        Parameters
        ----------
            dataset (Dataset): The dataset to load
            config (dict[str, Scalar]): dictionary containing dataloader parameters
            is_train (bool): Whether this is a training loader (affects shuffling)

        Returns
        -------
            DataLoader: A pytorch dataloader iterable for training/testing
        """
        batch_size = int(config["batch_size"])
        num_workers = int(config.get("num_workers", 0))
        
        if len(dataset) == 0:
            logger.warning(f"Empty dataset for client {self.cid}, creating empty DataLoader")
        
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=(is_train),  # Only drop last batch for training
        )

    def _train(
        self, net: nn.Module, train_loader: DataLoader, config: dict[str, Scalar]
    ) -> float:
        """Train the network on local data.
        
        Parameters
        ----------
            net (nn.Module): The neural network to train
            train_loader (DataLoader): DataLoader with training data
            config (dict[str, Scalar]): Configuration parameters
            
        Returns
        -------
            float: The training loss
        """
        # Determine dataset type based on dataset properties
        if hasattr(self.train_dataset, 'name') and 'femnist' in getattr(self.train_dataset, 'name', '').lower():
            # FEMNIST-specific training
            return train_femnist(
                net=net,
                train_loader=train_loader,
                epochs=int(config["local_epochs"]),
                device=self.device,
                optimizer=torch.optim.AdamW(
                    net.parameters(),
                    lr=float(config["learning_rate"]),
                    weight_decay=float(config["weight_decay"]),
                ),
                criterion=torch.nn.CrossEntropyLoss(),
                max_batches=int(config.get("max_batches")) if config.get("max_batches") is not None else None,
            )
        else:
            # General training for other datasets like CIFAR10
            net.train()
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                net.parameters(),
                lr=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
            )
            
            epochs = int(config["local_epochs"])
            total_loss = 0.0
            num_batches = 0
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Track loss
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    # Check if we should limit batches
                    max_batches = config.get("max_batches")
                    if max_batches is not None and batch_count >= int(max_batches):
                        break
                
                # Calculate average epoch loss
                if batch_count > 0:
                    avg_epoch_loss = epoch_loss / batch_count
                    total_loss += avg_epoch_loss
                    num_batches += 1
                    logger.info(f"Client {self.cid} - Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            
            # Return average loss across all epochs
            if num_batches > 0:
                return total_loss / num_batches
            else:
                return 0.0

    def _test(
        self, net: nn.Module, test_loader: DataLoader, config: dict[str, Scalar]
    ) -> tuple[float, float]:
        """Test the network on local data.
        
        Parameters
        ----------
            net (nn.Module): The neural network to test
            test_loader (DataLoader): DataLoader with test data
            config (dict[str, Scalar]): Configuration parameters
            
        Returns
        -------
            tuple[float, float]: The test loss and accuracy
        """
        if len(test_loader.dataset) == 0:
            logger.warning(f"Empty test dataset for client {self.cid}")
            return 0.0, 0.0
            
        # Determine dataset type based on dataset properties
        if hasattr(self.test_dataset, 'name') and 'femnist' in getattr(self.test_dataset, 'name', '').lower():
            # FEMNIST-specific testing
            return test_femnist(
                net=net,
                test_loader=test_loader,
                device=self.device,
                criterion=torch.nn.CrossEntropyLoss(),
                max_batches=int(config.get("max_batches")) if config.get("max_batches") is not None else None,
            )
        else:
            # General testing for other datasets like CIFAR10
            net.eval()
            criterion = torch.nn.CrossEntropyLoss()
            
            total_loss = 0.0
            correct = 0
            total = 0
            batch_count = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Track metrics
                    total_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    batch_count += 1
                    
                    # Check if we should limit batches
                    max_batches = config.get("max_batches")
                    if max_batches is not None and batch_count >= int(max_batches):
                        break
            
            # Calculate average loss and accuracy
            if total > 0:
                avg_loss = total_loss / total
                accuracy = correct / total
                logger.info(f"Client {self.cid} - Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                return avg_loss, accuracy
            else:
                logger.warning(f"Client {self.cid} - Empty test dataset")
                return 0.0, 0.0

    def get_properties(self, config: dict[str, Scalar]) -> dict[str, Scalar]:
        """Return properties for this client.

        Parameters
        ----------
            config (dict[str, Scalar]): Options to be used for selecting specific properties.

        Returns
        -------
            dict[str, Scalar]: Returned properties.
        """
        return self.properties

    def get_train_set_size(self) -> int:
        """Return the client train set size.

        Returns
        -------
            int: train set size of the client.
        """
        return len(self.train_dataset)

    def get_test_set_size(self) -> int:
        """Return the client test set size.

        Returns
        -------
            int: test set size of the client.
        """
        return len(self.test_dataset)


def get_flower_dataset_client_generator(
    model_generator: Callable[[], nn.Module],
    train_partitions: List[Dataset],
    val_partitions: List[Dataset],
    mapping_fn: Callable[[int], int] | None = None,
) -> Callable[[str], FlowerDatasetClient]:
    """Create a client generator function for Flower simulation using direct dataset objects.

    Parameters
    ----------
        model_generator (Callable[[], Module]): model generator function.
        train_partitions (List[Dataset]): List of training datasets, one per client
        val_partitions (List[Dataset]): List of validation/test datasets, one per client.
                                       In Flower, these are used for client-side evaluation
                                       when num_evaluation_clients > 0 in the config.
                                       Otherwise, they're not used in the federated learning process.
        mapping_fn (Optional[Callable[[int], int]]): function mapping sorted/filtered ids to real cid.

    Returns
    -------
        Callable[[str], FlowerDatasetClient]: client generator function.
    """

    def client_fn(cid: str) -> FlowerDatasetClient:
        """Create a single client instance given the client id `cid`.

        Parameters
        ----------
            cid (str): client id, Flower requires this to be of type str.

        Returns
        -------
            FlowerDatasetClient: client instance.
        """
        # Map client ID to partition index
        idx = mapping_fn(int(cid)) if mapping_fn is not None else int(cid)
        
        # Ensure index is within bounds by checking
        if idx >= len(train_partitions):
            raise ValueError(f"Client ID {idx} out of bounds. Only {len(train_partitions)} partitions available.")
        
        # Also ensure validation partitions exist for this client
        if idx >= len(val_partitions):
            logger.warning(f"No matching validation partition for client {idx}. Using validation partition 0.")
            val_idx = 0
        else:
            val_idx = idx
            
        # Each client gets a unique training and validation partition (no more modulo)
        return FlowerDatasetClient(
            cid=idx,
            train_dataset=train_partitions[idx],
            test_dataset=val_partitions[val_idx],
            model_generator=model_generator
        )

    return client_fn


def fit_client_seeded(
    client: FlowerDatasetClient,
    params: NDArrays,
    conf: dict[str, Any],
    seed: int,
    **kwargs: Any,
) -> tuple[NDArrays, int, dict]:
    """Wrap to always seed client training for reproducibility.
    
    Parameters
    ----------
        client (FlowerDatasetClient): The client to train
        params (NDArrays): Model parameters
        conf (dict[str, Any]): Configuration dictionary
        seed (int): Random seed
        
    Returns
    -------
        tuple[NDArrays, int, dict]: Client fit results
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return client.fit(params, conf, **kwargs)


class DataManager:
    """Manages dataset loading and partitioning for federated learning experiments."""
    
    def __init__(self, config: FederatedExperimentConfig):
        """Initialize the data manager with experiment configuration.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.dataset_name = config.dataset_name
        self.data_dir = Path(config.data_dir)
        self.data_distribution = config.data_distribution
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")
        
        # All available client IDs
        self.available_client_ids = []
        
        # Set random seed for client selection
        # This ensures client selection is reproducible across runs
        self.random_generator = random.Random(config.seed)
        
    def get_dataset(self):
        """Get the appropriate dataset based on configuration.
        
        Returns:
            tuple: (train_partitions, val_partitions, test_set, num_classes)
        """
        if self.config.dataset_name.lower() == "femnist":
            return self._get_femnist()
        elif self.config.dataset_name.lower() == "cifar10":
            return self._get_cifar10()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
            
    def _select_client_ids(self, available_ids, num_clients):
        """Randomly select client IDs.
        
        Args:
            available_ids: List of available client IDs
            num_clients: Number of clients to select
            
        Returns:
            List of selected client IDs
        """
        # Ensure we don't try to select more clients than available
        num_to_select = min(num_clients, len(available_ids))
        return random.sample(available_ids, num_to_select)
        
    def _get_available_client_ids(self, mapping_dir):
        """
        Get available client IDs from a mapping directory
        
        Args:
            mapping_dir: Path to the mapping directory
            
        Returns:
            List of available client IDs
        """
        client_ids = []
        
        if not mapping_dir.exists():
            logger.warning(f"Mapping directory {mapping_dir} does not exist")
            return client_ids
        
        for client_dir in mapping_dir.iterdir():
            if client_dir.is_dir():
                try:
                    client_id = int(client_dir.name)
                    client_ids.append(client_id)
                except ValueError:
                    # Skip directories with non-integer names
                    pass
        
        return sorted(client_ids)
            
    def _get_femnist(self):
        """
        Get FEMNIST dataset references
        
        Returns:
            Dictionary with FEMNIST dataset references
        """
        # Set up mapping directory
        partition_dir = self.data_dir / "femnist" / "client_data_mappings" / f"fed_{self.data_distribution}"
        
        # Get available client IDs
        self.available_client_ids = self._get_available_client_ids(partition_dir)
        
        return {
            "dataset_name": "femnist",
            "data_dir": self.data_dir,
            "partition_dir": partition_dir
        }
    
    def _get_cifar10(self):
        """
        Return references to CIFAR-10 data instead of actual datasets
        
        Returns:
            Dictionary with dataset metadata
        """
        return {
            "dataset_name": "cifar10",
            "data_dir": self.data_dir,
            "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        }

    def partition_data(self, data_refs, num_partitions):
        """
        Create data partitions based on the dataset type and distribution
        
        The approach is:
        1. First extract a universal test set from the full dataset (using stratified sampling)
        2. Then partition the remaining data (IID or otherwise) across models
        3. Finally, for each model, split its portion into training and validation sets
        
        Args:
            data_refs: References to data depending on the dataset
            num_partitions: Number of partitions to create
            
        Returns:
            Tuple of (train_partitions, val_partitions, test_set) for all datasets
        """        
        return corrected_partition_data(
            config=self.config,
            data_refs=data_refs,
            num_partitions=num_partitions,
            available_client_ids=self.available_client_ids
        )
    
    
    def create_dataloaders(self, train_partitions, val_partitions, test_set, batch_size, eval_batch_size):
        """
        Create DataLoaders for train, validation and test partitions
        
        Args:
            train_partitions: List of training dataset partitions
            val_partitions: List of validation dataset partitions
            test_set: Universal test dataset for evaluation
            batch_size: Batch size for training
            eval_batch_size: Batch size for evaluation (validation and test)
            
        Returns:
            Tuple of (train_loaders, val_loaders, test_loader)
        """
        # Special handling for FEMNIST dataset to ensure proper loading
        if self.dataset_name == "femnist":
            logger.info(f"Creating FEMNIST-specific dataloaders")
            train_loaders = [
                DataLoader(
                    partition, 
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    drop_last=True
                ) 
                for partition in train_partitions
            ]
            
            val_loaders = [
                DataLoader(
                    partition, 
                    batch_size=eval_batch_size,
                    shuffle=False,  # No need to shuffle validation data
                    num_workers=0,
                    drop_last=False  # Include all validation samples
                ) 
                for partition in val_partitions
            ]
            
            test_loader = DataLoader(
                test_set, 
                batch_size=eval_batch_size, 
                shuffle=False,  # No need to shuffle test data
                drop_last=False  # Include all test samples
            )
        elif self.dataset_name == "cifar10":
            logger.info(f"Creating CIFAR10-specific dataloaders")
            train_loaders = [
                DataLoader(
                    partition, 
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2,  # CIFAR10 can benefit from more workers
                    drop_last=True,
                    pin_memory=True  # Better performance with GPU
                ) 
                for partition in train_partitions
            ]
            
            val_loaders = [
                DataLoader(
                    partition, 
                    batch_size=eval_batch_size,
                    shuffle=False,
                    num_workers=2,
                    drop_last=False,
                    pin_memory=True
                ) 
                for partition in val_partitions
            ]
            
            test_loader = DataLoader(
                test_set, 
                batch_size=eval_batch_size, 
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                drop_last=False
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. Only 'femnist' and 'cifar10' are supported.")
        
        # Log the number of batches in each loader
        for i, loader in enumerate(train_loaders):
            logger.info(f"Train loader {i}: {len(loader)} batches, {len(loader.dataset)} samples")
        
        for i, loader in enumerate(val_loaders):
            logger.info(f"Validation loader {i}: {len(loader)} batches, {len(loader.dataset)} samples")
            
        logger.info(f"Test loader: {len(test_loader)} batches, {len(test_loader.dataset)} samples")
        
        return train_loaders, val_loaders, test_loader
        
    # def extract_samples_from_partition(self, partition):
    #     """Extract samples from a partition object.
        
    #     Handles different types of partitions:
    #     - Subset objects
    #     - FEMNISTFromSamples objects
    #     - Other Dataset objects if needed
        
    #     Args:
    #         partition: The partition to extract samples from
            
    #     Returns:
    #         List of (path, label) tuples
    #     """
    #     samples = []
        
    #     if isinstance(partition, torch.utils.data.Subset):
    #         # Handle Subset objects
    #         dataset = partition.dataset
    #         indices = partition.indices
            
    #         if hasattr(dataset, 'data') and isinstance(dataset.data, list):
    #             # FEMNISTFromSamples or similar
    #             samples = [dataset.data[idx] for idx in indices]
    #         else:
    #             # Handle other dataset types - this is a fallback
    #             if hasattr(dataset, 'data') and hasattr(dataset, 'targets'):
    #                 # CIFAR-like dataset
    #                 data = dataset.data
    #                 targets = dataset.targets
                    
    #                 if isinstance(data, np.ndarray) and isinstance(targets, list):
    #                     samples = [(data[idx], targets[idx]) for idx in indices]
    #                 elif hasattr(dataset, '__getitem__'):
    #                     # Last resort - use __getitem__ (slower)
    #                     samples = [dataset[idx] for idx in indices]
        
    #     elif hasattr(partition, 'data') and isinstance(partition.data, list):
    #         # Direct access to data attribute
    #         samples = partition.data
        
    #     elif hasattr(partition, '__getitem__') and hasattr(partition, '__len__'):
    #         # Last resort - use __getitem__ (slower)
    #         samples = [partition[i] for i in range(len(partition))]
            
    #     return samples


class FederatedExperimentRunner:
    """Main class for running federated learning experiments"""
    def __init__(self, config: FederatedExperimentConfig, use_direct_partitioning: bool = True):
        self.config = config
        self.experiment_name = config.experiment_name
        
        # Set whether to use direct partitioning or predefined partitions
        self.use_direct_partitioning = use_direct_partitioning
        
        # Set random seed for reproducibility
        set_seed(config.seed)
        
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create output directory for results
        self.output_dir = Path(config.output_dir) / self.experiment_name
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save configuration
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=4)
            
        # Initialize results dictionary
        self.results = {
            "config": asdict(self.config),
            "rounds": [],
            "final_model": {},
            "runtime": 0
        }
        
        self.model_generator = None
        self.partition_dir = None  # Directory containing client partitions
        self.client_ids = []
        self.test_dataset = None
        self.global_test_loader = None
        self.parameters_for_each_round = None
        self.history = None
        
        # Random number generator for client selection
        self.random_generator = random.Random(config.seed)
        
    def setup(self):
        """Set up the experiment, loading datasets and setting up the model."""
        logger.info(f"Setting up federated experiment: {self.config.experiment_name}")

        self.data_manager = DataManager(self.config)
        
        # Get data references for the specified dataset
        data_refs = self.data_manager.get_dataset()
        
        # Store the partition directory path for client access
        if self.config.dataset_name.lower() == "femnist":
            self.partition_dir = data_refs.get("partition_dir")
            self.available_client_ids = self.data_manager.available_client_ids
            logger.info(f"Using FEMNIST partition directory: {self.partition_dir}")
            logger.info(f"Found {len(self.available_client_ids)} available client IDs")
        elif self.config.dataset_name.lower() == "cifar10":
            self.partition_dir = data_refs.get("partition_dir")
            self.available_client_ids = self.data_manager.available_client_ids
            logger.info(f"Using CIFAR10 partition directory: {self.partition_dir}")
            logger.info(f"Found {len(self.available_client_ids)} available client IDs")
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}. This implementation only supports FEMNIST and CIFAR10.")
        
        # Get test set for centralized evaluation
        _, _, self.test_dataset = self.data_manager.partition_data(data_refs, 1)  # We only need the test set
        
        # Set up model generator
        if self.config.dataset_name.lower() == "femnist":
            if self.config.initialization_type == "identical":
                network_generator = get_network_generator_cnn()
                self.models = [network_generator() for _ in range(self.config.num_clients_per_round)]
                logger.info(f"Created {len(self.models)} identical FEMNIST models")
            elif self.config.initialization_type == "random":
                network_generator = get_network_generator_cnn_random()
                self.models = [network_generator() for _ in range(self.config.num_clients_per_round)]
                logger.info(f"Created {len(self.models)} FEMNIST models with diverse initialization")
            else: 
                raise ValueError(f"Unsupported initialization type: {self.config.initialization_type}")
        elif self.config.dataset_name.lower() == "cifar10":
            if self.config.initialization_type == "identical":
                network_generator = get_network_generator_tiny_resnet()
                self.models = [network_generator() for _ in range(self.config.num_clients_per_round)]
                logger.info("Using TinyResNet model for CIFAR10")
            elif self.config.initialization_type == "random":
                network_generator = get_network_generator_tiny_resnet_random()
                self.models = [network_generator() for _ in range(self.config.num_clients_per_round)]
                logger.info(f"Created {len(self.models)} CIFAR10 models with diverse initialization")
            else:
                raise ValueError(f"Unsupported initialization type: {self.config.initialization_type}")
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
        
        self.model_generator = network_generator
        
        # Set up client IDs - use the available client IDs from the partition directory
        if len(self.available_client_ids) > 0:
            self.client_ids = self.available_client_ids
            logger.info(f"Using {len(self.client_ids)} available client IDs from partition directory")
        else:
            # Fallback: use sequential IDs if no clients found
            self.client_ids = list(range(self.config.num_total_clients))
            logger.warning(f"No client IDs found in partition directory. Creating {len(self.client_ids)} sequential client IDs.")
        
        # Create global test loader
        self.global_test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        logger.info(f"Created global test loader with {len(self.test_dataset)} samples")
        
        # Initialize the parameters and history
        self.parameters_for_each_round = []
        self.history = History()
        
    def _select_client_ids(self, available_ids, num_clients):
        """Select a subset of client IDs deterministically based on seed"""
        if len(available_ids) >= num_clients:
            # Use our seeded random generator instead of the global one
            return self.random_generator.sample(available_ids, num_clients)
        else:
            logger.warning(f"Requested {num_clients} clients but only {len(available_ids)} available. Using all available clients.")
            return available_ids

    def _create_federated_train_config(self) -> Dict[str, Any]:
        """Create training configuration for federated clients.
        
        Returns:
            Dict[str, Any]: Configuration dictionary for client training
        """
        return {
            "local_epochs": self.config.local_epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "num_workers": 0,
            "max_batches": None  # Use all batches
        }
    
    def _create_test_config(self) -> Dict[str, Any]:
        """Create testing configuration for federated clients.
        
        Returns:
            Dict[str, Any]: Configuration dictionary for client evaluation
        """
        return {
            "batch_size": self.config.eval_batch_size,
            "num_workers": 0,
            "max_batches": None  # Use all batches
        }
    
    def _evaluate_global_model(self, parameters: NDArrays) -> Dict[str, float]:
        """Evaluate the global model on the test dataset.
        
        Parameters:
            parameters (NDArrays): Model parameters to evaluate
            
        Returns:
            Dict[str, float]: Dictionary with evaluation metrics
        """
        # Create model and load parameters
        model = self.model_generator()
        model = set_model_parameters(model, parameters)
        model.to(self.device)
        model.eval()
        
        # Evaluate based on dataset type
        if self.config.dataset_name == "femnist":
            # FEMNIST-specific evaluation
            if hasattr(self, 'test_dataset') and hasattr(self, 'global_test_loader'):
                test_loss, test_acc = test_femnist(
                    net=model,
                    test_loader=self.global_test_loader,
                    device=self.device,
                    criterion=nn.CrossEntropyLoss(),
                    max_batches=None  # Use all batches
                )
                logger.info(f"FEMNIST global model evaluation - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
                return {
                    "loss": test_loss,
                    "accuracy": test_acc
                }
            else:
                logger.warning("No FEMNIST test dataset available for evaluation")
                return {
                    "loss": float('inf'),
                    "accuracy": 0.0
                }
        elif self.config.dataset_name == "cifar10":
            # CIFAR10-specific evaluation
            if hasattr(self, 'test_dataset') and hasattr(self, 'global_test_loader'):
                # Evaluate CIFAR10 model
                criterion = nn.CrossEntropyLoss()
                total_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, targets in self.global_test_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        total_loss += loss.item() * inputs.size(0)
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                
                # Calculate metrics for CIFAR10
                if total > 0:
                    avg_loss = total_loss / total
                    accuracy = correct / total
                    logger.info(f"CIFAR10 global model evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                    return {
                        "loss": avg_loss,
                        "accuracy": accuracy
                    }
                else:
                    logger.warning("Empty CIFAR10 test dataset")
                    return {
                        "loss": float('inf'),
                        "accuracy": 0.0
                    }
            else:
                logger.warning("No CIFAR10 test dataset available for evaluation")
                return {
                    "loss": float('inf'),
                    "accuracy": 0.0
                }
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}. Only 'femnist' and 'cifar10' are supported.")
    
    def run(self):
        """Run the federated learning experiment."""
        logger.info(f"Running federated experiment '{self.experiment_name}'...")
        start_time = time.time()
        
        # Check if partition directory exists
        if self.partition_dir is None or not self.partition_dir.exists():
            logger.error(f"Partition directory not found: {self.partition_dir}")
            raise FileNotFoundError(f"Partition directory not found: {self.partition_dir}")
        
        # Create client generator using FlowerRayClient
        federated_client_generator = get_flower_client_generator(
            model_generator=self.model_generator,
            partition_dir=self.partition_dir,
            data_dir=Path(self.config.data_dir)
        )
        logger.info(f"Using file-based client generator with partitions from: {self.partition_dir}")
        
        # Create evaluation function for central evaluation
        def federated_evaluation_function(
            server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            """Evaluate the global model on a central test set."""
            metrics = self._evaluate_global_model(parameters)
            return metrics["loss"], {"accuracy": metrics["accuracy"]}
        
        # Set up training and test configs
        federated_train_config = self._create_federated_train_config()
        test_config = self._create_test_config()
        
        # Create functions to return configs
        def fit_config_fn(server_round: int) -> Dict[str, Scalar]:
            """Return training configuration for clients."""
            return federated_train_config
        
        def eval_config_fn(server_round: int) -> Dict[str, Scalar]:
            """Return evaluation configuration for clients."""
            return test_config
        
        # Get initial parameters from the model
        initial_model = self.model_generator()
        initial_parameters = get_model_parameters(initial_model)
        initial_parameters_flwr = ndarrays_to_parameters(initial_parameters)
        
        # Calculate fractions for client selection
        fraction_fit = self.config.num_clients_per_round / max(len(self.client_ids), 1)
        fraction_evaluate = self.config.num_evaluation_clients / max(len(self.client_ids), 1)
        
        # Log information about client evaluation
        if self.config.num_evaluation_clients > 0:
            logger.info(f"Client-side evaluation enabled with {self.config.num_evaluation_clients} evaluation clients")
        else:
            logger.info("Client-side evaluation disabled (num_evaluation_clients=0)")
            logger.info("Only centralized evaluation on the global test set will be performed")

        # Create strategy based on dataset name
        if self.config.dataset_name.lower() == "femnist":
            # For FEMNIST, use FrankWolfeSync with file-based strategy
            try:
                from c2m3.flower.fed_frank_wolfe_strategy import FrankWolfeSync
                logger.info("Using FrankWolfeSync strategy for FEMNIST")
                strategy = FrankWolfeSync(
                    fraction_fit=fraction_fit,
                    fraction_evaluate=fraction_evaluate,
                    min_fit_clients=self.config.num_clients_per_round,
                    min_evaluate_clients=self.config.num_evaluation_clients,
                    min_available_clients=len(self.client_ids),
                    on_fit_config_fn=fit_config_fn,
                    on_evaluate_config_fn=eval_config_fn,
                    evaluate_fn=federated_evaluation_function,
                    initial_parameters=initial_parameters_flwr,
                    accept_failures=False
                )
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"FrankWolfeSync strategy not available: {str(e)}. Falling back to FedAvg")
                strategy = FedAvg(
                    fraction_fit=fraction_fit,
                    fraction_evaluate=fraction_evaluate,
                    min_fit_clients=self.config.num_clients_per_round,
                    min_evaluate_clients=self.config.num_evaluation_clients,
                    min_available_clients=len(self.client_ids),
                    on_fit_config_fn=fit_config_fn,
                    on_evaluate_config_fn=eval_config_fn,
                    evaluate_fn=federated_evaluation_function,
                    initial_parameters=initial_parameters_flwr,
                    accept_failures=False,
                    server_learning_rate=self.config.server_learning_rate,
                    server_momentum=self.config.server_momentum
                )
        elif self.config.dataset_name.lower() == "cifar10":
            # For CIFAR10, try to use FrankWolfeSyncTinyResNet
            try:
                from c2m3.flower.fed_frank_wolfe_cifar_strategy import FrankWolfeSyncTinyResNet
                logger.info("Using FrankWolfeSyncTinyResNet strategy for CIFAR10")
                strategy = FrankWolfeSyncTinyResNet(
                    fraction_fit=fraction_fit,
                    fraction_evaluate=fraction_evaluate,
                    min_fit_clients=self.config.num_clients_per_round,
                    min_evaluate_clients=self.config.num_evaluation_clients,
                    min_available_clients=len(self.client_ids),
                    on_fit_config_fn=fit_config_fn,
                    on_evaluate_config_fn=eval_config_fn,
                    evaluate_fn=federated_evaluation_function,
                    initial_parameters=initial_parameters_flwr,
                    accept_failures=False
                )
            except (ImportError, ModuleNotFoundError) as e:
                logger.warning(f"FrankWolfeSyncTinyResNet strategy not available: {str(e)}. Falling back to FedAvg")
                strategy = FedAvg(
                    fraction_fit=fraction_fit,
                    fraction_evaluate=fraction_evaluate,
                    min_fit_clients=self.config.num_clients_per_round,
                    min_evaluate_clients=self.config.num_evaluation_clients,
                    min_available_clients=len(self.client_ids),
                    on_fit_config_fn=fit_config_fn,
                    on_evaluate_config_fn=eval_config_fn,
                    evaluate_fn=federated_evaluation_function,
                    initial_parameters=initial_parameters_flwr,
                    accept_failures=False,
                    server_learning_rate=self.config.server_learning_rate,
                    server_momentum=self.config.server_momentum
                )
        else:
            # For other datasets, use standard FedAvg
            logger.warning(f"Using standard FedAvg strategy for {self.config.dataset_name}")
            strategy = FedAvg(
                fraction_fit=fraction_fit,
                fraction_evaluate=fraction_evaluate,
                min_fit_clients=self.config.num_clients_per_round,
                min_evaluate_clients=self.config.num_evaluation_clients,
                min_available_clients=len(self.client_ids),
                on_fit_config_fn=fit_config_fn,
                on_evaluate_config_fn=eval_config_fn,
                evaluate_fn=federated_evaluation_function,
                initial_parameters=initial_parameters_flwr,
                accept_failures=False,
                server_learning_rate=self.config.server_learning_rate,
                server_momentum=self.config.server_momentum
            )
        
        # Create server config with custom aggregation frequency
        server_config = ServerConfig(num_rounds=self.config.num_rounds)
        
        # Define helper function for starting simulation with fixed seed
        def start_seeded_simulation(
            client_fn: Callable[[str], Client],
            num_clients: int,
            config: ServerConfig,
            strategy,
            seed: int,
            **kwargs: Any
        ) -> Tuple[List[Tuple[int, NDArrays]], History]:
            """Start simulation with fixed seed for reproducibility."""
            # Set seed before simulation
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            
            # Start simulation using no-Ray implementation to avoid memory issues
            parameter_list, hist = flwr.simulation.start_simulation_no_ray(
                client_fn=client_fn,
                num_clients=num_clients,
                client_resources={},
                config=config,
                strategy=strategy,
            )
            return parameter_list, hist
        
        # Convert client generator to return Client instances (not NumPyClient)
        def simulator_client_generator(cid: str) -> Client:
            logger.debug(f"Creating client with ID: {cid}")
            try:
                numpy_client = federated_client_generator(cid)
                return numpy_client.to_client()
            except Exception as e:
                logger.error(f"Error creating client with ID {cid}: {str(e)}")
                raise
        
        # Run the simulation
        logger.info(f"Starting federated simulation with {self.config.num_rounds} rounds")
        logger.info(f"Strategy: {strategy.__class__.__name__}")
        logger.info(f"Number of total clients: {len(self.client_ids)}")
        logger.info(f"Number of clients per round: {self.config.num_clients_per_round}")

        try:
            self.parameters_for_each_round, self.history = start_seeded_simulation(
                client_fn=simulator_client_generator,
                num_clients=len(self.client_ids),
                config=server_config,
                strategy=strategy,
                seed=self.config.seed,
            )
            logger.info("Federated simulation completed successfully")
        except Exception as e:
            logger.error(f"Error in federated simulation: {str(e)}")
            raise
        
        # Extract and store results
        metrics_distributed = self.history.metrics_distributed
        metrics_centralized = self.history.metrics_centralized
        
        # Process results for each round
        for round_idx in range(self.config.num_rounds):
            round_results = {
                "round": round_idx + 1,
                "distributed": {},
                "centralized": {}
            }
            
            # Extract distributed metrics if available
            if metrics_distributed and "fit" in metrics_distributed:
                fit_metrics = metrics_distributed["fit"]
                if len(fit_metrics) > round_idx:
                    round_results["distributed"]["fit"] = fit_metrics[round_idx][1]
            
            if metrics_distributed and "evaluate" in metrics_distributed:
                eval_metrics = metrics_distributed["evaluate"]
                if len(eval_metrics) > round_idx:
                    round_results["distributed"]["evaluate"] = eval_metrics[round_idx][1]
            
            # Extract centralized metrics if available
            if metrics_centralized and "evaluate" in metrics_centralized:
                central_metrics = metrics_centralized["evaluate"]
                if len(central_metrics) > round_idx:
                    round_results["centralized"]["evaluate"] = central_metrics[round_idx][1]
            
            # Store round results
            self.results["rounds"].append(round_results)
        
        # Get final model parameters
        final_parameters = self.parameters_for_each_round[-1][1] if self.parameters_for_each_round else initial_parameters
        
        # Evaluate final model
        final_metrics = self._evaluate_global_model(final_parameters)
        self.results["final_model"] = {
            "metrics": final_metrics,
            "parameters": None  # We don't store the actual parameters in the JSON results
        }
        
        # Save final model if requested
        if self.config.save_models:
            final_model = self.model_generator()
            final_model = set_model_parameters(final_model, final_parameters)
            torch.save(final_model.state_dict(), self.output_dir / "final_model.pt")
        
        # Record total runtime
        self.results["runtime"] = time.time() - start_time
        
        # Save results
        if self.config.save_results:
            with open(self.output_dir / "results.json", "w") as f:
                # Convert numpy values to standard Python types for JSON serialization
                serializable_results = self._make_json_serializable(self.results)
                json.dump(serializable_results, f, indent=4)
            
            # Save the Flower history object
            save_history(self.output_dir, self.history, "flower_history")
        
        logger.info(f"Experiment '{self.experiment_name}' completed in {self.results['runtime']:.2f} seconds.")
        logger.info(f"Final model accuracy: {final_metrics['accuracy']:.4f}")
        
        return self.results
    
    def _make_json_serializable(self, obj):
        """Convert all numpy values to standard Python types for JSON serialization."""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        else:
            return obj

    def visualize_results(self):
        """Visualize the results of the federated learning experiment."""
        if not self.results or not self.results.get("rounds"):
            logger.error("No results to visualize. Run the experiment first.")
            return
        
        # Set up the figure
        plt.figure(figsize=(12, 8))
        
        # Extract round numbers and accuracies
        rounds = [r["round"] for r in self.results["rounds"]]
        accuracies = []
        
        for r in self.results["rounds"]:
            if "centralized" in r and "evaluate" in r["centralized"] and "accuracy" in r["centralized"]["evaluate"]:
                accuracies.append(r["centralized"]["evaluate"]["accuracy"])
            else:
                # If no centralized evaluation, try to get distributed evaluation
                acc = 0.0
                if "distributed" in r and "evaluate" in r["distributed"]:
                    metrics = r["distributed"]["evaluate"]
                    if "local_accuracy" in metrics:
                        acc = metrics["local_accuracy"]
                accuracies.append(acc)
        
        # Plot accuracy over rounds
        plt.plot(rounds, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title(f'Federated Learning Experiment: {self.experiment_name}', fontsize=16)
        
        # Add horizontal line for final accuracy
        final_acc = self.results["final_model"]["metrics"]["accuracy"]
        plt.axhline(y=final_acc, color='r', linestyle='--', alpha=0.7)
        plt.text(0.5, final_acc + 0.01, f'Final Accuracy: {final_acc:.4f}', 
                 horizontalalignment='center', color='r')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.output_dir / "accuracy_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # If we have train loss data, plot it
        plt.figure(figsize=(12, 8))
        train_losses = []
        
        for r in self.results["rounds"]:
            if "distributed" in r and "fit" in r["distributed"] and "train_loss" in r["distributed"]["fit"]:
                train_losses.append(r["distributed"]["fit"]["train_loss"])
            else:
                train_losses.append(None)  # Handle missing data
        
        # Only plot if we have loss data
        if any(loss is not None for loss in train_losses):
            # Filter out None values for plotting
            valid_rounds = [r for r, loss in zip(rounds, train_losses) if loss is not None]
            valid_losses = [loss for loss in train_losses if loss is not None]
            
            plt.plot(valid_rounds, valid_losses, marker='o', linestyle='-', linewidth=2, markersize=8, color='orange')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Round', fontsize=14)
            plt.ylabel('Training Loss', fontsize=14)
            plt.title(f'Training Loss: {self.experiment_name}', fontsize=16)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(self.output_dir / "train_loss_plot.png", dpi=300, bbox_inches="tight")
        
        plt.close()
        
        # Print summary statistics
        logger.info("=== Experiment Summary ===")
        logger.info(f"Runtime: {self.results['runtime']:.2f} seconds")
        logger.info(f"Final model accuracy: {final_acc:.4f}")
        logger.info(f"Number of rounds: {self.config.num_rounds}")
        logger.info(f"Local epochs per round: {self.config.local_epochs}")
        logger.info(f"Clients per round: {self.config.num_clients_per_round}/{self.config.num_total_clients}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Server learning rate: {self.config.server_learning_rate}")
        
        return {"final_accuracy": final_acc}


def run_parameter_sweep(base_config: FederatedExperimentConfig, parameter_grid: Dict[str, List[Any]]):
    """Run experiments with all combinations of parameters in the grid.
    
    Parameters:
        base_config (FederatedExperimentConfig): Base configuration for experiments
        parameter_grid (Dict[str, List[Any]]): Dictionary mapping parameter names to lists of values
        
    Returns:
        List[Dict]: List of experiment results
    """
    # Generate all combinations of parameters
    from itertools import product
    
    # Get keys and values from parameter grid
    keys = parameter_grid.keys()
    values = parameter_grid.values()
    
    results = []
    
    # Iterate through all combinations
    for combination in product(*values):
        # Create a copy of the base config
        config_dict = asdict(base_config)
        
        # Update with the current parameter combination
        for key, value in zip(keys, combination):
            config_dict[key] = value
        
        # Create a unique experiment name based on varied parameters
        param_str = "_".join(f"{key}={value}" for key, value in zip(keys, combination))
        config_dict["experiment_name"] = f"{base_config.experiment_name}_{param_str}"
        
        # Create config and runner
        config = FederatedExperimentConfig(**config_dict)
        runner = FederatedExperimentRunner(config)
        
        try:
            # Setup and run experiment
            runner.setup()
            result = runner.run()
            runner.visualize_results()
            results.append(result)
        except Exception as e:
            logger.error(f"Error running experiment {config_dict['experiment_name']}: {e}")
            continue
    
    return results


# def run_learning_rate_experiment(base_config, learning_rates, output_dir="./results/federated_experiments"):
#     """Run experiments with different learning rates for client training.
    
#     Parameters:
#         base_config (FederatedExperimentConfig): Base configuration for experiments
#         learning_rates (List[float]): List of learning rate values to try
#         output_dir (str): Directory to save results
        
#     Returns:
#         pd.DataFrame: DataFrame with experiment results
#     """
#     parameter_grid = {"learning_rate": learning_rates}
#     results = run_parameter_sweep(base_config, parameter_grid)
    
#     # Convert results to DataFrame for analysis
#     df_data = []
#     for result in results:
#         config = result["config"]
#         final_accuracy = result["final_model"]["metrics"]["accuracy"]
#         df_data.append({
#             "learning_rate": config["learning_rate"],
#             "final_accuracy": final_accuracy,
#             "runtime": result["runtime"]
#         })
    
#     df = pd.DataFrame(df_data)
    
#     # Plot results
#     plt.figure(figsize=(10, 6))
#     plt.plot(df["learning_rate"], df["final_accuracy"], marker='o', linestyle='-')
#     plt.xscale('log')  # Learning rates are often compared on a log scale
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.xlabel('Client Learning Rate', fontsize=14)
#     plt.ylabel('Final Accuracy', fontsize=14)
#     plt.title('Effect of Client Learning Rate on Model Accuracy', fontsize=16)
    
#     # Create output directory
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True, parents=True)
    
#     # Save plot and data
#     plt.tight_layout()
#     plt.savefig(output_path / "learning_rate_comparison.png", dpi=300, bbox_inches="tight")
#     df.to_csv(output_path / "learning_rate_comparison.csv", index=False)
    
#     return df


def run_aggregation_frequency_experiment(base_config, frequencies, output_dir="./results/federated_experiments"):
    """Run experiments with different aggregation frequencies.
    
    Parameters:
        base_config (FederatedExperimentConfig): Base configuration for experiments
        frequencies (List[int]): List of aggregation frequency values to try
        output_dir (str): Directory to save results
        
    Returns:
        pd.DataFrame: DataFrame with experiment results
    """
    parameter_grid = {"aggregation_frequency": frequencies}
    results = run_parameter_sweep(base_config, parameter_grid)
    
    # Convert results to DataFrame for analysis
    df_data = []
    for result in results:
        config = result["config"]
        final_accuracy = result["final_model"]["metrics"]["accuracy"]
        df_data.append({
            "aggregation_frequency": config["aggregation_frequency"],
            "final_accuracy": final_accuracy,
            "runtime": result["runtime"]
        })
    
    df = pd.DataFrame(df_data)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(df["aggregation_frequency"], df["final_accuracy"], marker='o', linestyle='-')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Aggregation Frequency (rounds)', fontsize=14)
    plt.ylabel('Final Accuracy', fontsize=14)
    plt.title('Effect of Aggregation Frequency on Model Accuracy', fontsize=16)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save plot and data
    plt.tight_layout()
    plt.savefig(output_path / "aggregation_frequency_comparison.png", dpi=300, bbox_inches="tight")
    df.to_csv(output_path / "aggregation_frequency_comparison.csv", index=False)
    
    return df


# def run_server_lr_experiment(base_config, server_learning_rates, output_dir="./results/federated_experiments"):
#     """Run experiments with different server learning rates.
    
#     Parameters:
#         base_config (FederatedExperimentConfig): Base configuration for experiments
#         server_learning_rates (List[float]): List of server learning rate values to try
#         output_dir (str): Directory to save results
        
#     Returns:
#         pd.DataFrame: DataFrame with experiment results
#     """
#     parameter_grid = {"server_learning_rate": server_learning_rates}
#     results = run_parameter_sweep(base_config, parameter_grid)
    
#     # Convert results to DataFrame for analysis
#     df_data = []
#     for result in results:
#         config = result["config"]
#         final_accuracy = result["final_model"]["metrics"]["accuracy"]
#         df_data.append({
#             "server_learning_rate": config["server_learning_rate"],
#             "final_accuracy": final_accuracy,
#             "runtime": result["runtime"]
#         })
    
#     df = pd.DataFrame(df_data)
    
#     # Plot results
#     plt.figure(figsize=(10, 6))
#     plt.plot(df["server_learning_rate"], df["final_accuracy"], marker='o', linestyle='-')
#     plt.xscale('log')  # Learning rates are often compared on a log scale
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.xlabel('Server Learning Rate', fontsize=14)
#     plt.ylabel('Final Accuracy', fontsize=14)
#     plt.title('Effect of Server Learning Rate on Model Accuracy', fontsize=16)
    
#     # Create output directory
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True, parents=True)
    
#     # Save plot and data
#     plt.tight_layout()
#     plt.savefig(output_path / "server_lr_comparison.png", dpi=300, bbox_inches="tight")
#     df.to_csv(output_path / "server_lr_comparison.csv", index=False)
    
#     return df


# def run_local_epochs_experiment(base_config, epochs_values, output_dir="./results/federated_experiments"):
#     """Run experiments with different numbers of local epochs.
    
#     Parameters:
#         base_config (FederatedExperimentConfig): Base configuration for experiments
#         epochs_values (List[int]): List of local epoch values to try
#         output_dir (str): Directory to save results
        
#     Returns:
#         pd.DataFrame: DataFrame with experiment results
#     """
#     parameter_grid = {"local_epochs": epochs_values}
#     results = run_parameter_sweep(base_config, parameter_grid)
    
#     # Convert results to DataFrame for analysis
#     df_data = []
#     for result in results:
#         config = result["config"]
#         final_accuracy = result["final_model"]["metrics"]["accuracy"]
#         df_data.append({
#             "local_epochs": config["local_epochs"],
#             "final_accuracy": final_accuracy,
#             "runtime": result["runtime"]
#         })
    
#     df = pd.DataFrame(df_data)
    
#     # Plot results
#     plt.figure(figsize=(10, 6))
#     plt.plot(df["local_epochs"], df["final_accuracy"], marker='o', linestyle='-')
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.xlabel('Local Epochs', fontsize=14)
#     plt.ylabel('Final Accuracy', fontsize=14)
#     plt.title('Effect of Local Epochs on Model Accuracy', fontsize=16)
    
#     # Create output directory
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True, parents=True)
    
#     # Save plot and data
#     plt.tight_layout()
#     plt.savefig(output_path / "local_epochs_comparison.png", dpi=300, bbox_inches="tight")
#     df.to_csv(output_path / "local_epochs_comparison.csv", index=False)
    
#     return df


# def run_multi_seed_experiment(base_config, seeds, output_dir="./results/federated_experiments"):
#     """Run experiments with multiple random seeds to assess stability.
    
#     Parameters:
#         base_config (FederatedExperimentConfig): Base configuration for experiments
#         seeds (List[int]): List of random seed values to try
#         output_dir (str): Directory to save results
        
#     Returns:
#         pd.DataFrame: DataFrame with experiment results
#     """
#     parameter_grid = {"seed": seeds}
#     results = run_parameter_sweep(base_config, parameter_grid)
    
#     # Convert results to DataFrame for analysis
#     df_data = []
#     for result in results:
#         config = result["config"]
#         final_accuracy = result["final_model"]["metrics"]["accuracy"]
#         df_data.append({
#             "seed": config["seed"],
#             "final_accuracy": final_accuracy,
#             "runtime": result["runtime"]
#         })
    
#     df = pd.DataFrame(df_data)
    
#     # Calculate statistics
#     mean_accuracy = df["final_accuracy"].mean()
#     std_accuracy = df["final_accuracy"].std()
    
#     # Plot results as a bar chart
#     plt.figure(figsize=(10, 6))
#     plt.bar(df["seed"].astype(str), df["final_accuracy"])
#     plt.axhline(y=mean_accuracy, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_accuracy:.4f} ± {std_accuracy:.4f}')
#     plt.grid(True, linestyle='--', alpha=0.7, axis='y')
#     plt.xlabel('Random Seed', fontsize=14)
#     plt.ylabel('Final Accuracy', fontsize=14)
#     plt.title('Model Accuracy Across Different Random Seeds', fontsize=16)
#     plt.legend()
    
#     # Create output directory
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True, parents=True)
    
#     # Save plot and data
#     plt.tight_layout()
#     plt.savefig(output_path / "multi_seed_comparison.png", dpi=300, bbox_inches="tight")
#     df.to_csv(output_path / "multi_seed_comparison.csv", index=False)
    
#     # Print summary
#     logger.info(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
#     return df


def run_multi_seed_learning_rates_experiment(base_config, learning_rates, seeds=[42, 123, 456, 789, 101], output_dir=None):
    """
    Run federated learning experiments with multiple seeds and learning rates.
    
    Args:
        base_config (FederatedExperimentConfig): Base configuration for experiments
        learning_rates (List[float]): Learning rate values to evaluate
        seeds (List[int]): Random seeds to use for reproducibility
        output_dir (str, optional): Directory to save results, defaults to "./results/{experiment_name}"
    
    Returns:
        Dict: Dictionary mapping learning rates to lists of result dictionaries (one per seed)
    """
    # Make sure learning rates are sorted
    learning_rates = sorted(learning_rates)
    
    # Structure to hold all results
    all_results = {lr: [] for lr in learning_rates}
    
    # Create base experiment name once
    base_experiment_name = base_config.experiment_name
    
    # Create parent directory for all experiment results
    from pathlib import Path
    if output_dir is None:
        parent_dir = Path(f"./results/{base_experiment_name}_multi_seed_lr")
    else:
        parent_dir = Path(output_dir)
    parent_dir.mkdir(exist_ok=True, parents=True)
    
    # For each seed
    for idx, seed in enumerate(seeds):
        print(f"Using seed {seed}...")
        
        # For each learning rate
        for lr in learning_rates:
            print(f"  Training with learning rate {lr}...")
            
            # Create a new config with the seed and learning rate changed
            from dataclasses import asdict
            config_dict = asdict(base_config)
            
            # Save original experiment name before removing from dict
            original_name = config_dict["experiment_name"]
            
            # Remove experiment_name from the dictionary to avoid duplication
            del config_dict["experiment_name"]
            
            # Update the seed and learning rate
            config_dict["seed"] = seed
            config_dict["learning_rate"] = lr
            
            # Modify the output directory to be within the parent directory
            config_dict["output_dir"] = str(parent_dir)
            
            # Create a new experiment name including seed and learning rate
            experiment_name = f"{original_name}_seed{seed}_lr{lr}"
            
            # Create the experiment config
            current_config = FederatedExperimentConfig(
                **config_dict,
                experiment_name=experiment_name
            )
            
            # Create and run the experiment
            runner = FederatedExperimentRunner(current_config)
            runner.setup()
            result = runner.run()
            
            # Optional: Generate visualizations
            runner.visualize_results()
            
            # Add a field to track which learning rate this run is associated with
            result["tracked_learning_rate"] = lr
            result["tracked_seed"] = seed
            
            # Store the result
            all_results[lr].append(result)
    
    # Now create visualizations across all seeds and learning rates
    visualize_learning_rate_results(all_results, base_experiment_name, parent_dir, seeds)
    
    return all_results


def run_multi_seed_aggregation_freq_experiment(base_config, aggregation_frequencies, seeds=[42, 123, 456, 789, 101], output_dir=None):
    """
    Run federated learning experiments with multiple seeds and aggregation frequencies.
    
    Args:
        base_config (FederatedExperimentConfig): Base configuration for experiments
        aggregation_frequencies (List[int]): Aggregation frequency values to evaluate
        seeds (List[int]): Random seeds to use for reproducibility
        output_dir (str, optional): Directory to save results, defaults to "./results/{experiment_name}"
    
    Returns:
        Dict: Dictionary mapping aggregation frequencies to lists of result dictionaries (one per seed)
    """
    # Make sure aggregation frequencies are sorted
    aggregation_frequencies = sorted(aggregation_frequencies)
    
    # Structure to hold all results
    all_results = {freq: [] for freq in aggregation_frequencies}
    
    # Create base experiment name once
    base_experiment_name = base_config.experiment_name
    
    # Create parent directory for all experiment results
    from pathlib import Path
    if output_dir is None:
        parent_dir = Path(f"./results/{base_experiment_name}_multi_seed_agg_freq")
    else:
        parent_dir = Path(output_dir)
    parent_dir.mkdir(exist_ok=True, parents=True)
    
    # For each seed
    for idx, seed in enumerate(seeds):
        print(f"Using seed {seed}...")
        
        # For each aggregation frequency
        for freq in aggregation_frequencies:
            print(f"  Training with aggregation frequency {freq}...")
            
            # Create a new config with the seed and aggregation frequency changed
            from dataclasses import asdict
            config_dict = asdict(base_config)
            
            # Save original experiment name before removing from dict
            original_name = config_dict["experiment_name"]
            
            # Remove experiment_name from the dictionary to avoid duplication
            del config_dict["experiment_name"]
            
            # Update the seed and aggregation frequency
            config_dict["seed"] = seed
            config_dict["aggregation_frequency"] = freq
            
            # Modify the output directory to be within the parent directory
            config_dict["output_dir"] = str(parent_dir)
            
            # Create a new experiment name including seed and aggregation frequency
            experiment_name = f"{original_name}_seed{seed}_agg{freq}"
            
            # Create the experiment config
            current_config = FederatedExperimentConfig(
                **config_dict,
                experiment_name=experiment_name
            )
            
            # Create and run the experiment
            runner = FederatedExperimentRunner(current_config)
            runner.setup()
            result = runner.run()
            
            # Optional: Generate visualizations
            runner.visualize_results()
            
            # Add a field to track which aggregation frequency this run is associated with
            result["tracked_aggregation_frequency"] = freq
            result["tracked_seed"] = seed
            
            # Store the result
            all_results[freq].append(result)
    
    # Now create visualizations across all seeds and aggregation frequencies
    visualize_aggregation_freq_results(all_results, base_experiment_name, parent_dir, seeds)
    
    return all_results


def visualize_learning_rate_results(all_results, experiment_name, parent_dir, seeds):
    """
    Visualize results aggregated across multiple seeds for different learning rates.
    
    Args:
        all_results: Dictionary mapping learning rates to lists of result dictionaries
        experiment_name: Name of the experiment for plot titles and filenames
        parent_dir: Parent directory to save visualizations
        seeds: List of seeds used in the experiment
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    # Set style
    sns.set(style="whitegrid")
    
    # Use provided parent directory
    output_dir = parent_dir
    
    # Initialize data structure for plotting
    plot_data = []
    
    # Initialize structure for raw seed-level data
    raw_data = []
    
    # Process results for each learning rate
    for lr, results_list in all_results.items():
        # Extract final model accuracies for this learning rate across all seeds
        accuracies = [result["final_model"]["metrics"]["accuracy"] for result in results_list]
        
        # Calculate statistics
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        median_acc = np.median(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)
        
        # Calculate 95% confidence interval
        n = len(accuracies)
        confidence = 0.95
        degrees_freedom = n - 1
        t_value = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
        ci_margin = t_value * (std_acc / np.sqrt(n))
        ci_lower = mean_acc - ci_margin
        ci_upper = mean_acc + ci_margin
        
        # Store aggregated statistics for plotting
        plot_data.append({
            "Learning Rate": lr,
            "Accuracy": mean_acc,
            "Std": std_acc,
            "Median": median_acc,
            "Min": min_acc,
            "Max": max_acc,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper
        })
        
        # Store raw seed-level data
        for i, acc in enumerate(accuracies):
            # Get the corresponding seed
            seed = results_list[i]["tracked_seed"]
            raw_data.append({
                "Learning Rate": lr,
                "Seed": seed,
                "Accuracy": acc
            })
    
    # Convert to dataframe
    plot_df = pd.DataFrame(plot_data)
    
    # Save this data
    plot_df.to_csv(output_dir / f"learning_rate_aggregated_results.csv", index=False)
    
    # Convert raw data to dataframe and save
    raw_df = pd.DataFrame(raw_data)
    raw_df.to_csv(output_dir / f"learning_rate_raw_results.csv", index=False)
    
    # Create line plot with error bands for different learning rates
    plt.figure(figsize=(14, 8))
    
    # Sort by learning rate
    plot_df = plot_df.sort_values("Learning Rate")
    
    # Plot the line with confidence interval
    plt.errorbar(
        plot_df["Learning Rate"], 
        plot_df["Accuracy"],
        yerr=[plot_df["Accuracy"] - plot_df["CI_Lower"], plot_df["CI_Upper"] - plot_df["Accuracy"]],
        marker='o', 
        capsize=5,
        linestyle='-'
    )
    
    # Add labels and title
    plt.title(f"Model Performance by Learning Rate - {experiment_name}")
    plt.xlabel("Learning Rate")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.xscale('log')  # Log scale is often better for learning rates
    
    # Save figure
    plt.savefig(output_dir / f"learning_rate_performance.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"learning_rate_performance.pdf", bbox_inches='tight')
    
    # Create a violin plot
    plt.figure(figsize=(14, 8))
    
    # Create violin plot
    sns.violinplot(x="Learning Rate", y="Accuracy", data=raw_df, inner="points")
    
    # Add mean markers
    sns.pointplot(
        x="Learning Rate", 
        y="Accuracy", 
        data=raw_df, 
        estimator=np.mean, 
        color="r",
        markers="d", 
        scale=0.7,
        ci=None
    )
    
    # Add labels and title
    plt.title(f"Model Performance Distribution by Learning Rate - {experiment_name}")
    plt.xlabel("Learning Rate")
    plt.ylabel("Test Accuracy")
    
    # Save figure
    plt.savefig(output_dir / f"learning_rate_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"learning_rate_distribution.pdf", bbox_inches='tight')
    
    # Return the plot dataframe for additional analysis if needed
    return plot_df


def visualize_aggregation_freq_results(all_results, experiment_name, parent_dir, seeds):
    """
    Visualize results aggregated across multiple seeds for different aggregation frequencies.
    
    Args:
        all_results: Dictionary mapping aggregation frequencies to lists of result dictionaries
        experiment_name: Name of the experiment for plot titles and filenames
        parent_dir: Parent directory to save visualizations
        seeds: List of seeds used in the experiment
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    
    # Set style
    sns.set(style="whitegrid")
    
    # Use provided parent directory
    output_dir = parent_dir
    
    # Initialize data structure for plotting
    plot_data = []
    
    # Initialize structure for raw seed-level data
    raw_data = []
    
    # Extract training round data for learning curves
    round_data = []
    
    # Process results for each aggregation frequency
    for freq, results_list in all_results.items():
        # Extract final model accuracies for this frequency across all seeds
        accuracies = [result["final_model"]["metrics"]["accuracy"] for result in results_list]
        
        # Extract round-by-round data for learning curves
        for result_idx, result in enumerate(results_list):
            seed = result["tracked_seed"]
            
            # Extract round information if available
            if "rounds" in result:
                for round_info in result["rounds"]:
                    round_num = round_info.get("round", 0)
                    
                    # Try to get accuracy from centralized evaluation
                    accuracy = None
                    if "centralized" in round_info and "evaluate" in round_info["centralized"]:
                        accuracy = round_info["centralized"]["evaluate"].get("accuracy")
                    
                    # If centralized evaluation not available, try distributed
                    if accuracy is None and "distributed" in round_info and "evaluate" in round_info["distributed"]:
                        accuracy = round_info["distributed"]["evaluate"].get("accuracy", 0.0)
                    
                    if accuracy is not None:
                        round_data.append({
                            "Aggregation Frequency": freq,
                            "Round": round_num,
                            "Accuracy": accuracy,
                            "Seed": seed
                        })
        
        # Calculate statistics
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        median_acc = np.median(accuracies)
        min_acc = np.min(accuracies)
        max_acc = np.max(accuracies)
        
        # Calculate 95% confidence interval
        n = len(accuracies)
        confidence = 0.95
        degrees_freedom = n - 1
        t_value = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
        ci_margin = t_value * (std_acc / np.sqrt(n))
        ci_lower = mean_acc - ci_margin
        ci_upper = mean_acc + ci_margin
        
        # Store aggregated statistics for plotting
        plot_data.append({
            "Aggregation Frequency": freq,
            "Accuracy": mean_acc,
            "Std": std_acc,
            "Median": median_acc,
            "Min": min_acc,
            "Max": max_acc,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper
        })
        
        # Store raw seed-level data
        for i, acc in enumerate(accuracies):
            # Get the corresponding seed
            seed = results_list[i]["tracked_seed"]
            raw_data.append({
                "Aggregation Frequency": freq,
                "Seed": seed,
                "Accuracy": acc
            })
    
    # Convert to dataframes
    plot_df = pd.DataFrame(plot_data)
    raw_df = pd.DataFrame(raw_data)
    round_df = pd.DataFrame(round_data)
    
    # Save this data
    plot_df.to_csv(output_dir / f"aggregation_freq_aggregated_results.csv", index=False)
    raw_df.to_csv(output_dir / f"aggregation_freq_raw_results.csv", index=False)
    round_df.to_csv(output_dir / f"aggregation_freq_round_data.csv", index=False)
    
    # Create line plot with error bands for different aggregation frequencies
    plt.figure(figsize=(14, 8))
    
    # Sort by aggregation frequency
    plot_df = plot_df.sort_values("Aggregation Frequency")
    
    # Plot the line with confidence interval
    plt.errorbar(
        plot_df["Aggregation Frequency"], 
        plot_df["Accuracy"],
        yerr=[plot_df["Accuracy"] - plot_df["CI_Lower"], plot_df["CI_Upper"] - plot_df["Accuracy"]],
        marker='o', 
        capsize=5,
        linestyle='-'
    )
    
    # Add labels and title
    plt.title(f"Model Performance by Aggregation Frequency - {experiment_name}")
    plt.xlabel("Aggregation Frequency (rounds)")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.xticks(plot_df["Aggregation Frequency"])  # Ensure all frequencies are shown
    
    # Save figure
    plt.savefig(output_dir / f"aggregation_freq_performance.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"aggregation_freq_performance.pdf", bbox_inches='tight')
    
    # Create a violin plot
    plt.figure(figsize=(14, 8))
    
    # Create violin plot
    sns.violinplot(x="Aggregation Frequency", y="Accuracy", data=raw_df, inner="points")
    
    # Add mean markers
    sns.pointplot(
        x="Aggregation Frequency", 
        y="Accuracy", 
        data=raw_df, 
        estimator=np.mean, 
        color="r",
        markers="d", 
        scale=0.7,
        ci=None
    )
    
    # Add labels and title
    plt.title(f"Model Performance Distribution by Aggregation Frequency - {experiment_name}")
    plt.xlabel("Aggregation Frequency (rounds)")
    plt.ylabel("Test Accuracy")
    
    # Save figure
    plt.savefig(output_dir / f"aggregation_freq_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"aggregation_freq_distribution.pdf", bbox_inches='tight')
    
    # Create learning curves for different aggregation frequencies
    if not round_df.empty:
        plt.figure(figsize=(14, 8))
        
        # For each aggregation frequency, plot the average learning curve across seeds
        for freq in sorted(round_df["Aggregation Frequency"].unique()):
            freq_data = round_df[round_df["Aggregation Frequency"] == freq]
            
            # Group by round and calculate mean and std
            round_stats = freq_data.groupby("Round")["Accuracy"].agg(["mean", "std"]).reset_index()
            
            # Plot
            plt.errorbar(
                round_stats["Round"],
                round_stats["mean"],
                yerr=round_stats["std"],
                label=f"Freq={freq}",
                marker='o',
                capsize=3
            )
        
        plt.title(f"Learning Curves by Aggregation Frequency - {experiment_name}")
        plt.xlabel("Round")
        plt.ylabel("Test Accuracy")
        plt.grid(True)
        plt.legend()
        
        # Save figure
        plt.savefig(output_dir / f"aggregation_freq_learning_curves.png", dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f"aggregation_freq_learning_curves.pdf", bbox_inches='tight')
    
    # Return the plot dataframe for additional analysis if needed
    return plot_df


class FlowerRayClient(flwr.client.NumPyClient):
    """Flower client for federated learning with file-based data"""

    def __init__(
        self,
        cid: int,
        partition_dir: Path,
        model_generator: Callable[[], nn.Module],
        data_dir: Path
    ) -> None:
        """Initialize the client with its unique id and the folder to load data from.

        Parameters
        ----------
            cid (int): Unique client id for a client used to map it to its data partition
            partition_dir (Path): The directory containing data for each client/client id
            model_generator (Callable[[], Module]): The model generator function
            data_dir (Path): Directory containing the raw data files
        """
        self.cid = cid
        logger.info(f"Initializing client {self.cid}")
        self.partition_dir = partition_dir
        self.device = str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model_generator = model_generator
        self.properties: dict[str, Scalar] = {
            "tensor_type": "numpy.ndarray",
            "partition": str(self.partition_dir),
            "cid": self.cid
        }
        self.data_dir = data_dir

    def set_parameters(self, parameters: NDArrays) -> nn.Module:
        """Load weights inside the network.

        Parameters
        ----------
            parameters (NDArrays): set of weights to be loaded.

        Returns
        -------
            [Module]: Network with new set of weights.
        """
        net = self.model_generator()
        return set_model_parameters(net, parameters)

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return weights from a given model.

        If no model is passed, then a local model is created.
        This can be used to initialise a model in the server.
        The config param is not used but is mandatory in Flower.

        Parameters
        ----------
            config (dict[int, Scalar]): dictionary containing configuration info.

        Returns
        -------
            NDArrays: weights from the model.
        """
        net = self.model_generator()
        return get_model_parameters(net)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict]:
        """Receive and train a model on the local client data.

        Parameters
        ----------
            parameters (NDArrays): Pytorch model parameters
            config (dict[str, Scalar]): dictionary describing the training parameters

        Returns
        -------
            tuple[NDArrays, int, dict]: Returns the updated model, the size of the local
                dataset and other metrics
        """
        # Only create model right before training/testing to lower memory usage when idle
        net = self.set_parameters(parameters)
        net.to(self.device)

        train_loader: DataLoader = self._create_data_loader(config, name="train")
        train_loss = self._train(net, train_loader=train_loader, config=config)
        return get_model_parameters(net), len(train_loader.dataset), {"train_loss": train_loss}

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict]:
        """Receive and test a model on the local client data.

        Parameters
        ----------
            parameters (NDArrays): Pytorch model parameters
            config (dict[str, Scalar]): dictionary describing the testing parameters

        Returns
        -------
            tuple[float, int, dict]: Returns the loss accumulated during testing, the
                size of the local dataset and other metrics such as accuracy
        """
        net = self.set_parameters(parameters)
        net.to(self.device)

        test_loader: DataLoader = self._create_data_loader(config, name="test")
        loss, accuracy = self._test(net, test_loader=test_loader, config=config)
        return loss, len(test_loader.dataset), {"local_accuracy": accuracy}

    def _create_data_loader(self, config: dict[str, Scalar], name: str) -> DataLoader:
        """Create the data loader using the specified config parameters.

        Parameters
        ----------
            config (dict[str, Scalar]): dictionary containing dataloader and dataset parameters
            name (str): Load the training or testing set for the client

        Returns
        -------
            DataLoader: A pytorch dataloader iterable for training/testing
        """
        batch_size = int(config["batch_size"])
        num_workers = int(config.get("num_workers", 0))
        dataset = self._load_dataset(name)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=(name == "train"),
        )

    def _load_dataset(self, name: str) -> Dataset:
        """Load dataset for client from mapping file.
        
        Parameters
        ----------
            name (str): 'train' or 'test' to specify which dataset to load
            
        Returns
        -------
            Dataset: The loaded dataset
        """
        full_file: Path = self.partition_dir / str(self.cid)
        
        # Determine dataset type from partition_dir
        if "femnist" in str(self.partition_dir).lower():
            # FEMNIST-specific dataset loading
            # Make sure data_dir points to the femnist/data subdirectory
            femnist_data_dir = self.data_dir / "femnist" / "data"
            return load_femnist_dataset(
                mapping=full_file,
                name=name,
                data_dir=femnist_data_dir,
            )
        elif "cifar10" in str(self.partition_dir).lower():
            # CIFAR10-specific dataset loading using the centralized function
            return load_cifar10_dataset(
                mapping=full_file,
                name=name,
                data_dir=self.data_dir,
            )
        else:
            raise ValueError("Unsupported dataset. Only 'femnist' and 'cifar10' are supported.")

    def _train(
        self, net: nn.Module, train_loader: DataLoader, config: dict[str, Scalar]
    ) -> float:
        """Train the network on local data.
        
        Parameters
        ----------
            net (nn.Module): The neural network to train
            train_loader (DataLoader): DataLoader with training data
            config (dict[str, Scalar]): Configuration parameters
            
        Returns
        -------
            float: The training loss
        """
        # Determine dataset type from partition_dir
        if "femnist" in str(self.partition_dir).lower():
            # FEMNIST-specific training
            return train_femnist(
                net=net,
                train_loader=train_loader,
                epochs=int(config["local_epochs"]),
                device=self.device,
                optimizer=torch.optim.AdamW(
                    net.parameters(),
                    lr=float(config["learning_rate"]),
                    weight_decay=float(config["weight_decay"]),
                ),
                criterion=torch.nn.CrossEntropyLoss(),
                max_batches=int(config.get("max_batches")) if config.get("max_batches") is not None else None,
            )
        elif "cifar10" in str(self.partition_dir).lower():
            # CIFAR10-specific training
            net.train()
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(
                net.parameters(),
                lr=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
            )
            
            epochs = int(config["local_epochs"])
            total_loss = 0.0
            num_batches = 0
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                batch_count = 0
                
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Track loss
                    epoch_loss += loss.item()
                    batch_count += 1
                    
                    # Check if we should limit batches
                    max_batches = config.get("max_batches")
                    if max_batches is not None and batch_count >= int(max_batches):
                        break
                
                # Calculate average epoch loss
                if batch_count > 0:
                    avg_epoch_loss = epoch_loss / batch_count
                    total_loss += avg_epoch_loss
                    num_batches += 1
                    logger.info(f"CIFAR10 Client {self.cid} - Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            
            # Return average loss across all epochs
            if num_batches > 0:
                return total_loss / num_batches
            else:
                return 0.0
        else:
            raise ValueError("Unsupported dataset. Only 'femnist' and 'cifar10' are supported.")

    def _test(
        self, net: nn.Module, test_loader: DataLoader, config: dict[str, Scalar]
    ) -> tuple[float, float]:
        """Test the network on local data.
        
        Parameters
        ----------
            net (nn.Module): The neural network to test
            test_loader (DataLoader): DataLoader with test data
            config (dict[str, Scalar]): Configuration parameters
            
        Returns
        -------
            tuple[float, float]: The test loss and accuracy
        """
        # Determine dataset type from partition_dir
        if "femnist" in str(self.partition_dir).lower():
            # FEMNIST-specific testing
            return test_femnist(
                net=net,
                test_loader=test_loader,
                device=self.device,
                criterion=torch.nn.CrossEntropyLoss(),
                max_batches=int(config.get("max_batches")) if config.get("max_batches") is not None else None,
            )
        elif "cifar10" in str(self.partition_dir).lower():
            # CIFAR10-specific testing
            net.eval()
            criterion = torch.nn.CrossEntropyLoss()
            
            total_loss = 0.0
            correct = 0
            total = 0
            batch_count = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Track metrics
                    total_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    batch_count += 1
                    
                    # Check if we should limit batches
                    max_batches = config.get("max_batches")
                    if max_batches is not None and batch_count >= int(max_batches):
                        break
            
            # Calculate average loss and accuracy
            if total > 0:
                avg_loss = total_loss / total
                accuracy = correct / total
                logger.info(f"CIFAR10 Client {self.cid} - Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                return avg_loss, accuracy
            else:
                logger.warning(f"CIFAR10 Client {self.cid} - Empty test dataset")
                return 0.0, 0.0
        else:
            raise ValueError("Unsupported dataset. Only 'femnist' and 'cifar10' are supported.")

    def get_properties(self, config: dict[str, Scalar]) -> dict[str, Scalar]:
        """Return properties for this client.

        Parameters
        ----------
            config (dict[str, Scalar]): Options to be used for selecting specific properties.

        Returns
        -------
            dict[str, Scalar]: Returned properties.
        """
        return self.properties

    def get_train_set_size(self) -> int:
        """Return the client train set size.

        Returns
        -------
            int: train set size of the client.
        """
        return len(self._load_dataset("train"))

    def get_test_set_size(self) -> int:
        """Return the client test set size.

        Returns
        -------
            int: test set size of the client.
        """
        return len(self._load_dataset("test"))


def get_flower_client_generator(
    model_generator: Callable[[], nn.Module],
    partition_dir: Path,
    data_dir: Path,
    mapping_fn: Callable[[int], int] | None = None,
) -> Callable[[str], FlowerRayClient]:
    """Create a client generator function for Flower simulation using file-based partitioning.

    Parameters
    ----------
        model_generator (Callable[[], Module]): model generator function.
        partition_dir (Path): directory containing the partition.
        data_dir (Path): directory containing the raw data files.
        mapping_fn (Optional[Callable[[int], int]]): function mapping sorted/filtered ids to real cid.

    Returns
    -------
        Callable[[str], FlowerRayClient]: client generator function.
    """

    def client_fn(cid: str) -> FlowerRayClient:
        """Create a single client instance given the client id `cid`.

        Parameters
        ----------
            cid (str): client id, Flower requires this to be of type str.

        Returns
        -------
            FlowerRayClient: client instance.
        """
        return FlowerRayClient(
            cid=mapping_fn(int(cid)) if mapping_fn is not None else int(cid),
            partition_dir=partition_dir,
            model_generator=model_generator,
            data_dir=data_dir
        )

    return client_fn


# Main execution block
if __name__ == "__main__":
    # Example configuration for FEMNIST - adjusted for file-based partitioning
    femnist_config = FederatedExperimentConfig(
        experiment_name="femnist_ray_client_experiment",
        dataset_name="femnist",
        model_name="cnn",
        num_rounds=10,
        local_epochs=20,
        learning_rate=0.01,
        server_learning_rate=1.0,
        data_distribution="natural",  # Use the 'natural' distribution
        num_clients_per_round=5,
        num_total_clients=20
    )
    
    # Example configuration for CIFAR10
    cifar10_config = FederatedExperimentConfig(
        experiment_name="cifar10_ray_client_experiment",
        dataset_name="cifar10",
        model_name="tiny_resnet",
        num_rounds=10,
        local_epochs=30,
        learning_rate=0.01,
        server_learning_rate=1.0,
        data_distribution="dirichlet",  # Using Dirichlet for non-IID distribution
        non_iid_alpha=0.5,
        num_clients_per_round=5,
        num_total_clients=20
    )
    
    # Choose configuration to use - uncomment the one you want to run
    config_to_use = femnist_config
    # config_to_use = cifar10_config
    
    # Run the experiment with FlowerRayClient
    logger.info(f"Starting federated experiment for {config_to_use.dataset_name} with FlowerRayClient using existing partition files")
    runner = FederatedExperimentRunner(config_to_use)
    runner.setup()
    results = runner.run()
    
    # Visualize results
    runner.visualize_results()
    
    # Uncomment below to run parameter sweeps if needed
    # run_multi_seed_learning_rates_experiment(
    #     config_to_use, 
    #     learning_rates=[0.001, 0.01, 0.1],
    #     seeds=[42, 123, 456, 789, 101]
    # )