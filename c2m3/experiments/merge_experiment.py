import os
import json
import time
import logging
from typing import Dict, List, Union, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
import pandas as pd

from c2m3.match.merger import FrankWolfeSynchronizedMerger
from c2m3.match.permutation_spec import (
    CNNPermutationSpecBuilder,
    AutoPermutationSpecBuilder
)
from c2m3.models.utils import get_model_class
from c2m3.modules.pl_module import MyLightningModule
from c2m3.utils.utils import set_seed
from c2m3.common.femnist_dataset import FEMNIST
from c2m3.match.ties_merging import merge_models_ties
from flwr.server.strategy.aggregate import aggregate as flwr_aggregate
from c2m3.data.partitioners import PartitionerRegistry, DatasetPartitioner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MergeExperimentConfig:
    """Configuration for model merging experiments"""
    # Required parameters (no defaults)
    experiment_name: str
    dataset_name: str  # 'femnist', 'cifar10', 'cifar100', 'shakespeare'
    model_name: str  # 'resnet18', 'cnn', 'lstm'
    
    # Parameters with default values
    seed: int = 42
    data_dir: str = str(Path(__file__).parent.parent / "data")
    
    # Training configuration
    num_models: int = 3  # Number of models to train and merge
    epochs_per_model: Union[int, List[int]] = 10  # Can be a single int or list of different epochs
    batch_size: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    
    # Data distribution
    data_distribution: str = "iid"  # 'iid', 'dirichlet', 'pathological', 'natural'
    non_iid_alpha: float = 0.5  # Dirichlet alpha parameter for non-IID distribution
    classes_per_partition: int = 1  # For pathological partitioning, how many classes per client
    
    # Initialization
    initialization_type: str = "identical"  # 'identical' or 'diverse'
    
    # Merging configuration
    merging_methods: List[str] = None  # ['c2m3', 'fedavg', 'ties']
    c2m3_max_iter: int = 100  # Max iterations for Frank-Wolfe
    c2m3_score_tolerance: float = 1e-6  # Convergence threshold for Frank-Wolfe
    c2m3_init_method: str = "identity"  # Initialization method for permutation matrices
    ties_alpha: float = 0.5  # Interpolation parameter for TIES merging
    
    # Evaluation
    eval_batch_size: int = 128
    
    # Output
    output_dir: str = "./results"
    save_models: bool = False
    save_results: bool = True
    
    # New parameters for federated subset experiment
    num_total_clients: int = 20  # Select 20 clients from the entire pool
    num_clients_per_round: int = 8  # Sample 8 clients per round
    
    def __post_init__(self):
        """Initialize default merging methods if none provided"""
        if self.merging_methods is None:
            self.merging_methods = ["c2m3", "fedavg", "ties"]
        
        # Convert epochs_per_model to list if it's an int
        if isinstance(self.epochs_per_model, int):
            self.epochs_per_model = [self.epochs_per_model] * self.num_models

class ModelTrainer:
    """Class for training individual models"""
    def __init__(self, model, train_loader, test_loader, lr, weight_decay, device="cpu"):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Move model to device
        self.model.to(self.device)
        
        # Metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Compute epoch metrics
        epoch_loss = total_loss / len(self.train_loader.dataset)
        epoch_acc = correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def evaluate(self):
        """Evaluate model on test set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Track metrics
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Compute metrics
        test_loss = total_loss / len(self.test_loader.dataset)
        test_acc = correct / total
        
        self.test_losses.append(test_loss)
        self.test_accuracies.append(test_acc)
        
        return test_loss, test_acc
    
    def train(self, epochs, verbose=True):
        """Train the model for specified epochs"""
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()
            
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies,
            'final_train_acc': self.train_accuracies[-1],
            'final_test_acc': self.test_accuracies[-1]
        }


class ModelMerger:
    """Class for merging models using different strategies"""
    def __init__(self, config, device, train_loaders=None):
        self.config = config
        self.device = device
        self.train_loaders = train_loaders  # Store train_loaders for use in merge methods
        
    def get_state_dicts(self, models):
        """Extract state dictionaries from models"""
        return {f"model_{i}": model.state_dict() for i, model in enumerate(models)}
    
    def merge_c2m3(self, models):
        """Merge models using C2M3 algorithm"""
        logger.info("Merging models with C2M3...")
        state_dicts = self.get_state_dicts(models)
        
        # Get permutation spec for the model architecture
        # Get appropriate permutation spec builder for the model architecture
        if self.config.model_name == "cnn":
            perm_spec_builder = CNNPermutationSpecBuilder()
            perm_spec = perm_spec_builder.create_permutation_spec()
        # elif self.config.model_name == "transformer":
        #     perm_spec_builder = TransformerPermutationSpecBuilder()
        #     perm_spec = perm_spec_builder.create_permutation_spec()
        elif self.config.model_name == "auto":
            # For auto, we would need a reference model and input
            # This is a placeholder - you'll need to provide the actual model and input
            perm_spec_builder = AutoPermutationSpecBuilder()
            ref_model = models[0]  # Use first model as reference
            ref_input = next(iter(self.train_loaders[0]))[0][:1].to(self.device)  # Get a single batch item
            perm_spec = perm_spec_builder.create_permutation_spec(ref_model=ref_model, ref_input=ref_input)
        else:
            raise ValueError(f"Unknown model name for permutation spec: {self.config.model_name}")
        
        # Create C2M3 merger
        merger = FrankWolfeSynchronizedMerger(
            name="c2m3",
            permutation_spec=perm_spec,
            initialization_method=self.config.c2m3_init_method,
            max_iter=self.config.c2m3_max_iter,
            score_tolerance=self.config.c2m3_score_tolerance,
            device=self.device
        )
        
        # We need to access the train_loaders
        if not hasattr(self, 'train_loaders'):
            raise ValueError("Train loaders not available for C2M3 merging. Cannot proceed with model merging.")
        else:
            # Use the train_loaders that were passed during initialization
            logger.info("Using train loaders for C2M3 merging to update batch norm statistics.")
            # Create MyLightningModule instances from state_dicts
            lightning_modules = {}
            network_generator = get_model_class(self.config.model_name)
            
            # Create symbols for models (a, b, c, etc.)
            for i, (model_key, state_dict) in enumerate(state_dicts.items()):
                symbol = chr(97 + i)  # 'a', 'b', 'c', ...
                
                # Create network and set parameters
                net = network_generator()
                net.load_state_dict(state_dict)
                
                # Wrap in MyLightningModule
                # TODO: Refine control flow for num_classes
                num_classes = 62 if self.config.dataset_name == "femnist" else 10  # Default to 10 for MNIST/CIFAR
                mlm = MyLightningModule(net, num_classes=num_classes)
                
                # Store in dictionary with symbol key
                lightning_modules[symbol] = mlm
            
            # Call merger with lightning modules and train loaders
            merged_model, repaired_model, models_permuted_to_universe = merger(lightning_modules, train_loader=self.train_loaders)
            
            # Extract state dict from the merged model
            merged_state_dict = merged_model.model.state_dict()
        
        # Create a new model and load the merged weights
        model_class = get_model_class(self.config.model_name)
        merged_model = model_class()
        merged_model.load_state_dict(merged_state_dict)
        merged_model.to(self.device)
        
        return merged_model
    
    def merge_fedavg(self, models):
        """Merge models using simple averaging (FedAvg)"""
        logger.info("Merging models with FedAvg...")
        
        # Get model parameters as NumPy arrays
        model_params = []
        num_examples = []  # Track the actual number of examples for each model
        
        for i, model in enumerate(models):
            # Extract model weights as state_dict
            state_dict = model.state_dict()
            # Convert to list of numpy arrays (same order as state_dict keys)
            param_list = [param.cpu().numpy() for param in state_dict.values()]
            model_params.append(param_list)
            
            # Get the actual number of examples in this model's dataset
            dataset_size = len(self.train_loaders[i].dataset)
            num_examples.append(dataset_size)
            logger.info(f"Model {i}: {dataset_size} training examples")
        
        # Create the list of tuples format that Flower's aggregate function expects
        results = [(model_params[i], num_examples[i]) for i in range(len(models))]
        
        # Call the aggregation function with the properly formatted input
        aggregated_params = flwr_aggregate(results)
        
        # Create a new model and load the merged weights
        model_class = get_model_class(self.config.model_name)
        merged_model = model_class()
        
        # Set the aggregated parameters to the model
        with torch.no_grad():
            state_dict = merged_model.state_dict()
            # Update state dict with aggregated parameters
            for i, (key, _) in enumerate(state_dict.items()):
                state_dict[key] = torch.tensor(aggregated_params[i])
        
        # Load the state dict and move model to device
        merged_model.load_state_dict(state_dict)
        merged_model.to(self.device)
        
        return merged_model
    
    def merge_ties(self, models):
        """Merge models using TIES algorithm"""
        logger.info("Merging models with TIES...")
        
        # Get TIES alpha parameter or use default of 0.5
        alpha = getattr(self.config, "ties_alpha", 0.5)
        
        # Apply TIES merging
        merged_model = merge_models_ties(models, alpha=alpha)
        merged_model.to(self.device)
        
        return merged_model
    
    def merge(self, models, method):
        """Merge models using specified method"""
        if method == "c2m3":
            return self.merge_c2m3(models)
        elif method == "fedavg":
            return self.merge_fedavg(models)
        elif method == "ties":
            return self.merge_ties(models)
        else:
            raise ValueError(f"Unknown merging method: {method}")


class DataManager:
    """Class for handling dataset loading and partitioning"""
    def __init__(self, config):
        self.config = config
        self.dataset_name = config.dataset_name
        self.data_dir = Path(config.data_dir)
        self.data_distribution = config.data_distribution
        self.num_total_clients = config.num_total_clients
        self.num_clients_per_round = config.num_clients_per_round
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Client IDs selected for this experiment
        self.selected_client_ids = []
        
        # Set random seed for client selection
        # This ensures client selection is reproducible across runs
        self.random_generator = random.Random(config.seed)
        
    def get_dataset(self):
        """Get the specified dataset"""
        if self.dataset_name == "femnist":
            return self._get_femnist()
        elif self.dataset_name == "cifar10":
            return self._get_cifar10()
        elif self.dataset_name == "cifar100":
            return self._get_cifar100()
        elif self.dataset_name == "shakespeare":
            return self._get_shakespeare()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
    def _get_available_client_ids(self, mapping_dir):
        """Get available client IDs from a directory"""
        available_client_ids = []
        if not mapping_dir.exists():
            logger.error(f"Partition directory not found at {mapping_dir}")
            logger.info("Please ensure the client mappings are properly set up.")
            raise FileNotFoundError(f"Partition mapping not found")
            
        for client_path in mapping_dir.iterdir():
            try:
                client_id = int(client_path.name)
                available_client_ids.append(client_id)
            except ValueError:
                # Skip non-integer directory names
                continue
                
        return available_client_ids
    
    def _select_client_ids(self, available_ids, num_clients):
        """Select a subset of client IDs deterministically based on seed"""
        if len(available_ids) >= num_clients:
            # Use our seeded random generator instead of the global one
            return self.random_generator.sample(available_ids, num_clients)
        else:
            logger.warning(f"Requested {num_clients} clients but only {len(available_ids)} available. Using all available clients.")
            return available_ids
    
    def _get_femnist(self):
        """Load FEMNIST dataset based on selected clients"""
        # Define paths for FEMNIST dataset
        data_dir = Path(self.data_dir) / "femnist" / "data"
        mapping_dir = Path(self.data_dir) / "femnist" / "client_data_mappings"
        
        # Create directories if they don't exist
        data_dir.mkdir(exist_ok=True, parents=True)
        mapping_dir.mkdir(exist_ok=True, parents=True)
        
        # Check if dataset is available
        if not (data_dir / "train").exists():
            logger.error(f"FEMNIST dataset files not found in {data_dir}")
            logger.info("Please ensure the FEMNIST dataset is properly downloaded and extracted.")
            logger.info("You can check c2m3/common/client_utils.py for the expected dataset structure.")
            raise FileNotFoundError(f"FEMNIST dataset not found in {data_dir}")
        
        # Determine which partition mapping to use
        if self.data_distribution == "iid":
            partition_dir = mapping_dir / "iid"
        elif self.data_distribution == "dirichlet":
            partition_dir = mapping_dir / f"lda_{self.config.non_iid_alpha}"
        elif self.data_distribution == "natural":
            partition_dir = mapping_dir / "fed_natural"
        else:
            # Default to natural partitioning if others aren't available
            partition_dir = mapping_dir / "fed_natural"
            logger.warning(f"Using 'fed_natural' partitioning as fallback for '{self.data_distribution}'")
        
        # Get available client IDs
        available_client_ids = self._get_available_client_ids(partition_dir)
        
        # First level of sampling: select num_total_clients clients
        self.selected_client_ids = self._select_client_ids(available_client_ids, self.num_total_clients)
        logger.info(f"Selected pool of {len(self.selected_client_ids)} clients from {len(available_client_ids)} available")
        
        # Define transforms
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL Image to tensor
            transforms.Lambda(lambda x: x.reshape(1, 28, 28)),  # Reshape to expected format
            transforms.Normalize((0.5,), (0.5,))  # Normalize
        ])
        
        # We only load the test sets from all selected clients to form our test set
        # Training data will be loaded as needed during partitioning
        test_datasets = []
        
        for client_id in self.selected_client_ids:
            client_mapping = partition_dir / str(client_id)
            
            try:
                # Load test set for this client
                test_set = FEMNIST(
                    mapping=client_mapping,
                    data_dir=data_dir,
                    name="test",
                    transform=transform
                )
                test_datasets.append(test_set)
            except Exception as e:
                logger.warning(f"Failed to load test data for client {client_id}: {e}")
        
        # Combine test datasets using ConcatDataset
        from torch.utils.data import ConcatDataset
        combined_test_set = ConcatDataset(test_datasets)
        
        # Return a placeholder as train_set - actual training partitions will be created during partitioning
        # This is a dummy dataset that will be replaced with actual client partitions
        dummy_train_set = FEMNIST(
            mapping=partition_dir / str(self.selected_client_ids[0]),
            data_dir=data_dir,
            name="train",
            transform=transform
        )
        
        test_size = sum(len(ds) for ds in test_datasets)
        logger.info(f"Loaded FEMNIST test set with {test_size} samples from {len(test_datasets)} clients")
        logger.info(f"Training data will be loaded during partitioning")
        
        return dummy_train_set, combined_test_set
    
    def partition_data(self, train_set, num_partitions):
        """
        Create partitions from client datasets based on num_clients_per_round
        
        This function selects num_clients_per_round clients from the pool of
        num_total_clients clients and loads their training data.
        """
        # For FEMNIST, we use the client partition approach
        if self.dataset_name == "femnist":
            # NOTE train_set is a dummy dataset that will be replaced with actual client partitions
            # Validate parameters
            if num_partitions > self.num_clients_per_round:
                logger.warning(f"Requested {num_partitions} partitions but only {self.num_clients_per_round} clients per round. Using all available clients.")
                num_partitions = self.num_clients_per_round
            
            # Define paths for FEMNIST dataset
            data_dir = Path(self.data_dir) / "femnist" / "data"
            mapping_dir = Path(self.data_dir) / "femnist" / "client_data_mappings"
            
            # Determine which partition mapping to use
            if self.data_distribution == "iid":
                partition_dir = mapping_dir / "iid"
            elif self.data_distribution == "dirichlet":
                partition_dir = mapping_dir / f"lda_{self.config.non_iid_alpha}"
            elif self.data_distribution == "natural":
                partition_dir = mapping_dir / "fed_natural"
            else:
                partition_dir = mapping_dir / "fed_natural"
            
            # Second level of sampling: select num_clients_per_round from the pool
            clients_for_round = self._select_client_ids(self.selected_client_ids, self.num_clients_per_round)
            clients_for_training = clients_for_round[:num_partitions]
            
            logger.info(f"Selected {len(clients_for_training)} clients for this training round")
            
            # Define transforms
            transform = transforms.Compose([
                transforms.ToTensor(),  # Convert PIL Image to tensor
                transforms.Lambda(lambda x: x.reshape(1, 28, 28)),  # Reshape to expected format
                transforms.Normalize((0.5,), (0.5,))  # Normalize
            ])
            
            # Load training data for selected clients
            train_datasets = []
            
            for client_id in clients_for_training:
                client_mapping = partition_dir / str(client_id)
                
                try:
                    # Load training data for this client
                    train_set = FEMNIST(
                        mapping=client_mapping,
                        data_dir=data_dir,
                        name="train",
                        transform=transform
                    )
                    train_datasets.append(train_set)
                    logger.info(f"Loaded training data for client {client_id}: {len(train_set)} samples")
                except Exception as e:
                    logger.warning(f"Failed to load training data for client {client_id}: {e}")
            
            # Ensure we have enough partitions
            if len(train_datasets) < num_partitions:
                logger.error(f"Could only load {len(train_datasets)} client datasets, but {num_partitions} were requested")
                raise ValueError(f"Not enough valid client datasets available")
            
            return train_datasets
        else:
            # For other datasets, use traditional partitioning methods
            # Get appropriate partitioner class for this dataset and distribution
            try:
                # Try to get dataset-specific partitioner first
                partitioner_class = PartitionerRegistry.get(self.dataset_name, self.data_distribution)
                logger.info(f"Using {self.dataset_name}-specific {self.data_distribution} partitioner")
            except ValueError:
                # Fall back to default partitioner
                partitioner_class = PartitionerRegistry.get("default", self.data_distribution)
                logger.info(f"Using default {self.data_distribution} partitioner for {self.dataset_name}")
            
            # Create partitioner instance with appropriate parameters
            if self.data_distribution == "dirichlet":
                partitioner = partitioner_class(alpha=self.config.non_iid_alpha)
            elif self.data_distribution == "pathological":
                partitioner = partitioner_class(classes_per_partition=self.config.classes_per_partition)
            elif self.data_distribution in ["natural", "by_writer", "by_character"]:
                partitioner = partitioner_class(data_dir=self.data_dir)
            else:
                partitioner = partitioner_class(data_dir=self.data_dir)
            
            # Partition the dataset
            return partitioner.partition(train_set, num_partitions)
        
    # Keep existing methods for other datasets
    def _get_cifar10(self):
        """Load CIFAR-10 dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_set = datasets.CIFAR10(root=self.data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=self.data_dir, train=False, download=True, transform=transform)
        
        return train_set, test_set
    
    def _get_cifar100(self):
        """Load CIFAR-100 dataset"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_set = datasets.CIFAR100(root=self.data_dir, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR100(root=self.data_dir, train=False, download=True, transform=transform)
        
        return train_set, test_set
    
    def _get_shakespeare(self):
        """Load Shakespeare dataset"""
        logger.info("Loading Shakespeare dataset...")
        
        try:
            from torchtext.datasets import CharacterLevel
            from torch.utils.data import random_split
            
            # Create a character-level language modeling dataset
            dataset = CharacterLevel('shakespeare', root=self.data_dir)
            
            # Split into train and test
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_set, test_set = random_split(dataset, [train_size, test_size])
            
            logger.info(f"Loaded Shakespeare dataset: {len(train_set)} training samples, {len(test_set)} test samples")
            
            return train_set, test_set
        except ImportError:
            logger.error("torchtext is required for Shakespeare dataset. Please install with: pip install torchtext")
            raise

    def create_dataloaders(self, partitions, test_set, batch_size, eval_batch_size):
        """Create DataLoaders for train partitions and test set"""
        train_loaders = [
            DataLoader(partition, batch_size=batch_size, shuffle=True) 
            for partition in partitions
        ]
        
        test_loader = DataLoader(test_set, batch_size=eval_batch_size, shuffle=False)
        
        return train_loaders, test_loader


class MergeExperimentRunner:
    """Main class for running model merging experiments"""
    def __init__(self, config: MergeExperimentConfig):
        self.config = config
        self.experiment_name = config.experiment_name
        
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
            "models": {},
            "merged_models": {}
        }
        
        self.data_manager = None
        self.models = []
        self.train_loaders = []
        self.test_loader = None
        self.model_merger = None
        
    def setup(self):
        """Setup experiment - load datasets, create models, etc."""
        logger.info(f"Setting up experiment '{self.experiment_name}'...")
        
        # Setup data
        self.data_manager = DataManager(self.config)
        train_set, test_set = self.data_manager.get_dataset()
        
        # Partition training data
        partitions = self.data_manager.partition_data(train_set, self.config.num_models)
        
        # Create dataloaders
        self.train_loaders, self.test_loader = self.data_manager.create_dataloaders(
            partitions, test_set, self.config.batch_size, self.config.eval_batch_size
        )
        
        # Initialize models
        model_class = get_model_class(self.config.model_name)
        
        if self.config.initialization_type == "identical":
            # Create a base model
            base_model = model_class()
            state_dict = base_model.state_dict()
            
            # Initialize all models with same weights
            self.models = [model_class() for _ in range(self.config.num_models)]
            for model in self.models:
                model.load_state_dict(state_dict)
        else:  # diverse initialization
            # Initialize each model randomly
            self.models = [model_class() for _ in range(self.config.num_models)]
        
        logger.info("Experiment setup complete.")
        
    def run(self):
        """Run the experiment - train models, merge them, evaluate"""
        logger.info(f"Running experiment '{self.experiment_name}'...")
        start_time = time.time()
        
        # Train individual models
        self.results["models"] = {}
        
        for i, (model, train_loader, epochs) in enumerate(zip(
            self.models, self.train_loaders, self.config.epochs_per_model
        )):
            logger.info(f"Training model {i+1}/{self.config.num_models} for {epochs} epochs...")
            
            trainer = ModelTrainer(
                model, train_loader, self.test_loader,
                self.config.learning_rate, self.config.weight_decay,
                self.device
            )
            
            # Train and record results
            model_results = trainer.train(epochs)
            
            # Save model results
            self.results["models"][f"model_{i}"] = {
                "epochs": epochs,
                **model_results
            }
            
            # Save model if requested
            if self.config.save_models:
                torch.save(model.state_dict(), self.output_dir / f"model_{i}.pt")
        
        # Merge models using different methods
        self.results["merged_models"] = {}
        
        # Initialize model merger with train_loaders
        self.model_merger = ModelMerger(self.config, self.device, self.train_loaders)
        
        for method in self.config.merging_methods:
            logger.info(f"Merging models using {method}...")
            
            # Merge models
            merged_model = self.model_merger.merge(self.models, method)
            
            # Evaluate merged model
            trainer = ModelTrainer(
                merged_model, None, self.test_loader,
                self.config.learning_rate, self.config.weight_decay,
                self.device
            )
            
            _, test_acc = trainer.evaluate()
            
            # Save results
            self.results["merged_models"][method] = {
                "test_accuracy": test_acc
            }
            
            # Save merged model if requested
            if self.config.save_models:
                torch.save(merged_model.state_dict(), self.output_dir / f"merged_model_{method}.pt")
        
        # Record total runtime
        self.results["runtime"] = time.time() - start_time
        
        # Save results
        if self.config.save_results:
            with open(self.output_dir / "results.json", "w") as f:
                json.dump(self.results, f, indent=4)
        
        logger.info(f"Experiment '{self.experiment_name}' completed in {self.results['runtime']:.2f} seconds.")
        
        return self.results
        
    def visualize_results(self):
        """Generate plots for the experiment results"""
        if not self.results:
            logger.warning("No results to visualize. Run the experiment first.")
            return
        
        # Create plots directory
        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Set Seaborn style
        sns.set_style("whitegrid")
        
        # Plot 1: Training curves for individual models
        plt.figure(figsize=(12, 8))
        
        # Plot training accuracy
        plt.subplot(2, 2, 1)
        for model_id, model_results in self.results["models"].items():
            epochs = model_results["epochs"]
            plt.plot(range(1, epochs+1), model_results["train_accuracies"], label=f"{model_id}")
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        
        # Plot training loss
        plt.subplot(2, 2, 2)
        for model_id, model_results in self.results["models"].items():
            epochs = model_results["epochs"]
            plt.plot(range(1, epochs+1), model_results["train_losses"], label=f"{model_id}")
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        # Plot test accuracy
        plt.subplot(2, 2, 3)
        for model_id, model_results in self.results["models"].items():
            epochs = model_results["epochs"]
            plt.plot(range(1, epochs+1), model_results["test_accuracies"], label=f"{model_id}")
        plt.title("Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        
        # Plot test loss
        plt.subplot(2, 2, 4)
        for model_id, model_results in self.results["models"].items():
            epochs = model_results["epochs"]
            plt.plot(range(1, epochs+1), model_results["test_losses"], label=f"{model_id}")
        plt.title("Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / "training_curves.png", dpi=300)
        
        # Plot 2: Comparison of merging methods (Test Accuracy)
        plt.figure(figsize=(10, 6))
        
        # Extract test accuracies for merged models
        methods = list(self.results["merged_models"].keys())
        accuracies = [results["test_accuracy"] for results in self.results["merged_models"].values()]
        
        # Plot comparison bar chart for test accuracy
        plt.bar(methods, accuracies, color=sns.color_palette("muted", len(methods)))
        plt.title("Merging Methods Comparison - Test Accuracy")
        plt.xlabel("Merging Method")
        plt.ylabel("Test Accuracy")
        plt.ylim(0, 1)
        
        # Add individual model accuracies as reference lines
        for i, model_results in enumerate(self.results["models"].values()):
            plt.axhline(y=model_results["final_test_acc"], linestyle='--', 
                        color=f'C{i}', alpha=0.7, label=f"Model {i} Accuracy")
        
        # Add average of individual models as another reference line
        avg_acc = sum(model_results["final_test_acc"] for model_results in self.results["models"].values()) / len(self.results["models"])
        plt.axhline(y=avg_acc, linestyle='-', color='black', alpha=0.7, label="Avg Individual Accuracy")
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "merging_accuracy_comparison.png", dpi=300)
        
        # Plot 3: Comparison of merging methods with individual models (Test Accuracy)
        plt.figure(figsize=(12, 6))
        
        # Combine data for all models and merging methods
        all_methods = []
        all_accuracies = []
        
        # Add individual models
        for i, model_results in enumerate(self.results["models"].values()):
            all_methods.append(f"Model {i}")
            all_accuracies.append(model_results["final_test_acc"])
        
        # Add merged models
        for method, results in self.results["merged_models"].items():
            all_methods.append(method)
            all_accuracies.append(results["test_accuracy"])
        
        # Add average of individual models
        all_methods.append("Avg Individual")
        all_accuracies.append(avg_acc)
        
        # Plot comprehensive comparison
        sns.barplot(x=all_methods, y=all_accuracies)
        plt.title("Comprehensive Accuracy Comparison")
        plt.xlabel("Model / Merging Method")
        plt.ylabel("Test Accuracy")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "comprehensive_accuracy_comparison.png", dpi=300)
        
        # If we have loss data for merged models, plot that too
        if any("test_loss" in results for results in self.results["merged_models"].values()):
            plt.figure(figsize=(10, 6))
            
            # Extract test losses for merged models
            methods = list(self.results["merged_models"].keys())
            losses = [results.get("test_loss", 0) for results in self.results["merged_models"].values()]
            
            # Plot comparison bar chart for test loss
            plt.bar(methods, losses, color=sns.color_palette("muted", len(methods)))
            plt.title("Merging Methods Comparison - Test Loss")
            plt.xlabel("Merging Method")
            plt.ylabel("Test Loss")
            
            # Add individual model losses as reference lines
            for i, model_results in enumerate(self.results["models"].values()):
                plt.axhline(y=model_results["test_losses"][-1], linestyle='--', 
                            color=f'C{i}', alpha=0.7, label=f"Model {i} Loss")
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "merging_loss_comparison.png", dpi=300)
        
        logger.info(f"Results visualized and saved to {plots_dir}")


def run_parameter_sweep(base_config: MergeExperimentConfig, parameter_grid: Dict[str, List[Any]]):
    """Run experiments with all combinations of parameters in the grid"""
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
        config = MergeExperimentConfig(**config_dict)
        runner = MergeExperimentRunner(config)
        
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


def run_single_parameter_comparison(base_config, parameter_name, parameter_values, output_dir="../outputs/merger_results"):
    """
    Run experiments with different values for a single parameter and plot the results.
    
    Args:
        base_config: Base configuration for experiments
        parameter_name: Name of the parameter to vary
        parameter_values: List of values to use for the parameter
        output_dir: Directory to save the comparison results
    """
    # Create parameter grid that only varies one parameter
    parameter_grid = {parameter_name: parameter_values}
    
    # Run parameter sweep
    results = run_parameter_sweep(base_config, parameter_grid)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Extract and organize results for plotting
    comparison_data = []
    
    for result in results:
        # Get parameter value used in this experiment
        param_value = result["config"][parameter_name]
        
        # Get metrics for all merging methods
        for method, metrics in result["merged_models"].items():
            comparison_data.append({
                parameter_name: param_value,
                "method": method,
                "test_accuracy": metrics["test_accuracy"]
            })
        
        # Also add individual model results
        for model_id, model_result in result["models"].items():
            comparison_data.append({
                parameter_name: param_value,
                "method": f"{model_id}",
                "test_accuracy": model_result["final_test_acc"]
            })
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(comparison_data)
    
    # Create line plot showing how the parameter affects test accuracy
    plt.figure(figsize=(12, 8))
    # Use parameter_name for both the column name and the x-axis label
    # First make sure the parameter name exists as a column in the dataframe
    if parameter_name not in df.columns:
        df["param_value"] = df[parameter_name] if parameter_name in df.columns else None
        sns.lineplot(data=df, x="param_value", y="test_accuracy", hue="method", marker="o")
    else:
        sns.lineplot(data=df, x=parameter_name, y="test_accuracy", hue="method", marker="o")
    
    plt.title(f"Effect of {parameter_name} on Test Accuracy")
    plt.xlabel(parameter_name)
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path / f"{parameter_name}_comparison.png", dpi=300)
    
    # Save the data
    df.to_csv(output_path / f"{parameter_name}_comparison.csv", index=False)
    
    return df


def create_comparison_matrix(base_config, param1_name, param1_values, param2_name, param2_values):
    """Create a matrix of experiments comparing two parameters"""
    # Create a grid with both parameters
    parameter_grid = {
        param1_name: param1_values,
        param2_name: param2_values
    }
    
    # Run all combinations
    results = run_parameter_sweep(base_config, parameter_grid)
    
    # Create a DataFrame for heatmap visualization
    heatmap_data = {}
    
    # Use the best merging method (e.g., c2m3)
    method = "c2m3"
    
    for result in results:
        # Get parameter values used in this experiment
        param1 = result["config"][param1_name]
        param2 = result["config"][param2_name]
        
        # Get accuracy for the selected merging method
        accuracy = result["merged_models"][method]["test_accuracy"]
        
        # Store in nested dictionary
        if param1 not in heatmap_data:
            heatmap_data[param1] = {}
        heatmap_data[param1][param2] = accuracy
    
    # Convert to DataFrame
    df = pd.DataFrame(heatmap_data).T
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="viridis", fmt=".3f")
    plt.title(f"Test Accuracy using {method} with varying {param1_name} and {param2_name}")
    plt.xlabel(param2_name)
    plt.ylabel(param1_name)
    plt.tight_layout()
    plt.savefig(f"comparison_{param1_name}_{param2_name}.png", dpi=300)
    
    return df


# Example usage
if __name__ == "__main__":
    # Create a base configuration
    base_config = MergeExperimentConfig(
        experiment_name="federated_subset_experiment",
        dataset_name="femnist",
        model_name="cnn",
        num_models=4,
        epochs_per_model=[30, 30, 30, 30],  # Reduced epochs for faster testing
        initialization_type="identical",
        data_distribution="natural",
        merging_methods=["c2m3", "fedavg"],
        num_total_clients=20,  # Select 20 clients from the entire pool
        num_clients_per_round=8,  # Sample 8 clients per round
        learning_rate=0.01  # Fixed learning rate for this experiment
    )
    
    # Create experiment runner
    runner = MergeExperimentRunner(base_config)
    
    # Setup and run experiment
    runner.setup()
    results = runner.run()
    runner.visualize_results()
    
    # Example with parameter sweep
    # Run comparison for client selection parameters
    """
    # Uncomment to run parameter sweep
    clients_per_round_values = [2, 4, 8, 16]
    comparison_df = run_single_parameter_comparison(
        base_config,
        parameter_name="num_clients_per_round",
        parameter_values=clients_per_round_values
    )
    """