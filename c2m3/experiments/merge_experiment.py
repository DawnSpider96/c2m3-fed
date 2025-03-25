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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
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
from c2m3.match.ties_merging import merge_models_ties
from flwr.server.strategy.aggregate import aggregate as flwr_aggregate
from c2m3.data.partitioners import PartitionerRegistry, DatasetPartitioner
from c2m3.common.client_utils import (
    load_femnist_dataset, 
    get_network_generator_cnn as get_network_generator,
    train_femnist,
    test_femnist,
    save_history,
    get_model_parameters,
    # calculate_accuracy, 
    # calculate_f1, 
    # calculate_recall, 
    # calculate_balanced_accuracy,
    set_model_parameters
)

from c2m3.experiments.corrected_partition_data import partition_data as corrected_partition_data
from c2m3.experiments.corrected_partition_data import FEMNISTFromSamples

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    
    # Early stopping configuration
    early_stopping: bool = False  # Whether to use early stopping
    early_stopping_patience: int = 5  # Number of epochs with no improvement to wait before stopping
    early_stopping_min_delta: float = 0.001  # Minimum change to qualify as improvement
    early_stopping_metric: str = "accuracy"  # Metric to monitor: 'accuracy', 'f1', or 'recall'
    
    # Data distribution
    data_distribution: str = "iid"  # 'iid', 'dirichlet', 'pathological', 'natural'
    non_iid_alpha: float = 0.5  # Dirichlet alpha parameter for non-IID distribution
    classes_per_partition: int = 1  # For pathological partitioning, how many classes per client
    samples_per_partition: Optional[int] = 4000  # Number of samples per partition (for IID FEMNIST), None for equal division
    
    # Initialization
    initialization_type: str = "identical"  # 'identical' or 'diverse'
    
    # Merging configuration
    merging_methods: List[str] = None  # ['c2m3', 'fedavg', 'ties', 'median']
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
    
    def __post_init__(self):
        """Initialize default merging methods if none provided"""
        if self.merging_methods is None:
            self.merging_methods = ["c2m3", "fedavg", "ties", "median"]
        
        # Convert epochs_per_model to list if it's an int
        if isinstance(self.epochs_per_model, int):
            self.epochs_per_model = [self.epochs_per_model] * self.num_models

class ModelTrainer:
    """Class for training individual models"""
    def __init__(self, model, train_loader, test_loader, lr, weight_decay, device="cpu", dataset_name="cifar10", 
                val_loader=None, patience=5, min_delta=0.001):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.dataset_name = dataset_name
        
        # Early stopping parameters
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_metric = 0
        self.wait_count = 0
        
        # Move model to device
        self.model.to(self.device)
        
        # Metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_balanced_accuracies = []
        
        # Initialize test metrics - these will only be populated at the end
        self.test_losses = []
        self.test_accuracies = []
        
        # For metrics calculation
        try:
            from sklearn.metrics import f1_score, recall_score
            self.has_sklearn = True
        except ImportError:
            logger.warning("scikit-learn not available; F1 and recall metrics will not be calculated")
            self.has_sklearn = False
    
    def train_epoch(self):
        """Train for one epoch"""
        if self.dataset_name == "femnist":
            # Make sure model is in training mode
            self.model.train()
            
            # Use the FEMNIST-specific training function
            epoch_loss = train_femnist(
                net=self.model,
                train_loader=self.train_loader,
                epochs=1,  # Just one epoch at a time
                device=self.device,
                optimizer=self.optimizer,
                criterion=self.criterion,
                max_batches=None  # Use all batches
            )
            
            # Since train_femnist doesn't return accuracy, we need to evaluate to get it
            self.model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in self.train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            # Check for division by zero
            if total == 0:
                logger.warning(f"Empty dataset detected during training evaluation. Setting accuracy to 0.")
                epoch_acc = 0.0
            else:
                epoch_acc = correct / total
        else:
            # Use the generic training approach
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
            # Check for division by zero
            if total == 0:
                logger.warning(f"Empty dataset detected during training. Setting loss and accuracy to 0.")
                epoch_loss = 0.0
                epoch_acc = 0.0
            else:
                epoch_loss = total_loss / total
                epoch_acc = correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def calculate_metrics(self, predictions, targets):
        """Calculate balanced accuracy"""
        if not self.has_sklearn:
            return 0.0
        
        from sklearn.metrics import balanced_accuracy_score
        
        # Convert tensors to numpy arrays
        preds_np = predictions.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Calculate balanced accuracy
        try:
            # Get unique classes in the ground truth
            unique_classes = np.unique(targets_np)
            
            # Create a mask for predictions that match classes in ground truth
            valid_indices = np.isin(preds_np, unique_classes)
            
            # If we have no valid predictions, return 0
            if not np.any(valid_indices):
                logger.warning("No valid predictions found (all predicted classes missing from ground truth)")
                return 0.0
                
            # Filter predictions and targets to only include valid indices
            filtered_preds = preds_np[valid_indices]
            filtered_targets = targets_np[valid_indices]
            
            # Calculate balanced accuracy on filtered data
            balanced_acc = balanced_accuracy_score(filtered_targets, filtered_preds)
            return balanced_acc
        except Exception as e:
            logger.warning(f"Error calculating balanced accuracy: {e}")
            return 0.0
    
    def evaluate_validation(self):
        """Evaluate model on validation set"""
        if not self.val_loader:
            logger.warning("No validation loader provided. Skipping validation evaluation.")
            return 0.0, 0.0, 0.0
        
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Track metrics
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Save predictions and targets for balanced accuracy
                all_predictions.append(predicted)
                all_targets.append(targets)
        
        # Compute metrics
        if total == 0 or len(self.val_loader.dataset) == 0:
            logger.warning(f"Empty validation dataset detected. Setting metrics to 0.")
            val_loss = 0.0
            val_acc = 0.0
            val_balanced_acc = 0.0
        else:
            val_loss = total_loss / len(self.val_loader.dataset)
            val_acc = correct / total
            
            # Calculate balanced accuracy
            if self.has_sklearn and all_predictions and all_targets:
                all_predictions = torch.cat(all_predictions)
                all_targets = torch.cat(all_targets)
                val_balanced_acc = self.calculate_metrics(all_predictions, all_targets)
            else:
                val_balanced_acc = 0.0
        
        return val_loss, val_acc, val_balanced_acc
    
    def evaluate(self):
        """Evaluate model on test set"""
        if self.dataset_name == "femnist":
            # Use the FEMNIST-specific testing function
            test_loss, test_acc = test_femnist(
                net=self.model,
                test_loader=self.test_loader,
                device=self.device,
                criterion=self.criterion,
                max_batches=None  # Use all batches
            )
        else:
            # Use the generic testing approach
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
            # Check for division by zero
            if total == 0 or len(self.test_loader.dataset) == 0:
                logger.warning(f"Empty test dataset detected. Setting loss and accuracy to 0.")
                test_loss = 0.0
                test_acc = 0.0
            else:
                test_loss = total_loss / len(self.test_loader.dataset)
                test_acc = correct / total
        
        return test_loss, test_acc
    
    def should_stop_early(self, val_metric):
        """Check if training should be stopped early based on validation metric"""
        if val_metric > self.best_val_metric + self.min_delta:
            # Improvement detected
            self.best_val_metric = val_metric
            self.wait_count = 0
            return False
        
        # No significant improvement
        self.wait_count += 1
        return self.wait_count >= self.patience
    
    def train(self, epochs, early_stopping=False, verbose=True, early_stopping_metric='accuracy'):
        """Train the model for specified epochs with optional early stopping"""
        early_stopped = False
        best_model_state = None
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            
            # Evaluate on validation set if available
            if self.val_loader:
                val_loss, val_acc, val_balanced_acc = self.evaluate_validation()
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)
                self.val_balanced_accuracies.append(val_balanced_acc)
                
                # Determine early stopping metric
                val_metric = val_balanced_acc  # Use balanced accuracy
                
                # Save the best model so far
                if val_metric > self.best_val_metric:
                    best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                val_loss, val_acc, val_balanced_acc = 0.0, 0.0, 0.0
                val_metric = 0.0
            
            # Only log test metrics without actually running evaluation during epochs
            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                if self.val_loader:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}")
                else:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Check for early stopping if enabled and validation loader is available
            if early_stopping and self.val_loader and self.should_stop_early(val_metric):
                early_stopped = True
                logger.info(f"Early stopping triggered at epoch {epoch+1}. Best validation balanced accuracy: {self.best_val_metric:.4f}")
                break
        
        # Restore best model if we did early stopping and have a best model
        if early_stopped and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("Restored model to best validation performance")
            
            # Re-evaluate best model on validation
            if self.val_loader:
                val_loss, val_acc, val_balanced_acc = self.evaluate_validation()
        
        # Only evaluate on test set at the end of training
        test_loss, test_acc = self.evaluate()
        logger.info(f"Final test performance - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
        
        # Prepare return dictionary with training history
        history = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_losses': [test_loss],  # Only store the final test loss
            'test_accuracies': [test_acc],  # Only store the final test accuracy
            'final_train_acc': self.train_accuracies[-1],
            'final_test_acc': test_acc,
            'epochs_trained': epoch + 1,
            'early_stopped': early_stopped
        }
        
        # Add validation metrics if available
        if self.val_loader:
            history.update({
                'val_losses': self.val_losses,
                'val_accuracies': self.val_accuracies,
                'val_balanced_accuracies': self.val_balanced_accuracies,
                'final_val_acc': self.val_accuracies[-1],
                'final_val_balanced_acc': self.val_balanced_accuracies[-1],
                'best_val_metric': self.best_val_metric
            })
        
        return history


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
        """
        Merge models using C2M3 algorithm
        
        For FEMNIST models, this ensures:
        1. The CNN permutation spec is used
        2. The correct number of classes (62) is set
        3. Batch norm statistics are properly updated using the train_loaders
        """
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
        if not hasattr(self, 'train_loaders') or self.train_loaders is None:
            raise ValueError("Train loaders not available for C2M3 merging. Cannot proceed with model merging.")
        else:
            # Use the train_loaders that were passed during initialization
            logger.info("Using train loaders for C2M3 merging to update batch norm statistics.")
            # Create MyLightningModule instances from state_dicts
            lightning_modules = {}
            
            # Get the correct model generator
            if self.config.dataset_name == "femnist":
                # For FEMNIST, use the dedicated network generator
                network_generator = get_network_generator()
                logger.info("Using FEMNIST-specific network generator for merging")
            else:
                # For other datasets, use the generic model class
                network_generator = get_model_class(self.config.model_name)
            
            # Create symbols for models (a, b, c, etc.)
            for i, (model_key, state_dict) in enumerate(state_dicts.items()):
                symbol = chr(97 + i)  # 'a', 'b', 'c', ...
                
                # Create network and set parameters
                net = network_generator()
                net.load_state_dict(state_dict)
                
                # Wrap in MyLightningModule
                if self.config.dataset_name == "femnist":
                    # FEMNIST has 62 classes (letters and digits)
                    num_classes = 62
                else:
                    # Default to 10 for MNIST/CIFAR or determine from model's output size
                    num_classes = getattr(net, 'num_classes', 10)
                
                mlm = MyLightningModule(net, num_classes=num_classes)
                
                # Store in dictionary with symbol key
                lightning_modules[symbol] = mlm
            
            # Call merger with lightning modules and train loaders
            merged_model, repaired_model, models_permuted_to_universe = merger(lightning_modules, train_loader=self.train_loaders)
            
            # Extract state dict from the merged model
            merged_state_dict = merged_model.model.state_dict()
        
        # Create a new model and load the merged weights
        if self.config.dataset_name == "femnist":
            # For FEMNIST, use the dedicated network generator
            merged_model = get_network_generator()()
        else:
            # For other datasets, use the generic model class
            model_class = get_model_class(self.config.model_name)
            merged_model = model_class()
        
        merged_model.load_state_dict(merged_state_dict)
        merged_model.to(self.device)
        
        return merged_model
    
    def merge_fedavg(self, models):
        """
        Merge models using simple averaging (FedAvg)
        
        For FEMNIST, this uses the actual number of examples in each client's dataset
        to perform a weighted average of the model parameters.
        """
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
        if self.config.dataset_name == "femnist":
            # For FEMNIST, use the dedicated network generator
            merged_model = get_network_generator()()
        else:
            # For other datasets, use the generic model class
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
        """
        Merge models using TIES algorithm
        
        TIES (Transfusion of Information Elements at Singularities) performs
        a spatial-wise weighted average of models to address permutation ambiguities
        without additional optimization.
        """
        logger.info("Merging models with TIES...")
        
        # Get TIES alpha parameter or use default of 0.5
        alpha = getattr(self.config, "ties_alpha", 0.5)
        
        # Apply TIES merging
        merged_model = merge_models_ties(models, alpha=alpha)
        merged_model.to(self.device)
        
        return merged_model
    
    def merge_simple_avg(self, models):
        """
        Merge models using simple averaging (equal weights)
        
        This differs from fedavg in that it:
        1. Uses equal weights regardless of dataset size
        2. Directly averages the parameters without the Flower library
        3. Is a simpler, more lightweight implementation
        """
        logger.info("Merging models with simple averaging...")
        
        # Create a new model to hold the merged parameters
        if self.config.dataset_name == "femnist":
            # For FEMNIST, use the dedicated network generator
            from c2m3.common.client_utils import get_network_generator_cnn
            merged_model = get_network_generator_cnn()()
        else:
            # For other datasets, use the generic model class
            model_class = get_model_class(self.config.model_name)
            merged_model = model_class()
        
        # Get the state dict of the merged model
        merged_state_dict = merged_model.state_dict()
        
        # For each parameter in the model
        for key in merged_state_dict.keys():
            # Stack the same parameter from all models
            # Then take the mean along the first dimension (the model dimension)
            stacked_params = torch.stack([model.state_dict()[key] for model in models])
            merged_state_dict[key] = torch.mean(stacked_params, dim=0)
        
        merged_model.load_state_dict(merged_state_dict)
        merged_model.to(self.device)
        
        return merged_model
    
    def merge_median(self, models):
        """
        Merge models using element-wise median aggregation
        
        This method computes the median value for each parameter across all models.
        It's a robust aggregation method that's less sensitive to outliers than mean-based
        methods like FedAvg.
        """
        logger.info("Merging models with median-based aggregation...")
        
        # Create a new model to hold the merged parameters
        if self.config.dataset_name == "femnist":
            # For FEMNIST, use the dedicated network generator
            merged_model = get_network_generator()()
        else:
            # For other datasets, use the generic model class
            model_class = get_model_class(self.config.model_name)
            merged_model = model_class()
        
        # Get the state dict of the merged model
        merged_state_dict = merged_model.state_dict()
        
        # For each parameter in the model
        for key in merged_state_dict.keys():
            # Stack the same parameter from all models
            stacked_params = torch.stack([model.state_dict()[key].cpu() for model in models])
            
            # Compute median along the first dimension (the model dimension)
            # PyTorch doesn't have a direct median function that keeps gradients,
            # so we use the numpy median and convert back to torch
            if stacked_params.dtype == torch.bool:
                # For boolean tensors, we use majority voting
                merged_state_dict[key] = torch.mode(stacked_params, dim=0).values
            else:
                # For numerical tensors, we use median
                median_values = torch.from_numpy(
                    np.median(stacked_params.detach().numpy(), axis=0)
                ).to(stacked_params.dtype)
                merged_state_dict[key] = median_values
        
        merged_model.load_state_dict(merged_state_dict)
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
        elif method == "simple_avg":
            return self.merge_simple_avg(models)
        elif method == "median":
            return self.merge_median(models)
        else:
            raise ValueError(f"Unknown merging method: {method}")


class DataManager:
    """Class for handling dataset loading and partitioning"""
    def __init__(self, config):
        self.config = config
        self.dataset_name = config.dataset_name
        self.data_dir = Path(config.data_dir)
        self.data_distribution = config.data_distribution
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # All available client IDs
        self.available_client_ids = []
        
        # Set random seed for client selection
        # This ensures client selection is reproducible across runs
        self.random_generator = random.Random(config.seed)
        
    def _select_client_ids(self, available_ids, num_clients):
        """Select a subset of client IDs deterministically based on seed"""
        if len(available_ids) >= num_clients:
            # Use our seeded random generator instead of the global one
            return self.random_generator.sample(available_ids, num_clients)
        else:
            logger.warning(f"Requested {num_clients} clients but only {len(available_ids)} available. Using all available clients.")
            return available_ids
    
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
        """
        Get available client IDs from a mapping directory
        
        Args:
            mapping_dir: Path to the mapping directory
            
        Returns:
            List of available client IDs
        """
        client_ids = []
        
        # Check if mapping directory exists
        if not mapping_dir.exists():
            logger.warning(f"Mapping directory {mapping_dir} does not exist")
            return client_ids
        
        # Get available client IDs
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
        # Use the corrected implementation from corrected_partition_data.py
        from c2m3.experiments.corrected_partition_data import partition_data as corrected_partition_data
        
        return corrected_partition_data(
            config=self.config,
            data_refs=data_refs,
            num_partitions=num_partitions,
            available_client_ids=self.available_client_ids
        )
    
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
    
    def _get_cifar100(self):
        """
        Return references to CIFAR-100 data instead of actual datasets
        
        Returns:
            Dictionary with dataset metadata
        """
        return {
            "dataset_name": "cifar100",
            "data_dir": self.data_dir,
            "transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        }
    
    def _get_shakespeare(self):
        """
        Return references to Shakespeare data instead of actual datasets
        
        Returns:
            Dictionary with dataset metadata
        """
        try:
            # Just verify that the required package is available
            import torchtext
            
            return {
                "dataset_name": "shakespeare",
                "data_dir": self.data_dir
            }
        except ImportError:
            logger.error("torchtext is required for Shakespeare dataset. Please install with: pip install torchtext")
            raise

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
        else:
            # Generic approach for other datasets
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
                drop_last=False
            )
        
        # Log the number of batches in each loader
        for i, loader in enumerate(train_loaders):
            logger.info(f"Train loader {i}: {len(loader)} batches, {len(loader.dataset)} samples")
        
        for i, loader in enumerate(val_loaders):
            logger.info(f"Validation loader {i}: {len(loader)} batches, {len(loader.dataset)} samples")
            
        logger.info(f"Test loader: {len(test_loader)} batches, {len(test_loader.dataset)} samples")
        
        return train_loaders, val_loaders, test_loader


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
        self.val_loaders = []
        self.test_loader = None
        self.model_merger = None
        
    def setup(self):
        """Setup experiment - load datasets, create models, etc."""
        logger.info(f"Setting up experiment '{self.experiment_name}'...")
        
        # Setup data
        self.data_manager = DataManager(self.config)
        
        # Get data references for the specified dataset
        data_refs = self.data_manager.get_dataset()
        
        # Partition data, getting train, validation, and test partitions
        # Our corrected implementation returns a single test set that all models will use
        train_partitions, val_partitions, test_set = self.data_manager.partition_data(data_refs, self.config.num_models)
        
        # Create dataloaders
        self.train_loaders, self.val_loaders, self.test_loader = self.data_manager.create_dataloaders(
            train_partitions, val_partitions, test_set, self.config.batch_size, self.config.eval_batch_size
        )
        
        # Initialize models
        if self.config.dataset_name == "femnist":
            logger.info("Using FEMNIST-specific network generator")
            # For FEMNIST, we use the get_network_generator function which ensures consistent initialization
            network_generator = get_network_generator()
            
            if self.config.initialization_type == "identical":
                # For identical initialization, we generate models from the same generator
                # which ensures they have identical starting weights
                self.models = [network_generator() for _ in range(self.config.num_models)]
                logger.info(f"Created {len(self.models)} identical FEMNIST models")
            else:
                # For diverse initialization, we could use separate random initializations
                # But we still use the generator for consistency with other code
                self.models = [network_generator() for _ in range(self.config.num_models)]
                logger.info(f"Created {len(self.models)} FEMNIST models with diverse initialization")
        else:
            # For other datasets, use the standard model_class approach
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
        
        # Add early stopping configuration to results
        self.results["early_stopping"] = {
            "enabled": hasattr(self.config, "early_stopping") and self.config.early_stopping,
            "patience": getattr(self.config, "early_stopping_patience", 5),
            "min_delta": getattr(self.config, "early_stopping_min_delta", 0.001),
            "metric": getattr(self.config, "early_stopping_metric", "accuracy")
        }
        
        # Extract early stopping parameters
        early_stopping = self.results["early_stopping"]["enabled"]
        early_stopping_patience = self.results["early_stopping"]["patience"]
        early_stopping_min_delta = self.results["early_stopping"]["min_delta"]
        early_stopping_metric = self.results["early_stopping"]["metric"]
        
        logger.info(f"Early stopping: {'enabled' if early_stopping else 'disabled'}, "
                   f"patience={early_stopping_patience}, min_delta={early_stopping_min_delta}, "
                   f"metric={early_stopping_metric}")
        
        for i, (model, train_loader, val_loader, epochs) in enumerate(zip(
            self.models, self.train_loaders, self.val_loaders, self.config.epochs_per_model
        )):
            logger.info(f"Training model {i+1}/{self.config.num_models} for {epochs} epochs...")
            
            trainer = ModelTrainer(
                model, train_loader, self.test_loader,
                self.config.learning_rate, self.config.weight_decay,
                self.device, self.config.dataset_name,
                val_loader=val_loader,
                patience=early_stopping_patience,
                min_delta=early_stopping_min_delta
            )
            
            # Train and record results with early stopping if enabled
            model_results = trainer.train(
                epochs, 
                early_stopping=early_stopping,
                early_stopping_metric=early_stopping_metric
            )
            
            # Save model results
            self.results["models"][f"model_{i}"] = {
                "epochs": epochs,
                "actual_epochs": model_results.get("epochs_trained", epochs),
                "early_stopped": model_results.get("early_stopped", False),
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
                self.device, self.config.dataset_name
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
        """Visualize experiment results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        from collections import defaultdict
        
        # Set style
        sns.set(style="whitegrid")
        plt.figure(figsize=(18, 12))
        
        # Plot individual model metrics
        # Create 3x2 grid of subplots for different metrics
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 15))
        fig.suptitle(f"Model Training Results - {self.experiment_name}", fontsize=16)
        
        # Define metrics to plot
        plot_metrics = [
            {"title": "Training Loss", "key": "train_losses", "color": "blue"},
            {"title": "Validation Loss", "key": "val_losses", "color": "green"},
            {"title": "Test Loss", "key": "test_losses", "color": "red"},
            {"title": "Training Accuracy", "key": "train_accuracies", "color": "blue"},
            {"title": "Validation Accuracy", "key": "val_accuracies", "color": "green"},
            {"title": "Test Accuracy", "key": "test_accuracies", "color": "red"},
        ]
        
        # Additional validation metrics for F1 score and recall in a separate figure
        additional_metrics = [
            {"title": "Validation F1 Score", "key": "val_f1_scores", "color": "purple"},
            {"title": "Validation Recall", "key": "val_recalls", "color": "orange"}
        ]
        
        # Plot training and evaluation metrics
        for i, metric in enumerate(plot_metrics):
            ax = axes[i//2, i%2]
            ax.set_title(metric["title"])
            ax.set_xlabel("Epoch")
            
            if "loss" in metric["key"].lower():
                ax.set_ylabel("Loss")
            else:
                ax.set_ylabel("Accuracy")
            
            # Plot for each model
            for model_idx, model_key in enumerate(sorted(self.results["models"].keys())):
                model_data = self.results["models"][model_key]
                
                # Check if the metric exists for this model
                if metric["key"] in model_data:
                    # Get actual number of epochs trained (for early stopping)
                    epochs_trained = model_data.get("epochs_trained", len(model_data[metric["key"]]))
                    x_values = list(range(1, epochs_trained + 1))
                    
                    # Plot the metric
                    ax.plot(
                        x_values, 
                        model_data[metric["key"]][:epochs_trained], 
                        label=f"Model {model_idx+1}",
                        color=sns.color_palette()[model_idx],
                        alpha=0.8,
                        marker='o' if len(x_values) < 15 else None,
                        markersize=4
                    )
                    
                    # Mark early stopping point if applicable
                    if model_data.get("early_stopped", False) and "val" in metric["key"] and epochs_trained < model_data["epochs"]:
                        ax.axvline(x=epochs_trained, color=sns.color_palette()[model_idx], linestyle='--', alpha=0.5)
                        ax.scatter([epochs_trained], [model_data[metric["key"]][epochs_trained-1]], 
                                 color=sns.color_palette()[model_idx], s=100, marker='X', 
                                 label=f"Early stop - Model {model_idx+1}" if i == 4 else None)  # Only add label in validation accuracy
            
            # Set legend for the first plot only to avoid clutter
            if i == 0:
                ax.legend(loc="best")
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make room for suptitle
        
        # Save the figure if output_dir is set
        if self.config.save_results:
            plt.savefig(self.output_dir / "training_metrics.png", dpi=300, bbox_inches="tight")
        
        # Plot additional validation metrics if available
        if any("val_f1_scores" in self.results["models"][model_key] for model_key in self.results["models"]):
            fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
            fig2.suptitle(f"Validation Metrics - {self.experiment_name}", fontsize=16)
            
            for i, metric in enumerate(additional_metrics):
                ax = axes2[i]
                ax.set_title(metric["title"])
                ax.set_xlabel("Epoch")
                ax.set_ylabel(metric["title"].split()[-2])  # "F1" or "Recall"
                
                for model_idx, model_key in enumerate(sorted(self.results["models"].keys())):
                    model_data = self.results["models"][model_key]
                    
                    if metric["key"] in model_data:
                        epochs_trained = model_data.get("epochs_trained", len(model_data[metric["key"]]))
                        x_values = list(range(1, epochs_trained + 1))
                        
                        ax.plot(
                            x_values, 
                            model_data[metric["key"]][:epochs_trained], 
                            label=f"Model {model_idx+1}",
                            color=sns.color_palette()[model_idx],
                            alpha=0.8,
                            marker='o' if len(x_values) < 15 else None,
                            markersize=4
                        )
                        
                        # Mark early stopping point if applicable
                        if model_data.get("early_stopped", False) and self.results.get("early_stopping", {}).get("metric", "") == metric["key"].split("_")[-1][:-1]:
                            ax.axvline(x=epochs_trained, color=sns.color_palette()[model_idx], linestyle='--', alpha=0.5)
                            ax.scatter([epochs_trained], [model_data[metric["key"]][epochs_trained-1]], 
                                     color=sns.color_palette()[model_idx], s=100, marker='X')
                
                if i == 0:
                    ax.legend(loc="best")
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            if self.config.save_results:
                plt.savefig(self.output_dir / "validation_metrics.png", dpi=300, bbox_inches="tight")
        
        # Plot merged model comparison
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        # Extract test accuracies for merged models
        merged_methods = []
        merged_accuracies = []
        
        for method, data in self.results["merged_models"].items():
            merged_methods.append(method)
            merged_accuracies.append(data["test_accuracy"])
        
        # Calculate average final test accuracy for individual models
        individual_test_accs = [model_data["final_test_acc"] for model_data in self.results["models"].values()]
        avg_individual_acc = sum(individual_test_accs) / len(individual_test_accs)
        best_individual_acc = max(individual_test_accs)
        
        # Add individual model references
        merged_methods.extend(["Avg Individual", "Best Individual"])
        merged_accuracies.extend([avg_individual_acc, best_individual_acc])
        
        # Plotting
        bar_colors = sns.color_palette("viridis", len(merged_methods))
        plt.figure(figsize=(10, 6))
        bars = plt.bar(merged_methods, merged_accuracies, color=bar_colors)
        
        # Add value labels on top of bars
        for bar, acc in zip(bars, merged_accuracies):
            plt.text(
                bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.01, 
                f'{acc:.4f}', 
                ha='center', va='bottom', 
                fontweight='bold'
            )
        
        plt.title(f"Comparison of Model Merging Methods - {self.experiment_name}", fontsize=14)
        plt.ylabel("Test Accuracy", fontsize=12)
        plt.ylim(top=max(merged_accuracies) * 1.1)  # Add some space for the value labels
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        if self.config.save_results:
            plt.savefig(self.output_dir / "merging_comparison.png", dpi=300, bbox_inches="tight")
        
        # Display all plots
        plt.show()
        
        # Print summary statistics
        logger.info("=== Experiment Summary ===")
        logger.info(f"Runtime: {self.results['runtime']:.2f} seconds")
        logger.info("Individual Models:")
        
        for model_key, model_data in self.results["models"].items():
            early_stopped = model_data.get("early_stopped", False)
            early_stop_info = f" (early stopped at epoch {model_data.get('epochs_trained', '?')})" if early_stopped else ""
            logger.info(f"  {model_key}: Train Acc={model_data['final_train_acc']:.4f}, Test Acc={model_data['final_test_acc']:.4f}{early_stop_info}")
            
            # Add validation metrics if available
            if "final_val_acc" in model_data:
                logger.info(f"    Validation: Acc={model_data['final_val_acc']:.4f}, F1={model_data.get('final_val_f1', 0):.4f}, Recall={model_data.get('final_val_recall', 0):.4f}")
        
        logger.info("Merged Models:")
        for method, data in self.results["merged_models"].items():
            logger.info(f"  {method}: Test Acc={data['test_accuracy']:.4f}")
        
        logger.info(f"Best Individual Model: Test Acc={best_individual_acc:.4f}")
        logger.info(f"Average Individual Model: Test Acc={avg_individual_acc:.4f}")
        
        # Return summary for programmatic use
        return {
            "merged_models": {method: data["test_accuracy"] for method, data in self.results["merged_models"].items()},
            "best_individual": best_individual_acc,
            "avg_individual": avg_individual_acc,
            "early_stopping": self.results.get("early_stopping", {"enabled": False})
        }


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
    # Possible values for parameter_name:
    # - "learning_rate": typically [0.001, 0.01, 0.1]
    # - "batch_size": typically [16, 32, 64, 128]
    # - "hidden_size": typically [64, 128, 256, 512]
    # - "num_epochs": typically [5, 10, 20, 50]
    # - "weight_decay": typically [0.0, 0.0001, 0.001, 0.01]
    # - "dropout_rate": typically [0.0, 0.1, 0.2, 0.5]
    # - "merge_method": typically ["average", "weighted", "task_arithmetic"]
    # - "merge_weight": typically [0.1, 0.3, 0.5, 0.7, 0.9]
    
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


def run_increasing_epochs_experiment(base_config, epoch_values):
    """
    Run experiments with same model initialization but increasing epochs.
    
    Args:
        base_config: Base configuration for experiments
        epoch_values: List of epoch values to train models for (e.g., [5, 10, 20, 30])
    """
    # Ensure identical initialization
    base_config.initialization_type = "identical"
    
    # Create the experiment runner with base config
    runner = MergeExperimentRunner(base_config)
    runner.setup()
    
    # Save the initial state of all models
    initial_states = [model.state_dict().copy() for model in runner.models]
    
    results = []
    
    # For each epoch setting
    for epochs in epoch_values:
        print(f"Training with {epochs} epochs...")
        
        # Update config with new epoch value
        base_config.epochs_per_model = [epochs] * base_config.num_models
        runner.config = base_config
        
        # Reset models to initial state
        for model, initial_state in zip(runner.models, initial_states):
            model.load_state_dict(initial_state)
        
        # Run the experiment
        result = runner.run()
        results.append(result)
        
        # Optional: save visualizations for this epoch setting
        runner.visualize_results()
    
    return results


def run_multi_seed_epochs_experiment(base_config, epoch_values, seeds=[42, 123, 456, 789, 101]):
    """
    Run experiments with multiple seeds, tracking epoch values for analysis.
    
    Args:
        base_config: Base configuration for experiments
        epoch_values: Epoch values to use for training models
        seeds: List of random seeds to use for each run
    """
    # Structure to hold all results
    all_results = {epoch: [] for epoch in epoch_values}
    
    # Create base experiment name once
    base_experiment_name = base_config.experiment_name
    
    # For each epoch setting 
    for epoch_value in epoch_values:
        print(f"Processing epoch value {epoch_value}...")
        
        # For each seed
        for idx, seed in enumerate(seeds):
            print(f"  Using seed {seed}...")
            
            # Create a seed suffix for result identification
            seed_suffix = f"_seed{seed}_epochs{epoch_value}"
            
            # Create a new config with the seed changed and epoch value updated
            from dataclasses import asdict
            config_dict = asdict(base_config)
            
            # Save original experiment name before removing from dict
            original_name = config_dict["experiment_name"]
            
            # Remove experiment_name from the dictionary to avoid duplication
            del config_dict["experiment_name"]
            
            # Update the seed and epochs_per_model
            config_dict["seed"] = seed
            config_dict["epochs_per_model"] = [epoch_value] * base_config.num_models
            
            # Create the new config, explicitly providing the original name plus suffixes
            current_config = MergeExperimentConfig(
                **config_dict,
                experiment_name=original_name + seed_suffix
            )
            
            # Create a new runner for this configuration
            runner = MergeExperimentRunner(current_config)
            runner.setup()
            
            # Run the experiment (without visualization)
            result = runner.run()
            
            # Add a field to track which epoch value this run is associated with
            result["tracked_epoch_value"] = epoch_value
            
            # Store the result
            all_results[epoch_value].append(result)
    
    # Now create visualizations across all seeds and epochs
    visualize_aggregated_results(all_results, base_experiment_name)
    
    return all_results

def visualize_aggregated_results(all_results, experiment_name):
    """
    Visualize results aggregated across multiple seeds for each epoch value.
    
    Args:
        all_results: Dictionary mapping epoch values to lists of result dictionaries
        experiment_name: Name of the experiment for plot titles and filenames
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from scipy import stats
    
    # Set style
    sns.set(style="whitegrid")
    
    # Create output directory
    output_dir = Path(f"./results/{experiment_name}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract methods from first result (assuming all use the same methods)
    first_epoch = list(all_results.keys())[0]
    first_result = all_results[first_epoch][0]
    merging_methods = list(first_result["merged_models"].keys())
    
    # Initialize data structure for plotting
    plot_data = []
    
    # Initialize structure for raw seed-level data
    raw_data = []
    
    # Process results for each epoch
    for epoch, results_list in all_results.items():
        # For each merging method, calculate statistics across seeds
        for method in merging_methods:
            # Extract accuracies for this method across all seeds
            accuracies = [result["merged_models"][method]["test_accuracy"] for result in results_list]
            
            # Calculate all statistics
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            median_acc = np.median(accuracies)
            min_acc = np.min(accuracies)
            max_acc = np.max(accuracies)
            
            # Calculate 95% confidence interval using t-distribution
            # (more appropriate than normal distribution for small sample sizes)
            n = len(accuracies)
            confidence = 0.95
            degrees_freedom = n - 1
            t_value = stats.t.ppf((1 + confidence) / 2, degrees_freedom)
            ci_margin = t_value * (std_acc / np.sqrt(n))
            ci_lower = mean_acc - ci_margin
            ci_upper = mean_acc + ci_margin
            
            # Store aggregated statistics for plotting
            plot_data.append({
                "Epochs": epoch,
                "Method": method,
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
                # Get the corresponding seed from the result
                seed = results_list[i]["config"]["seed"]
                raw_data.append({
                    "Epochs": epoch,
                    "Method": method,
                    "Seed": seed,
                    "Accuracy": acc,
                    "Type": "Merged"
                })
        
        # Also calculate for individual models (average and best)
        ind_avg_accs = []
        ind_best_accs = []
        
        for result in results_list:
            # Extract best and average individual model accuracies
            model_accuracies = [model_data.get("final_test_acc", 0) 
                               for model_data in result["models"].values()]
            
            seed = result["config"]["seed"]
            
            if model_accuracies:
                avg_acc = sum(model_accuracies) / len(model_accuracies)
                best_acc = max(model_accuracies)
                
                ind_avg_accs.append(avg_acc)
                ind_best_accs.append(best_acc)
                
                # Save raw data for individual models too
                raw_data.append({
                    "Epochs": epoch,
                    "Method": "Avg Individual",
                    "Seed": seed,
                    "Accuracy": avg_acc,
                    "Type": "Individual"
                })
                
                raw_data.append({
                    "Epochs": epoch,
                    "Method": "Best Individual",
                    "Seed": seed,
                    "Accuracy": best_acc,
                    "Type": "Individual"
                })
        
        # Calculate stats for average individual models
        avg_ind_mean = np.mean(ind_avg_accs)
        avg_ind_std = np.std(ind_avg_accs)
        avg_ind_median = np.median(ind_avg_accs)
        avg_ind_min = np.min(ind_avg_accs)
        avg_ind_max = np.max(ind_avg_accs)
        
        # Calculate 95% CI for average individual
        n = len(ind_avg_accs)
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        ci_margin = t_value * (avg_ind_std / np.sqrt(n))
        avg_ind_ci_lower = avg_ind_mean - ci_margin
        avg_ind_ci_upper = avg_ind_mean + ci_margin
        
        plot_data.append({
            "Epochs": epoch,
            "Method": "Avg Individual",
            "Accuracy": avg_ind_mean,
            "Std": avg_ind_std,
            "Median": avg_ind_median,
            "Min": avg_ind_min,
            "Max": avg_ind_max,
            "CI_Lower": avg_ind_ci_lower,
            "CI_Upper": avg_ind_ci_upper
        })
        
        # Calculate stats for best individual models
        best_ind_mean = np.mean(ind_best_accs)
        best_ind_std = np.std(ind_best_accs)
        best_ind_median = np.median(ind_best_accs)
        best_ind_min = np.min(ind_best_accs)
        best_ind_max = np.max(ind_best_accs)
        
        # Calculate 95% CI for best individual
        n = len(ind_best_accs)
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        ci_margin = t_value * (best_ind_std / np.sqrt(n))
        best_ind_ci_lower = best_ind_mean - ci_margin
        best_ind_ci_upper = best_ind_mean + ci_margin
        
        plot_data.append({
            "Epochs": epoch,
            "Method": "Best Individual",
            "Accuracy": best_ind_mean,
            "Std": best_ind_std,
            "Median": best_ind_median,
            "Min": best_ind_min,
            "Max": best_ind_max,
            "CI_Lower": best_ind_ci_lower,
            "CI_Upper": best_ind_ci_upper
        })
    
    # Convert to DataFrames
    df = pd.DataFrame(plot_data)
    raw_df = pd.DataFrame(raw_data)
    
    # Save raw seed-level data for future analysis
    raw_df.to_csv(output_dir / "seed_level_results.csv", index=False)
    
    # Save aggregated statistics
    df.to_csv(output_dir / "aggregated_statistics.csv", index=False)
    
    # 1. Enhanced bar chart for each epoch value
    for epoch in all_results.keys():
        epoch_df = df[df["Epochs"] == epoch]
        
        plt.figure(figsize=(14, 10))
        
        # Plot bars with error bars showing 95% CI
        ax = sns.barplot(
            x="Method", 
            y="Accuracy", 
            data=epoch_df,
            palette="viridis",
            capsize=0.2,
        )
        
        # Add confidence interval as error bars
        for i, row in epoch_df.iterrows():
            ax.errorbar(
                i, row["Accuracy"], 
                yerr=[[row["Accuracy"]-row["CI_Lower"]], [row["CI_Upper"]-row["Accuracy"]]], 
                fmt='none', 
                c='black', 
                capsize=5
            )
        
        # Add min, median, max markers
        for i, row in epoch_df.iterrows():
            # Min marker
            plt.scatter(i, row["Min"], color='blue', marker='_', s=300, linewidth=2, label='Min' if i == 0 else "")
            # Median marker
            plt.scatter(i, row["Median"], color='green', marker='_', s=300, linewidth=2, label='Median' if i == 0 else "")
            # Max marker
            plt.scatter(i, row["Max"], color='red', marker='_', s=300, linewidth=2, label='Max' if i == 0 else "")
            
        # Customize plot
        plt.title(f"Model Merging Methods Comparison - {epoch} Epochs", fontsize=16)
        plt.ylabel("Test Accuracy", fontsize=14)
        plt.xlabel("Method", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of bars
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            row = epoch_df.iloc[i]
            ax.text(
                p.get_x() + p.get_width() / 2.,
                height + 0.01,
                f'mean: {row["Accuracy"]:.4f}\nstd: {row["Std"]:.4f}\nmedian: {row["Median"]:.4f}',
                ha="center", fontsize=10
            )
        
        # Add legend for min/median/max markers
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:  # Only add legend if we have items
            plt.legend(handles, labels, title="Statistics", loc='best')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_dir / f"merging_comparison_epoch{epoch}.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # Also create a violin plot to show the distribution
        plt.figure(figsize=(14, 10))
        # Extract seed-level data for this epoch
        epoch_raw_df = raw_df[raw_df["Epochs"] == epoch]
        # Create violin plot
        sns.violinplot(x="Method", y="Accuracy", data=epoch_raw_df, palette="viridis", inner="points")
        plt.title(f"Distribution of Accuracies Across Seeds - {epoch} Epochs", fontsize=16)
        plt.ylabel("Test Accuracy", fontsize=14)
        plt.xlabel("Method", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"accuracy_distribution_epoch{epoch}.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # 2. Enhanced line plot showing accuracy vs epochs for each method
    plt.figure(figsize=(14, 10))
    
    # Plot lines with confidence interval bands
    for method in df["Method"].unique():
        method_df = df[df["Method"] == method]
        
        # Sort by epochs for proper line plotting
        method_df = method_df.sort_values("Epochs")
        
        # Plot the mean line
        plt.plot(
            method_df["Epochs"], 
            method_df["Accuracy"], 
            marker='o',
            markersize=8,
            label=method
        )
        
        # Add error bands
        plt.fill_between(
            method_df["Epochs"],
            method_df["CI_Lower"],
            method_df["CI_Upper"],
            alpha=0.2
        )
    
    # Customize plot
    plt.title(f"Effect of Training Duration on Merging Methods", fontsize=16)
    plt.ylabel("Test Accuracy", fontsize=14)
    plt.xlabel("Number of Epochs", fontsize=14)
    plt.xticks(list(all_results.keys()), fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Method", fontsize=12, title_fontsize=14)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / f"merging_comparison_across_epochs.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Enhanced heatmap of accuracy improvement over best individual model
    improvement_data = []
    improvement_ci_data = []  # For confidence intervals on improvement
    
    for epoch in all_results.keys():
        epoch_df = df[df["Epochs"] == epoch]
        best_ind_row = epoch_df[epoch_df["Method"] == "Best Individual"].iloc[0]
        best_ind_acc = best_ind_row["Accuracy"]
        
        # Get raw seed-level data for calculating seed-paired improvements
        epoch_raw_df = raw_df[raw_df["Epochs"] == epoch]
        
        for method in merging_methods:
            method_row = epoch_df[epoch_df["Method"] == method].iloc[0]
            method_acc = method_row["Accuracy"]
            improvement = method_acc - best_ind_acc
            
            # Get confidence intervals on the improvement
            method_ci_lower = method_row["CI_Lower"] - best_ind_row["CI_Upper"]  # Conservative lower bound
            method_ci_upper = method_row["CI_Upper"] - best_ind_row["CI_Lower"]  # Conservative upper bound
            
            improvement_data.append({
                "Epochs": epoch,
                "Method": method,
                "Improvement": improvement,
                "CI_Lower": method_ci_lower,
                "CI_Upper": method_ci_upper
            })
            
            # Calculate per-seed paired improvements
            # This gives more accurate statistics by pairing results from same seed
            seeds = epoch_raw_df["Seed"].unique()
            seed_improvements = []
            
            for seed in seeds:
                seed_df = epoch_raw_df[epoch_raw_df["Seed"] == seed]
                try:
                    method_seed_acc = seed_df[(seed_df["Method"] == method)]["Accuracy"].values[0]
                    best_ind_seed_acc = seed_df[(seed_df["Method"] == "Best Individual")]["Accuracy"].values[0]
                    seed_improvement = method_seed_acc - best_ind_seed_acc
                    seed_improvements.append(seed_improvement)
                    
                    # Save detailed seed-level improvement data
                    improvement_ci_data.append({
                        "Epochs": epoch,
                        "Method": method,
                        "Seed": seed,
                        "Method_Acc": method_seed_acc,
                        "Best_Ind_Acc": best_ind_seed_acc,
                        "Improvement": seed_improvement
                    })
                except (IndexError, KeyError):
                    # Skip if we don't have both method and best individual for this seed
                    continue
    
    # Convert to DataFrames
    imp_df = pd.DataFrame(improvement_data)
    imp_ci_df = pd.DataFrame(improvement_ci_data)
    
    # Save detailed improvement data
    imp_ci_df.to_csv(output_dir / "seed_level_improvements.csv", index=False)
    
    # Create pivot table for heatmap
    pivot_df = imp_df.pivot(index="Method", columns="Epochs", values="Improvement")
    
    plt.figure(figsize=(12, 8))
    
    # Create heatmap with annotations showing mean improvement and confidence interval
    ax = sns.heatmap(
        pivot_df, 
        annot=False,  # We'll add custom annotations
        fmt="",
        cmap="RdYlGn",
        center=0,
        linewidths=.5
    )
    
    # Add custom annotations with confidence intervals
    for i, method in enumerate(pivot_df.index):
        for j, epoch in enumerate(pivot_df.columns):
            try:
                # Get improvement and CI values
                row = imp_df[(imp_df["Method"] == method) & (imp_df["Epochs"] == epoch)].iloc[0]
                improvement = row["Improvement"]
                ci_lower = row["CI_Lower"]
                ci_upper = row["CI_Upper"]
                
                # Create annotation text with CI
                text = f"{improvement:.4f}\n[{ci_lower:.4f}, {ci_upper:.4f}]"
                
                # Add text annotation
                ax.text(j + 0.5, i + 0.5, text, 
                        ha="center", va="center", 
                        fontsize=9,
                        color="black" if abs(improvement) < 0.3 else "white")  # Adjust text color for readability
            except (IndexError, KeyError):
                continue
    
    plt.title("Improvement Over Best Individual Model with 95% CI", fontsize=16)
    plt.ylabel("Merging Method", fontsize=14)
    plt.xlabel("Number of Epochs", fontsize=14)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / f"improvement_heatmap_with_ci.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. Create a detailed statistical summary table
    summary_data = []
    for method in list(merging_methods) + ["Avg Individual", "Best Individual"]:
        method_df = df[df["Method"] == method]
        
        for epoch in all_results.keys():
            try:
                row = method_df[method_df["Epochs"] == epoch].iloc[0]
                
                # Extract all statistics
                summary_data.append({
                    "Epochs": epoch,
                    "Method": method,
                    "Mean": row["Accuracy"],
                    "Std": row["Std"],
                    "Median": row["Median"],
                    "Min": row["Min"],
                    "Max": row["Max"],
                    "CI_Lower": row["CI_Lower"],
                    "CI_Upper": row["CI_Upper"],
                    "CI_Width": row["CI_Upper"] - row["CI_Lower"]
                })
            except IndexError:
                continue
    
    # Create and save summary table
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "statistical_summary.csv", index=False)
    
    print(f"Enhanced visualization complete. Results saved to {output_dir}")
    print(f"Raw seed-level data saved to {output_dir}/seed_level_results.csv")
    print(f"Seed-level improvements saved to {output_dir}/seed_level_improvements.csv")
    print(f"Full statistical summary saved to {output_dir}/statistical_summary.csv")


# Example usage
if __name__ == "__main__":
    base_config = MergeExperimentConfig(
        experiment_name="merge_experiment_iid_cnn",
        dataset_name="femnist",
        model_name="cnn",
        num_models=10,
        batch_size=64,
        initialization_type="identical",
        # seed will be set in the function
        merging_methods=["c2m3", "fedavg", "simple_avg", "median"],
        data_distribution="iid"
    )

    epoch_values = [21, 23, 25, 27, 29]
    seeds = [42, 123, 456, 789, 101]  # 5 random seeds as requested
    
    # Run the multi-seed experiment
    results = run_multi_seed_epochs_experiment(base_config, epoch_values, seeds)