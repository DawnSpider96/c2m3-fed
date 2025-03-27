import os
import json
import time
import logging
from typing import Dict, List, Union, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import random
import copy

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
    get_network_generator_cnn,
    get_network_generator_cnn_random,
    get_network_generator_tiny_resnet,
    get_network_generator_tiny_resnet_random,
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
    dataset_name: str  # 'femnist', 'cifar10'
    model_name: str  # 'resnet18', 'cnn'
    
    seed: int = -1
    data_dir: str = str(Path(__file__).parent.parent / "data")
    
    # Training configuration
    num_models: int = 5
    epochs_per_model: Union[int, List[int]] = 10
    batch_size: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    
    # Early stopping configuration
    early_stopping: bool = False    
    early_stopping_patience: int = 5  # Number of epochs with no improvement to wait before stopping
    early_stopping_min_delta: float = 0.001  # Minimum change to qualify as improvement
    early_stopping_metric: str = "accuracy"  # Metric to monitor: 'accuracy', 'f1', or 'recall'
    
    # Data distribution
    data_distribution: str = "iid"  # 'iid', 'dirichlet', 'pathological', 'natural'
    non_iid_alpha: float = 5.0  # Dirichlet alpha parameter for non-IID distribution
    classes_per_partition: int = 1  # For pathological partitioning, how many classes per client
    samples_per_partition: Optional[int] = 4000  # Number of samples per partition (for IID FEMNIST), None for equal division
    
    # Initialization
    initialization_type: str = "identical"  # 'identical' or 'random'
    
    # Merging configuration
    merging_methods: List[str] = None  # ['c2m3', 'fedavg', 'ties', 'median']
    c2m3_max_iter: int = 300  # Max iterations for Frank-Wolfe
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
        elif self.dataset_name == "cifar10":
            # Use the CIFAR10-specific training approach
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
            if total == 0 or len(self.train_loader.dataset) == 0:
                logger.warning(f"Empty CIFAR10 dataset detected during training. Setting loss and accuracy to 0.")
                epoch_loss = 0.0
                epoch_acc = 0.0
            else:
                epoch_loss = total_loss / len(self.train_loader.dataset)
                epoch_acc = correct / total
                
            logger.info(f"CIFAR10 training in progress - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. Only 'femnist' and 'cifar10' are supported.")
            
        # Store metrics
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
        elif self.dataset_name == "cifar10":
            # Use CIFAR10-specific testing approach
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
            
            # Compute metrics for CIFAR10
            if total == 0 or len(self.test_loader.dataset) == 0:
                logger.warning(f"Empty CIFAR10 test dataset detected. Setting loss and accuracy to 0.")
                test_loss = 0.0
                test_acc = 0.0
            else:
                test_loss = total_loss / len(self.test_loader.dataset)
                test_acc = correct / total
                logger.info(f"CIFAR10 test accuracy: {test_acc:.4f}")
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}. Only 'femnist' and 'cifar10' are supported.")
        
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
        
        if not hasattr(self, 'train_loaders') or self.train_loaders is None:
            raise ValueError("Train loaders not available for C2M3 merging. Cannot proceed with model merging.")
        else:
            # Use the train_loaders that were passed during initialization
            logger.info("Using train loaders for C2M3 merging to update batch norm statistics.")
            # Create MyLightningModule instances from state_dicts
            lightning_modules = {}
            
            if self.config.dataset_name == "femnist":
                network_generator = get_network_generator_cnn()
                logger.info("Using FEMNIST-specific network generator for merging")
            elif self.config.dataset_name == "cifar10":
                network_generator = get_network_generator_tiny_resnet()
                logger.info("Using CIFAR10-specific network generator for merging")
            else:
                raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
            
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
                elif self.config.dataset_name == "cifar10":
                    # cifar10 has 10 classes
                    num_classes = 10
                else:
                    raise ValueError(f"Unknown dataset: {self.config.dataset_name}")
                
                mlm = MyLightningModule(net, num_classes=num_classes)
                
                # Store in dictionary with symbol key
                lightning_modules[symbol] = mlm
            
            merged_model, repaired_model, models_permuted_to_universe = merger(lightning_modules, train_loader=self.train_loaders)
            
            merged_state_dict = repaired_model.model.state_dict()
        
        if self.config.dataset_name == "femnist":
            m = get_network_generator_cnn()()
        elif self.config.dataset_name == "cifar10":
            m = get_network_generator_tiny_resnet()()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
        
        m.load_state_dict(merged_state_dict)
        m.to(self.device)
        
        return m
    
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
        
        if self.config.dataset_name == "femnist":
            merged_model = get_network_generator_cnn()()
        elif self.config.dataset_name == "cifar10":
            merged_model = get_network_generator_tiny_resnet()()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
        
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
            merged_model = get_network_generator_cnn()()
        elif self.config.dataset_name == "cifar10":
            merged_model = get_network_generator_tiny_resnet()()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
        
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
            merged_model = get_network_generator_cnn()()
        elif self.config.dataset_name == "cifar10":
            merged_model = get_network_generator_tiny_resnet()()
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
        
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
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")
        
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
            if self.config.initialization_type == "identical":
                network_generator = get_network_generator_cnn()
                self.models = [network_generator() for _ in range(self.config.num_models)]
                logger.info(f"Created {len(self.models)} identical FEMNIST models")
            else: # random initialization
                network_generator = get_network_generator_cnn_random()
                self.models = [network_generator() for _ in range(self.config.num_models)]
                logger.info(f"Created {len(self.models)} FEMNIST models with diverse initialization")
        elif self.config.dataset_name == "cifar10":
            logger.info("Using CIFAR10-specific model initialization")            
            if self.config.initialization_type == "identical":
                network_generator = get_network_generator_tiny_resnet()
                self.models = [network_generator() for _ in range(self.config.num_models)]
                logger.info(f"Created {len(self.models)} identical CIFAR10 models")
            else:  # random initialization
                network_generator = get_network_generator_tiny_resnet_random()
                self.models = [network_generator() for _ in range(self.config.num_models)]
                logger.info(f"Created {len(self.models)} CIFAR10 models with diverse initialization")
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}. Only 'femnist' and 'cifar10' are supported.")
        
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
    
    # Create a parent directory for all parameter sweep results
    parent_dir = Path(f"./results/{base_config.experiment_name}_parameter_sweep")
    parent_dir.mkdir(exist_ok=True, parents=True)
    
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
        
        # Set output directory to parent directory
        config_dict["output_dir"] = str(parent_dir)
        
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


def run_multiple_seeds_experiment(base_config, seeds, output_dir=None):
    """
    Run the same experiment multiple times with different random seeds and analyze the results.
    Similar to run_single_parameter_comparison but specifically for seed variations.
    
    Args:
        base_config: Base configuration for experiments
        seeds: List of random seed values to use
        output_dir: Directory to save the results. Defaults to ./results/{experiment_name}_seeds
        
    Returns:
        Dictionary with aggregated results across all seeds
    """
    # Create a meaningful experiment name based on the base config
    base_experiment_name = base_config.experiment_name
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = f"./results/{base_experiment_name}_seeds"
    
    # Create parameter grid with seeds as the parameter
    parameter_grid = {"seed": seeds}
    
    # Create a copy of the base config to avoid modifying the original
    from dataclasses import asdict
    config_dict = asdict(base_config)
    config_copy = MergeExperimentConfig(**config_dict)
    
    # Set output directory for all experiments
    config_copy.output_dir = output_dir
    
    # Run parameter sweep with seeds as the parameter
    results = run_parameter_sweep(config_copy, parameter_grid)
    
    # Create output directory for aggregated results
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Aggregate results across seeds
    aggregated_results = {
        "config": asdict(base_config),
        "seeds": seeds,
        "num_seeds": len(seeds),
        "merged_models": {},
        "individual_models": {}
    }
    
    # Extract merged model results
    for method in results[0]["merged_models"].keys():
        # Extract test accuracies for this method across all seeds
        accuracies = [result["merged_models"][method]["test_accuracy"] for result in results]
        
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
        
        # Store statistics in aggregated results
        aggregated_results["merged_models"][method] = {
            "accuracies": accuracies,
            "mean": mean_acc,
            "std": std_acc,
            "median": median_acc,
            "min": min_acc,
            "max": max_acc,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }
    
    # Also aggregate individual model results
    aggregated_results["individual_models"] = {}
    
    # Calculate best individual model statistics
    best_individual_accs = []
    avg_individual_accs = []
    
    for result in results:
        # Extract individual model accuracies
        individual_accs = [model_data["final_test_acc"] for model_data in result["models"].values()]
        
        if individual_accs:
            best_individual_accs.append(max(individual_accs))
            avg_individual_accs.append(sum(individual_accs) / len(individual_accs))
    
    # Calculate statistics for best individual model
    aggregated_results["individual_models"]["best"] = {
        "accuracies": best_individual_accs,
        "mean": np.mean(best_individual_accs),
        "std": np.std(best_individual_accs),
        "median": np.median(best_individual_accs),
        "min": np.min(best_individual_accs),
        "max": np.max(best_individual_accs)
    }
    
    # Calculate statistics for average individual model
    aggregated_results["individual_models"]["average"] = {
        "accuracies": avg_individual_accs,
        "mean": np.mean(avg_individual_accs),
        "std": np.std(avg_individual_accs),
        "median": np.median(avg_individual_accs),
        "min": np.min(avg_individual_accs),
        "max": np.max(avg_individual_accs)
    }
    
    # Save aggregated results
    with open(output_path / "aggregated_results.json", "w") as f:
        json.dump(aggregated_results, f, indent=4)
    
    # Create visualizations
    visualize_seed_aggregated_results(aggregated_results, output_path)
    
    return aggregated_results


def visualize_seed_aggregated_results(aggregated_results, output_dir):
    """
    Create visualizations for results aggregated across multiple seeds.
    
    Args:
        aggregated_results: Dictionary with aggregated results
        output_dir: Directory to save visualizations
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert output_dir to Path object if it's a string
    output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
    
    # Set style
    sns.set(style="whitegrid")
    
    # 1. Create bar chart with error bars showing mean accuracy and confidence intervals
    plt.figure(figsize=(12, 8))
    
    # Prepare data for plotting
    methods = []
    accuracies = []
    ci_lowers = []
    ci_uppers = []
    
    # Add merged model methods
    for method, stats in aggregated_results["merged_models"].items():
        methods.append(method)
        accuracies.append(stats["mean"])
        ci_lowers.append(stats["ci_lower"])
        ci_uppers.append(stats["ci_upper"])
    
    # Add individual model references
    methods.append("Best Individual")
    accuracies.append(aggregated_results["individual_models"]["best"]["mean"])
    
    best_ind_std = aggregated_results["individual_models"]["best"]["std"]
    best_ind_n = len(aggregated_results["individual_models"]["best"]["accuracies"])
    t_value = stats.t.ppf(0.975, best_ind_n - 1)  # 95% CI
    best_ind_ci_margin = t_value * (best_ind_std / np.sqrt(best_ind_n))
    
    ci_lowers.append(aggregated_results["individual_models"]["best"]["mean"] - best_ind_ci_margin)
    ci_uppers.append(aggregated_results["individual_models"]["best"]["mean"] + best_ind_ci_margin)
    
    methods.append("Avg Individual")
    accuracies.append(aggregated_results["individual_models"]["average"]["mean"])
    
    avg_ind_std = aggregated_results["individual_models"]["average"]["std"]
    avg_ind_n = len(aggregated_results["individual_models"]["average"]["accuracies"])
    avg_ind_ci_margin = t_value * (avg_ind_std / np.sqrt(avg_ind_n))
    
    ci_lowers.append(aggregated_results["individual_models"]["average"]["mean"] - avg_ind_ci_margin)
    ci_uppers.append(aggregated_results["individual_models"]["average"]["mean"] + avg_ind_ci_margin)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        "Method": methods,
        "Accuracy": accuracies,
        "CI_Lower": ci_lowers,
        "CI_Upper": ci_uppers
    })
    
    # Create bar plot
    ax = sns.barplot(
        x="Method", 
        y="Accuracy", 
        data=df,
        palette="viridis",
        capsize=0.2,
    )
    
    # Add confidence interval as error bars
    for i, row in df.iterrows():
        ax.errorbar(
            i, row["Accuracy"], 
            yerr=[[row["Accuracy"]-row["CI_Lower"]], [row["CI_Upper"]-row["Accuracy"]]], 
            fmt='none', 
            c='black', 
            capsize=5
        )
    
    # Add individual data points to show distribution
    for i, method in enumerate(methods):
        if method in aggregated_results["merged_models"]:
            points = aggregated_results["merged_models"][method]["accuracies"]
        elif method == "Best Individual":
            points = aggregated_results["individual_models"]["best"]["accuracies"]
        else:  # Avg Individual
            points = aggregated_results["individual_models"]["average"]["accuracies"]
        
        # Add jittered points
        for point in points:
            jitter = np.random.uniform(-0.2, 0.2)
            plt.scatter(i + jitter, point, color='black', alpha=0.6, s=30)
    
    # Customize plot
    plt.title(f"Model Merging Methods Comparison Across {aggregated_results['num_seeds']} Seeds", fontsize=16)
    plt.ylabel("Test Accuracy", fontsize=14)
    plt.xlabel("Method", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.,
            height + 0.005,
            f'{df["Accuracy"].iloc[i]:.4f}',
            ha="center", fontsize=12, fontweight='bold'
        )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / "merging_comparison_across_seeds.png", dpi=300, bbox_inches="tight")
    
    # 2. Create boxplot to show distribution of accuracies
    plt.figure(figsize=(12, 8))
    
    # Prepare data for boxplot
    boxplot_data = []
    
    for method, stats in aggregated_results["merged_models"].items():
        for acc in stats["accuracies"]:
            boxplot_data.append({"Method": method, "Accuracy": acc})
    
    # Add individual model references
    for acc in aggregated_results["individual_models"]["best"]["accuracies"]:
        boxplot_data.append({"Method": "Best Individual", "Accuracy": acc})
    
    for acc in aggregated_results["individual_models"]["average"]["accuracies"]:
        boxplot_data.append({"Method": "Avg Individual", "Accuracy": acc})
    
    # Create DataFrame for plotting
    boxplot_df = pd.DataFrame(boxplot_data)
    
    # Create boxplot
    sns.boxplot(x="Method", y="Accuracy", data=boxplot_df, palette="viridis")
    
    # Add individual points
    sns.stripplot(x="Method", y="Accuracy", data=boxplot_df, color="black", alpha=0.5, jitter=True)
    
    # Customize plot
    plt.title(f"Distribution of Test Accuracies Across {aggregated_results['num_seeds']} Seeds", fontsize=16)
    plt.ylabel("Test Accuracy", fontsize=14)
    plt.xlabel("Method", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_distribution_across_seeds.png", dpi=300, bbox_inches="tight")
    
    # 3. Create a table with detailed statistics
    table_data = []
    
    # Add merged model methods
    for method, stats in aggregated_results["merged_models"].items():
        table_data.append({
            "Method": method,
            "Mean": stats["mean"],
            "Std": stats["std"],
            "Median": stats["median"],
            "Min": stats["min"],
            "Max": stats["max"],
            "95% CI Lower": stats["ci_lower"],
            "95% CI Upper": stats["ci_upper"]
        })
    
    # Add individual model references
    table_data.append({
        "Method": "Best Individual",
        "Mean": aggregated_results["individual_models"]["best"]["mean"],
        "Std": aggregated_results["individual_models"]["best"]["std"],
        "Median": aggregated_results["individual_models"]["best"]["median"],
        "Min": aggregated_results["individual_models"]["best"]["min"],
        "Max": aggregated_results["individual_models"]["best"]["max"],
        "95% CI Lower": df[df["Method"] == "Best Individual"]["CI_Lower"].values[0],
        "95% CI Upper": df[df["Method"] == "Best Individual"]["CI_Upper"].values[0]
    })
    
    table_data.append({
        "Method": "Avg Individual",
        "Mean": aggregated_results["individual_models"]["average"]["mean"],
        "Std": aggregated_results["individual_models"]["average"]["std"],
        "Median": aggregated_results["individual_models"]["average"]["median"],
        "Min": aggregated_results["individual_models"]["average"]["min"],
        "Max": aggregated_results["individual_models"]["average"]["max"],
        "95% CI Lower": df[df["Method"] == "Avg Individual"]["CI_Lower"].values[0],
        "95% CI Upper": df[df["Method"] == "Avg Individual"]["CI_Upper"].values[0]
    })
    
    # Create DataFrame for table
    table_df = pd.DataFrame(table_data)
    
    # Save table as CSV
    table_df.to_csv(output_dir / "statistical_summary.csv", index=False)
    
    # 4. Create heatmap of improvement over best individual model
    improvement_data = []
    
    for method, stats in aggregated_results["merged_models"].items():
        # Calculate improvement over best individual model
        improvement = stats["mean"] - aggregated_results["individual_models"]["best"]["mean"]
        
        # Calculate statistical significance
        # Use independent t-test
        from scipy import stats as scipy_stats
        t_stat, p_value = scipy_stats.ttest_ind(
            stats["accuracies"],
            aggregated_results["individual_models"]["best"]["accuracies"],
            equal_var=False  # Use Welch's t-test (doesn't assume equal variance)
        )
        
        improvement_data.append({
            "Method": method,
            "Improvement": improvement,
            "t_statistic": t_stat,
            "p_value": p_value,
            "Significant": p_value < 0.05
        })
    
    # Create DataFrame for improvement
    improvement_df = pd.DataFrame(improvement_data)
    
    # Save improvement data
    improvement_df.to_csv(output_dir / "improvement_statistics.csv", index=False)
    
    # Create bar plot for improvement
    plt.figure(figsize=(12, 6))
    
    bars = sns.barplot(
        x="Method", 
        y="Improvement", 
        data=improvement_df,
        palette=["green" if x else "red" for x in improvement_df["Significant"]]
    )
    
    # Add asterisks for significant improvements
    for i, row in improvement_df.iterrows():
        if row["Significant"]:
            plt.text(
                i, 
                row["Improvement"] + 0.002,
                "*", 
                ha="center", fontsize=20
            )
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Customize plot
    plt.title(f"Improvement Over Best Individual Model", fontsize=16)
    plt.ylabel("Improvement in Test Accuracy", fontsize=14)
    plt.xlabel("Method", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_dir / "improvement_over_best_individual.png", dpi=300, bbox_inches="tight")
    
    # Print summary statistics
    logger.info(f"=== Aggregated Results Across {aggregated_results['num_seeds']} Seeds ===")
    logger.info("Merged Models:")
    
    for method, stats in aggregated_results["merged_models"].items():
        logger.info(f"  {method}: Acc={stats['mean']:.4f} ± {stats['std']:.4f} (95% CI: [{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}])")
    
    logger.info(f"Best Individual: Acc={aggregated_results['individual_models']['best']['mean']:.4f} ± {aggregated_results['individual_models']['best']['std']:.4f}")
    logger.info(f"Avg Individual: Acc={aggregated_results['individual_models']['average']['mean']:.4f} ± {aggregated_results['individual_models']['average']['std']:.4f}")
    
    logger.info("Improvements over Best Individual:")
    for _, row in improvement_df.iterrows():
        significance = "significant" if row["Significant"] else "not significant"
        logger.info(f"  {row['Method']}: {row['Improvement']:.4f} ({significance}, p={row['p_value']:.4f})")
        

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
    Uses incremental training approach - models are trained once and evaluated at different epoch checkpoints.
    
    Args:
        base_config: Base configuration for experiments
        epoch_values: Epoch values to use for training models (in ascending order)
        seeds: List of random seeds to use for each run
    """
    # Make sure epoch values are sorted in ascending order
    epoch_values = sorted(epoch_values)
    
    # Structure to hold all results
    all_results = {epoch: [] for epoch in epoch_values}
    
    # Create base experiment name once
    base_experiment_name = base_config.experiment_name
    
    # Create parent directory for all experiment results
    from pathlib import Path
    parent_dir = Path(f"./results/{base_experiment_name}")
    parent_dir.mkdir(exist_ok=True, parents=True)
    
    # For each seed
    for idx, seed in enumerate(seeds):
        print(f"Using seed {seed}...")
        
        # Create a new config with the seed changed
        from dataclasses import asdict
        base_config_dict = asdict(base_config)
        
        # Save original experiment name before removing from dict
        original_name = base_config_dict["experiment_name"]
        
        # Remove experiment_name from the dictionary to avoid duplication
        del base_config_dict["experiment_name"]
        
        # Update the seed 
        base_config_dict["seed"] = seed
        
        # Modify the output directory to be within the parent directory
        base_config_dict["output_dir"] = str(parent_dir)
        
        # Create a runner instance for setup once
        # We'll use an initial config with the first epoch value for setup
        base_config_dict["epochs_per_model"] = [epoch_values[0]] * base_config.num_models
        initial_config = MergeExperimentConfig(
            **base_config_dict,
            experiment_name=original_name + f"_seed{seed}_incremental"
        )
        
        # Create a runner and do initial setup
        runner = MergeExperimentRunner(initial_config)
        runner.setup()
        
        # Keep track of accumulated epochs for each model
        accumulated_epochs = 0
        
        # We'll train each epoch interval separately
        for i, target_epoch in enumerate(epoch_values):
            print(f"  Training to epoch {target_epoch}...")
            
            # Calculate how many more epochs we need to train
            epochs_to_train = target_epoch - accumulated_epochs
            
            if epochs_to_train > 0:
                # Training for additional epochs
                for model_idx, (model, train_loader, val_loader) in enumerate(zip(
                    runner.models, runner.train_loaders, runner.val_loaders
                )):
                    print(f"    Training model {model_idx+1}/{base_config.num_models} for {epochs_to_train} more epochs...")
                    
                    trainer = ModelTrainer(
                        model, train_loader, runner.test_loader,
                        runner.config.learning_rate, runner.config.weight_decay,
                        runner.device, runner.config.dataset_name,
                        val_loader=val_loader,
                        patience=runner.config.early_stopping_patience if hasattr(runner.config, "early_stopping_patience") else 5,
                        min_delta=runner.config.early_stopping_min_delta if hasattr(runner.config, "early_stopping_min_delta") else 0.001
                    )
                    
                    # Train for additional epochs
                    # Note: early stopping is not used in incremental training as it would complicate the process
                    model_results = trainer.train(
                        epochs_to_train, 
                        early_stopping=False
                    )
                    
                    # Update model evaluation results in the runner
                    if "models" not in runner.results:
                        runner.results["models"] = {}
                    
                    runner.results["models"][f"model_{model_idx}"] = {
                        "epochs": target_epoch,  # Total epochs trained so far
                        "actual_epochs": target_epoch,  # No early stopping
                        "early_stopped": False,
                        **model_results
                    }
            
            # Update accumulated epochs
            accumulated_epochs = target_epoch
            
            # Now create copies of the models for merging and evaluation
            original_models = runner.models
            copied_models = [copy.deepcopy(model) for model in original_models]
            
            # Create a checkpoint-specific config for this evaluation
            checkpoint_config_dict = base_config_dict.copy()
            checkpoint_config_dict["epochs_per_model"] = [target_epoch] * base_config.num_models
            checkpoint_suffix = f"_seed{seed}_epochs{target_epoch}"
            
            checkpoint_config = MergeExperimentConfig(
                **checkpoint_config_dict,
                experiment_name=original_name + checkpoint_suffix
            )
            
            # Update runner's config for this checkpoint
            runner.config = checkpoint_config
            
            # Save the original models temporarily
            temp_models = runner.models
            
            # Set the copied models for merging
            runner.models = copied_models
            
            # Perform model merging and evaluation
            # We're reusing the runner's merge logic but with our copied models
            runner.results["merged_models"] = {}
            
            # Initialize model merger with train_loaders if not already done
            if runner.model_merger is None:
                runner.model_merger = ModelMerger(runner.config, runner.device, runner.train_loaders)
            
            for method in runner.config.merging_methods:
                print(f"Merging models using {method} at epoch {target_epoch}...")
                
                # Merge models
                merged_model = runner.model_merger.merge(copied_models, method)
                
                # Evaluate merged model
                trainer = ModelTrainer(
                    merged_model, None, runner.test_loader,
                    runner.config.learning_rate, runner.config.weight_decay,
                    runner.device, runner.config.dataset_name
                )
                
                _, test_acc = trainer.evaluate()
                
                # Save results
                runner.results["merged_models"][method] = {
                    "test_accuracy": test_acc
                }
            
            # Create a copy of the results for this checkpoint
            checkpoint_result = copy.deepcopy(runner.results)
            
            # Add a field to track which epoch value this run is associated with
            checkpoint_result["tracked_epoch_value"] = target_epoch
            checkpoint_result["config"] = asdict(checkpoint_config)
            
            # Store the result
            all_results[target_epoch].append(checkpoint_result)
            
            # Restore the original models for continued training
            runner.models = temp_models
    
    # Now create visualizations across all seeds and epochs
    visualize_incremental_results(all_results, base_experiment_name, parent_dir)
    
    return all_results

def visualize_incremental_results(all_results, experiment_name, parent_dir):
    """
    Visualize results aggregated across multiple seeds for each epoch value.
    Modified for incremental training approach.
    
    Args:
        all_results: Dictionary mapping epoch values to lists of result dictionaries
        experiment_name: Name of the experiment for plot titles and filenames
        parent_dir: Parent directory to save visualizations
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from scipy import stats
    import copy
    
    # Set style
    sns.set(style="whitegrid")
    
    # Use provided parent directory
    output_dir = parent_dir
    
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
        ci_lower = avg_ind_mean - ci_margin
        ci_upper = avg_ind_mean + ci_margin
        
        # Store aggregated stats
        plot_data.append({
            "Epochs": epoch,
            "Method": "Avg Individual",
            "Accuracy": avg_ind_mean,
            "Std": avg_ind_std,
            "Median": avg_ind_median,
            "Min": avg_ind_min,
            "Max": avg_ind_max,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper
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
        ci_lower = best_ind_mean - ci_margin
        ci_upper = best_ind_mean + ci_margin
        
        # Store aggregated stats
        plot_data.append({
            "Epochs": epoch,
            "Method": "Best Individual",
            "Accuracy": best_ind_mean,
            "Std": best_ind_std,
            "Median": best_ind_median,
            "Min": best_ind_min,
            "Max": best_ind_max,
            "CI_Lower": ci_lower,
            "CI_Upper": ci_upper
        })
    
    # Convert to dataframe
    plot_df = pd.DataFrame(plot_data)
    
    # Save this data
    plot_df.to_csv(output_dir / f"incremental_aggregated_results.csv", index=False)
    
    # Convert raw data to dataframe and save
    raw_df = pd.DataFrame(raw_data)
    raw_df.to_csv(output_dir / f"incremental_raw_results.csv", index=False)
    
    # Create line plot with error bands for different methods
    plt.figure(figsize=(14, 8))
    
    # Plot each method
    for method in sorted(set(plot_df["Method"])):
        method_data = plot_df[plot_df["Method"] == method]
        
        # Skip if we have no data
        if len(method_data) == 0:
            continue
            
        # Sort by epochs
        method_data = method_data.sort_values("Epochs")
        
        # Plot the line with confidence interval
        plt.plot(method_data["Epochs"], method_data["Accuracy"], 
                marker='o', label=method)
        
        # Add confidence interval
        plt.fill_between(
            method_data["Epochs"],
            method_data["CI_Lower"],
            method_data["CI_Upper"],
            alpha=0.2
        )
    
    # Add labels and title
    plt.title(f"Model Performance by Training Epochs - {experiment_name} (Incremental Training)")
    plt.xlabel("Training Epochs")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.grid(True)
    
    # Save figure
    plt.savefig(output_dir / f"incremental_performance_by_epochs.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"incremental_performance_by_epochs.pdf", bbox_inches='tight')
    
    # Create a violin plot comparing methods at each epoch
    plt.figure(figsize=(18, 10))
    
    # Filter for only merge methods (exclude individual models)
    merge_methods = [m for m in merging_methods]
    merge_raw_df = raw_df[raw_df["Method"].isin(merge_methods)]
    
    # Add individual models for comparison
    ind_methods = ["Avg Individual", "Best Individual"]
    ind_raw_df = raw_df[raw_df["Method"].isin(ind_methods)]
    
    # Combine dataframes
    plot_raw_df = pd.concat([merge_raw_df, ind_raw_df])
    
    # Create violin plot
    sns.violinplot(x="Epochs", y="Accuracy", hue="Method", data=plot_raw_df, 
                  palette="Set2", split=False, inner="quart")
    
    # Add labels and title
    plt.title(f"Model Performance Distribution by Training Epochs - {experiment_name} (Incremental Training)")
    plt.xlabel("Training Epochs")
    plt.ylabel("Test Accuracy")
    
    # Adjust legend
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save figure
    plt.savefig(output_dir / f"incremental_performance_distribution.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"incremental_performance_distribution.pdf", bbox_inches='tight')
    
    # Create a detailed heatmap showing relative performance to best individual model
    # This helps understand when merged models outperform the best individual model
    
    # First, create a pivot table with epochs as rows and methods as columns
    mean_perf = plot_df.pivot(index="Epochs", columns="Method", values="Accuracy")
    
    # Calculate relative performance compared to Best Individual
    best_ind_perf = mean_perf["Best Individual"].copy()
    relative_perf = mean_perf.copy()
    
    for col in relative_perf.columns:
        relative_perf[col] = ((relative_perf[col] / best_ind_perf) - 1) * 100  # Percentage improvement
    
    # Create heatmap
    plt.figure(figsize=(14, 8))
    
    # Create the heatmap, excluding the Best Individual column itself
    rel_perf_plot = relative_perf.drop(columns=["Best Individual"], errors='ignore')
    
    # Create custom diverging colormap centered at 0
    from matplotlib.colors import LinearSegmentedColormap
    colors = ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"]
    cmap = LinearSegmentedColormap.from_list("custom_diverging", colors, N=256)
    
    # Plot the heatmap with custom formatting
    sns.heatmap(rel_perf_plot, annot=True, cmap=cmap, center=0,
               vmin=-10, vmax=10, fmt=".2f", linewidths=0.5, 
               cbar_kws={"label": "% Improvement over Best Individual Model"})
    
    # Add labels and title
    plt.title(f"Relative Performance Compared to Best Individual Model - {experiment_name} (Incremental Training)")
    plt.xlabel("Merging Method")
    plt.ylabel("Training Epochs")
    
    # Save the figure
    plt.savefig(output_dir / f"incremental_relative_performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f"incremental_relative_performance_heatmap.pdf", bbox_inches='tight')
    
    # Return the plot dataframe for additional analysis if needed
    return plot_df


# Example usage
if __name__ == "__main__":
    # base_config = MergeExperimentConfig(
    #     experiment_name="merge_experiment_LDA_0.1",
    #     dataset_name="femnist",
    #     model_name="cnn",
    #     num_models=5,
    #     batch_size=64,
    #     initialization_type="identical",
    #     # seed will be set in the function
    #     merging_methods=["c2m3", "fedavg", "simple_avg", "median"],
    #     data_distribution="dirichlet",
    #     non_iid_alpha=0.1
    # )

    base_config = MergeExperimentConfig(
        experiment_name="merge_experiment_iid",
        dataset_name="femnist",
        model_name="cnn",
        num_models=5,
        batch_size=64,
        initialization_type="identical",
        # seed will be set in the function
        merging_methods=["c2m3", "fedavg", "simple_avg", "median"],
        data_distribution="iid"
    )

    epoch_values = [5, 10, 15, 20, 25]
    seeds = [42, 123, 456, 789, 101]  # 5 random seeds as requested
    
    # Run the multi-seed exWrite me a litmus test to differentiate between LDAR and LDAPR periment
    results = run_multi_seed_epochs_experiment(base_config, epoch_values, seeds)
    
    print("All experiment results are now organized in the results/merge_experiment_random_cnn directory")
    print("You can reproduce all visualizations using the saved CSV and JSON data in this directory")