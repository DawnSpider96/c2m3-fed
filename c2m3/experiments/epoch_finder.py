"""
K-fold cross-validation for determining optimal number of epochs.

This module uses k-fold cross-validation to determine the optimal number of epochs
for model training, which will then be used in the merge experiments.
"""

import os
import copy
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import KFold

from c2m3.experiments.merge_experiment import MergeExperimentConfig, ModelTrainer, DataManager
from c2m3.common.model_utils import get_model

logger = logging.getLogger(__name__)

@dataclass
class EpochFinderConfig:
    """Configuration for k-fold cross-validation to find optimal epochs"""
    # Base experiment configuration
    base_config: MergeExperimentConfig
    
    # K-fold specific configuration
    num_folds: int = 5  # Number of folds for cross-validation
    max_epochs: int = 100  # Maximum number of epochs to train for
    patience: int = 5  # Early stopping patience
    min_delta: float = 0.001  # Minimum improvement for early stopping
    output_dir: str = "./epoch_finder_results"  # Directory to save results
    save_plots: bool = True  # Whether to save plots of the learning curves
    
    # Analysis configuration
    early_stopping_metric: str = "balanced_accuracy"  # Metric to use for early stopping
    metric_goal: str = "max"  # Whether to maximize or minimize the metric


class EpochFinder:
    """
    Use k-fold cross-validation to find the optimal number of epochs for training.
    
    This class performs k-fold cross-validation on a dataset using the same model and
    training setup as the merge experiments, and determines the optimal number of epochs
    based on the validation performance across folds.
    """
    
    def __init__(self, config: EpochFinderConfig):
        """
        Initialize the EpochFinder.
        
        Args:
            config: Configuration for k-fold cross-validation
        """
        self.config = config
        self.base_config = config.base_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize results storage
        self.fold_histories = []
        self.optimal_epochs = None
        
    def setup_data(self):
        """
        Set up the dataset for k-fold cross-validation.
        
        Returns:
            The dataset to use for k-fold cross-validation
        """
        # Create data manager using the base config
        data_manager = DataManager(self.base_config)
        
        # Get the dataset
        logger.info(f"Loading dataset: {self.base_config.dataset_name}")
        train_dataset, test_dataset = data_manager.get_dataset()
        
        return train_dataset, test_dataset
    
    def run_kfold(self, dataset):
        """
        Run k-fold cross-validation to find optimal epochs.
        
        Args:
            dataset: Dataset to use for k-fold cross-validation
            
        Returns:
            Optimal number of epochs for training
        """
        logger.info(f"Running {self.config.num_folds}-fold cross-validation")
        
        # Create k-fold splitter
        kfold = KFold(n_splits=self.config.num_folds, shuffle=True, random_state=self.base_config.seed)
        
        # Get indices for dataset
        indices = list(range(len(dataset)))
        
        # Run k-fold cross-validation
        fold_histories = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            logger.info(f"Training fold {fold+1}/{self.config.num_folds}")
            
            # Create train and validation subsets
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            # Create data loaders
            train_loader = DataLoader(
                train_subset, 
                batch_size=self.base_config.batch_size, 
                shuffle=True
            )
            val_loader = DataLoader(
                val_subset, 
                batch_size=self.base_config.eval_batch_size, 
                shuffle=False
            )
            
            # Create model
            model = get_model(
                self.base_config.model_name, 
                self.base_config.dataset_name
            ).to(self.device)
            
            # Create trainer
            trainer = ModelTrainer(
                model=model,
                train_loader=train_loader,
                test_loader=None,  # We'll use val_loader instead
                lr=self.base_config.learning_rate,
                weight_decay=self.base_config.weight_decay,
                device=self.device,
                dataset_name=self.base_config.dataset_name,
                val_loader=val_loader,
                patience=self.config.patience,
                min_delta=self.config.min_delta
            )
            
            # Train with history tracking
            history = self._train_with_history(trainer)
            fold_histories.append(history)
            
            logger.info(f"Fold {fold+1} complete. Best epoch: {len(history['train_loss'])}")
        
        self.fold_histories = fold_histories
        
        # Analyze results to find optimal epochs
        optimal_epochs = self._analyze_fold_results(fold_histories)
        self.optimal_epochs = optimal_epochs
        
        # Generate plots
        if self.config.save_plots:
            self._generate_plots(fold_histories)
        
        return optimal_epochs
    
    def _train_with_history(self, trainer: ModelTrainer) -> Dict[str, List]:
        """
        Train the model and track metrics per epoch.
        
        Args:
            trainer: ModelTrainer instance
            
        Returns:
            Dictionary of training history (loss, accuracy per epoch)
        """
        # Initialize history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        # Train for max_epochs
        for epoch in range(self.config.max_epochs):
            # Train for one epoch
            train_loss, train_acc = trainer.train_epoch()
            
            # Evaluate on validation set
            val_loss, val_acc, val_balanced_acc = trainer.evaluate_validation()
            
            # Store metrics
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_acc)
            history['val_accuracy'].append(val_balanced_acc)  # Use balanced accuracy
            
            # Check for early stopping
            if trainer.should_stop_early(val_balanced_acc):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        return history
    
    def _analyze_fold_results(self, fold_histories: List[Dict[str, List]]) -> int:
        """
        Analyze results from all folds to determine optimal epochs.
        
        Args:
            fold_histories: List of training histories for each fold
            
        Returns:
            Optimal number of epochs for training
        """
        # Find the maximum length of each fold's history
        max_epochs = max([len(h['val_accuracy']) for h in fold_histories])
        
        # Create arrays to store metrics for each epoch across folds
        val_accuracies = np.zeros((len(fold_histories), max_epochs))
        val_losses = np.zeros((len(fold_histories), max_epochs))
        
        # Fill arrays with data from each fold
        for i, history in enumerate(fold_histories):
            epochs_completed = len(history['val_accuracy'])
            val_accuracies[i, :epochs_completed] = history['val_accuracy']
            val_losses[i, :epochs_completed] = history['val_loss']
            
            # Fill the rest with the last value (for folds that stopped early)
            if epochs_completed < max_epochs:
                val_accuracies[i, epochs_completed:] = history['val_accuracy'][-1]
                val_losses[i, epochs_completed:] = history['val_loss'][-1]
        
        # Calculate mean performance across folds for each epoch
        mean_val_accuracies = np.mean(val_accuracies, axis=0)
        mean_val_losses = np.mean(val_losses, axis=0)
        
        # Determine optimal epochs based on validation accuracy
        if self.config.metric_goal == 'max':
            optimal_epoch = np.argmax(mean_val_accuracies) + 1  # +1 because epochs are 1-indexed
        else:  # minimize
            optimal_epoch = np.argmin(mean_val_losses) + 1
            
        # Save results to CSV
        results_df = pd.DataFrame({
            'epoch': list(range(1, max_epochs + 1)),
            'mean_val_accuracy': mean_val_accuracies,
            'mean_val_loss': mean_val_losses
        })
        results_df.to_csv(os.path.join(self.config.output_dir, 'epoch_analysis.csv'), index=False)
        
        logger.info(f"Optimal number of epochs determined: {optimal_epoch}")
        
        # Also save a simple text file with just the optimal epochs
        with open(os.path.join(self.config.output_dir, 'optimal_epochs.txt'), 'w') as f:
            f.write(str(optimal_epoch))
        
        return optimal_epoch
    
    def _generate_plots(self, fold_histories: List[Dict[str, List]]):
        """
        Generate and save plots for the learning curves.
        
        Args:
            fold_histories: List of training histories for each fold
        """
        # Create figure for accuracy
        plt.figure(figsize=(10, 6))
        
        # Plot each fold's validation accuracy
        for i, history in enumerate(fold_histories):
            epochs = list(range(1, len(history['val_accuracy']) + 1))
            plt.plot(epochs, history['val_accuracy'], 'b--', alpha=0.3, label=f'Fold {i+1}' if i == 0 else None)
        
        # Calculate and plot mean validation accuracy
        max_epochs = max([len(h['val_accuracy']) for h in fold_histories])
        mean_val_acc = np.zeros(max_epochs)
        count = np.zeros(max_epochs)
        
        for history in fold_histories:
            for i, acc in enumerate(history['val_accuracy']):
                mean_val_acc[i] += acc
                count[i] += 1
        
        for i in range(max_epochs):
            if count[i] > 0:
                mean_val_acc[i] /= count[i]
        
        epochs = list(range(1, max_epochs + 1))
        plt.plot(epochs, mean_val_acc, 'b-', linewidth=2, label='Mean validation accuracy')
        
        # Mark optimal epochs
        if self.optimal_epochs:
            plt.axvline(x=self.optimal_epochs, color='r', linestyle='--', label=f'Optimal epochs: {self.optimal_epochs}')
        
        plt.title('Validation Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Balanced Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config.output_dir, 'validation_accuracy.png'))
        plt.close()
        
        # Create figure for loss
        plt.figure(figsize=(10, 6))
        
        # Plot each fold's validation loss
        for i, history in enumerate(fold_histories):
            epochs = list(range(1, len(history['val_loss']) + 1))
            plt.plot(epochs, history['val_loss'], 'r--', alpha=0.3, label=f'Fold {i+1}' if i == 0 else None)
        
        # Calculate and plot mean validation loss
        mean_val_loss = np.zeros(max_epochs)
        count = np.zeros(max_epochs)
        
        for history in fold_histories:
            for i, loss in enumerate(history['val_loss']):
                mean_val_loss[i] += loss
                count[i] += 1
        
        for i in range(max_epochs):
            if count[i] > 0:
                mean_val_loss[i] /= count[i]
        
        plt.plot(epochs, mean_val_loss, 'r-', linewidth=2, label='Mean validation loss')
        
        # Mark optimal epochs
        if self.optimal_epochs:
            plt.axvline(x=self.optimal_epochs, color='b', linestyle='--', label=f'Optimal epochs: {self.optimal_epochs}')
        
        plt.title('Validation Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.config.output_dir, 'validation_loss.png'))
        plt.close()
    
    def run(self) -> int:
        """
        Run the epoch finder process end-to-end.
        
        Returns:
            Optimal number of epochs for training
        """
        # Set up data
        train_dataset, _ = self.setup_data()
        
        # Run k-fold cross-validation
        optimal_epochs = self.run_kfold(train_dataset)
        
        logger.info(f"Epoch finding complete. Optimal epochs: {optimal_epochs}")
        
        return optimal_epochs


def run_epoch_finder(base_config: MergeExperimentConfig, 
                     num_folds: int = 5, 
                     max_epochs: int = 100, 
                     patience: int = 5,
                     output_dir: str = "./epoch_finder_results") -> int:
    """
    Convenience function to run the epoch finder.
    
    Args:
        base_config: Base configuration for the merge experiment
        num_folds: Number of folds for cross-validation
        max_epochs: Maximum number of epochs to train for
        patience: Early stopping patience
        output_dir: Directory to save results
        
    Returns:
        Optimal number of epochs for training
    """
    # Create epoch finder config
    finder_config = EpochFinderConfig(
        base_config=base_config,
        num_folds=num_folds,
        max_epochs=max_epochs,
        patience=patience,
        output_dir=output_dir
    )
    
    # Create and run epoch finder
    finder = EpochFinder(finder_config)
    optimal_epochs = finder.run()
    
    return optimal_epochs


if __name__ == "__main__":
    # Example usage
    from c2m3.experiments.merge_experiment import MergeExperimentConfig
    
    # Create a base configuration
    base_config = MergeExperimentConfig(
        experiment_name="epoch_finder_test",
        dataset_name="cifar10",
        model_name="resnet18",
    )
    
    # Run epoch finder
    optimal_epochs = run_epoch_finder(
        base_config=base_config,
        num_folds=5,
        max_epochs=50,
        output_dir="./epoch_finder_results"
    )
    
    print(f"Optimal number of epochs: {optimal_epochs}") 