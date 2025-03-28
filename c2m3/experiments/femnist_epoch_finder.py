"""
K-fold cross-validation for determining optimal number of epochs for FEMNIST.

This module uses k-fold cross-validation to determine the optimal number of epochs
specifically for FEMNIST dataset training, which will then be used in merge experiments.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import KFold

from c2m3.experiments.merge_experiment import MergeExperimentConfig, ModelTrainer, DataManager
from c2m3.common.client_utils import get_network_generator_cnn
from c2m3.models.cnn import Net

logger = logging.getLogger(__name__)

@dataclass
class FEMNISTEpochFinderConfig:
    """Configuration for k-fold cross-validation to find optimal epochs for FEMNIST"""
    
    experiment_name: str = "femnist_epoch_finder"
    
    
    data_dir: str = str(Path(__file__).parent.parent / "data")
    data_distribution: str = "iid"  
    samples_per_partition: int = 4000  
    
    
    model_name: str = "cnn"  
    
    
    batch_size: int = 64
    eval_batch_size: int = 128
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    
    
    num_folds: int = 10  
    max_epochs: int = 100  
    patience: int = 10  
    min_delta: float = 0.001  
    seed: int = 42  
    
    
    output_dir: str = "./femnist_epoch_finder_results"  
    save_plots: bool = True  
    
    
    early_stopping_metric: str = "balanced_accuracy"  
    metric_goal: str = "max"  
    
    def to_merge_experiment_config(self, epochs_per_model: Optional[int] = None) -> MergeExperimentConfig:
        """
        Convert to a MergeExperimentConfig for use in merge experiments.
        
        Args:
            epochs_per_model: Optional value to override epochs_per_model.
                If not provided, this will be left as default in MergeExperimentConfig.
                
        Returns:
            MergeExperimentConfig instance
        """
        config_dict = {
            "experiment_name": self.experiment_name,
            "dataset_name": "femnist",  
            "model_name": self.model_name,
            "seed": self.seed,
            "data_dir": self.data_dir,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "data_distribution": self.data_distribution,
            "samples_per_partition": self.samples_per_partition,
            "eval_batch_size": self.eval_batch_size,
            "output_dir": self.output_dir,
        }
        
        if epochs_per_model is not None:
            config_dict["epochs_per_model"] = epochs_per_model
            
            config_dict["early_stopping"] = False
            
        return MergeExperimentConfig(**config_dict)


class FEMNISTEpochFinder:
    """
    Use k-fold cross-validation to find the optimal number of epochs for FEMNIST training.
    
    This class performs k-fold cross-validation on the FEMNIST dataset using the same model 
    and training setup as the merge experiments, and determines the optimal number of epochs
    based on the validation performance across folds.
    """
    
    def __init__(self, config: FEMNISTEpochFinderConfig):
        """
        Initialize the FEMNISTEpochFinder.
        
        Args:
            config: Configuration for k-fold cross-validation
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        os.makedirs(config.output_dir, exist_ok=True)
        
        
        self.fold_histories = []
        self.optimal_epochs = None
        
    def setup_data(self):
        """
        Set up the FEMNIST dataset for k-fold cross-validation.
        
        Returns:
            Tuple of (train_dataset, test_dataset) for FEMNIST
        """
        
        base_config = self.config.to_merge_experiment_config()
        
        
        data_manager = DataManager(base_config)
        
        
        logger.info("Loading FEMNIST dataset")
        data_refs = data_manager.get_dataset()
        
        
        
        num_partitions = base_config.num_models
        train_partitions, val_partitions, test_partitions = data_manager.partition_data(
            data_refs, num_partitions
        )
        
        
        
        from torch.utils.data import ConcatDataset
        
        
        train_dataset = ConcatDataset(train_partitions)
        
        
        if len(test_partitions) > 1:
            
            test_dataset = ConcatDataset(test_partitions)
            logger.info(f"Combined {len(test_partitions)} test partitions into one dataset with {len(test_dataset)} samples")
        else:
            
            test_dataset = test_partitions[0]
            logger.info(f"Using full test dataset with {len(test_dataset)} samples")
        
        logger.info(f"Combined train dataset size: {len(train_dataset)}")
        
        return train_dataset, test_dataset
    
    def run_kfold(self, dataset):
        """
        Run k-fold cross-validation to find optimal epochs for FEMNIST.
        
        Args:
            dataset: FEMNIST dataset to use for k-fold cross-validation
            
        Returns:
            Optimal number of epochs for training
        """
        logger.info(f"Running {self.config.num_folds}-fold cross-validation on FEMNIST")
        
        
        kfold = KFold(n_splits=self.config.num_folds, shuffle=True, random_state=self.config.seed)
        
        
        indices = list(range(len(dataset)))
        
        
        fold_histories = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
            logger.info(f"Training fold {fold+1}/{self.config.num_folds}")
            
            
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            
            train_loader = DataLoader(
                train_subset, 
                batch_size=self.config.batch_size, 
                shuffle=True,
                drop_last=True
            )
            val_loader = DataLoader(
                val_subset, 
                batch_size=self.config.eval_batch_size, 
                shuffle=False,
                drop_last=False
            )
            
            
            network_generator = get_network_generator_cnn()
            model = network_generator().to(self.device)
            
            
            trainer = ModelTrainer(
                model=model,
                train_loader=train_loader,
                test_loader=None,  
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                device=self.device,
                dataset_name="femnist",  
                val_loader=val_loader,
                patience=self.config.patience,
                min_delta=self.config.min_delta
            )
            
            
            history = self._train_with_history(trainer)
            fold_histories.append(history)
            
            logger.info(f"Fold {fold+1} complete. Best epoch: {len(history['train_loss'])}")
        
        self.fold_histories = fold_histories
        
        
        optimal_epochs = self._analyze_fold_results(fold_histories)
        self.optimal_epochs = optimal_epochs
        
        
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
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'val_balanced_accuracy': []
        }
        
        early_stopped = False
        best_model_state = None
        best_val_metric = float('-inf') if self.config.metric_goal == 'max' else float('inf')
        patience_counter = 0
        
        
        for epoch in range(self.config.max_epochs):
            
            train_loss, train_acc = trainer.train_epoch()
            
            
            val_loss, val_acc, val_balanced_acc = trainer.evaluate_validation()
            
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)
            history['val_balanced_accuracy'].append(val_balanced_acc)
            
            
            logger.info(f"Epoch {epoch+1}/{self.config.max_epochs}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}")
            
            
            if self.config.early_stopping_metric == 'balanced_accuracy':
                val_metric = val_balanced_acc
            elif self.config.early_stopping_metric == 'accuracy':
                val_metric = val_acc
            else:  
                val_metric = val_balanced_acc
            
            
            if self.config.metric_goal == 'max':
                improved = val_metric > best_val_metric + self.config.min_delta
            else:  
                improved = val_metric < best_val_metric - self.config.min_delta
            
            
            if improved:
                best_val_metric = val_metric
                best_model_state = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}
                patience_counter = 0
                logger.info(f"Epoch {epoch+1}: New best model with {self.config.early_stopping_metric} = {val_metric:.4f}")
            else:
                patience_counter += 1
                logger.info(f"Epoch {epoch+1}: No improvement for {patience_counter}/{self.config.patience} epochs")
            
            
            if patience_counter >= self.config.patience:
                logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {self.config.patience} epochs)")
                early_stopped = True
                break
        
        
        if early_stopped and best_model_state is not None:
            trainer.model.load_state_dict(best_model_state)
            logger.info("Restored model to best validation performance")
            
            
            val_loss, val_acc, val_balanced_acc = trainer.evaluate_validation()
            logger.info(f"Best model - Val Acc: {val_acc:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}")
        
        
        history['early_stopped'] = early_stopped
        history['best_val_metric'] = best_val_metric
        history['epochs_trained'] = epoch + 1
        
        return history
    
    def _analyze_fold_results(self, fold_histories: List[Dict[str, List]]) -> int:
        """
        Analyze results from all folds to determine optimal epochs.
        
        Args:
            fold_histories: List of training histories for each fold
            
        Returns:
            Optimal number of epochs for training
        """
        
        max_epochs = max([h['epochs_trained'] for h in fold_histories])
        logger.info(f"Maximum epochs trained across folds: {max_epochs}")
        
        
        val_accuracies = np.zeros((len(fold_histories), max_epochs))
        val_balanced_accuracies = np.zeros((len(fold_histories), max_epochs))
        val_losses = np.zeros((len(fold_histories), max_epochs))
        
        
        best_epochs = []
        
        
        for i, history in enumerate(fold_histories):
            epochs_completed = history['epochs_trained']
            logger.info(f"Fold {i+1}: Trained for {epochs_completed} epochs")
            
            
            if self.config.metric_goal == 'max':
                if self.config.early_stopping_metric == 'balanced_accuracy':
                    best_epoch = np.argmax(history['val_balanced_accuracy']) + 1  
                else:  
                    best_epoch = np.argmax(history['val_accuracy']) + 1
            else:  
                best_epoch = np.argmin(history['val_loss']) + 1
            
            best_epochs.append(best_epoch)
            logger.info(f"Fold {i+1}: Best epoch = {best_epoch}")
            
            
            val_accuracies[i, :len(history['val_accuracy'])] = history['val_accuracy']
            val_balanced_accuracies[i, :len(history['val_balanced_accuracy'])] = history['val_balanced_accuracy']
            val_losses[i, :len(history['val_loss'])] = history['val_loss']
            
            
            if len(history['val_accuracy']) < max_epochs:
                val_accuracies[i, len(history['val_accuracy']):] = history['val_accuracy'][-1]
                val_balanced_accuracies[i, len(history['val_balanced_accuracy']):] = history['val_balanced_accuracy'][-1]
                val_losses[i, len(history['val_loss']):] = history['val_loss'][-1]
        
        
        mean_val_accuracies = np.mean(val_accuracies, axis=0)
        mean_val_balanced_accuracies = np.mean(val_balanced_accuracies, axis=0)
        mean_val_losses = np.mean(val_losses, axis=0)
        
        
        std_val_accuracies = np.std(val_accuracies, axis=0)
        std_val_balanced_accuracies = np.std(val_balanced_accuracies, axis=0)
        std_val_losses = np.std(val_losses, axis=0)
        
        
        if self.config.metric_goal == 'max':
            if self.config.early_stopping_metric == 'balanced_accuracy':
                optimal_epoch = np.argmax(mean_val_balanced_accuracies) + 1  
                best_value = mean_val_balanced_accuracies[optimal_epoch - 1]
                logger.info(f"Optimal epoch {optimal_epoch} based on max mean balanced accuracy: {best_value:.4f}")
            else:  
                optimal_epoch = np.argmax(mean_val_accuracies) + 1
                best_value = mean_val_accuracies[optimal_epoch - 1]
                logger.info(f"Optimal epoch {optimal_epoch} based on max mean accuracy: {best_value:.4f}")
        else:  
            optimal_epoch = np.argmin(mean_val_losses) + 1
            best_value = mean_val_losses[optimal_epoch - 1]
            logger.info(f"Optimal epoch {optimal_epoch} based on min mean loss: {best_value:.4f}")
        
        
        median_best_epoch = int(np.median(best_epochs))
        logger.info(f"Median of best epochs across folds: {median_best_epoch}")
        
        
        if abs(median_best_epoch - optimal_epoch) <= 3:
            final_optimal_epoch = median_best_epoch
            logger.info(f"Using median best epoch {median_best_epoch} as it's close to optimal epoch {optimal_epoch}")
        else:
            final_optimal_epoch = optimal_epoch
            logger.info(f"Using optimal epoch {optimal_epoch} based on mean performance")
            
        
        results_df = pd.DataFrame({
            'epoch': list(range(1, max_epochs + 1)),
            'mean_val_accuracy': mean_val_accuracies,
            'std_val_accuracy': std_val_accuracies,
            'mean_val_balanced_accuracy': mean_val_balanced_accuracies,
            'std_val_balanced_accuracy': std_val_balanced_accuracies,
            'mean_val_loss': mean_val_losses,
            'std_val_loss': std_val_losses
        })
        results_df.to_csv(os.path.join(self.config.output_dir, 'epoch_analysis.csv'), index=False)
        
        logger.info(f"Optimal number of epochs determined: {final_optimal_epoch}")
        
        
        with open(os.path.join(self.config.output_dir, 'optimal_epochs.txt'), 'w') as f:
            f.write(str(final_optimal_epoch))
        
        return final_optimal_epoch
    
    def _generate_plots(self, fold_histories: List[Dict[str, List]]):
        """
        Generate and save plots for the learning curves.
        
        Args:
            fold_histories: List of training histories for each fold
        """
        
        max_epochs = max([h['epochs_trained'] for h in fold_histories])
        epochs = list(range(1, max_epochs + 1))
        
        
        plt.figure(figsize=(12, 7))
        
        
        for i, history in enumerate(fold_histories):
            fold_epochs = list(range(1, len(history['val_balanced_accuracy']) + 1))
            plt.plot(fold_epochs, history['val_balanced_accuracy'], 'b--', alpha=0.3, 
                     label=f'Fold {i+1}' if i == 0 else None)
        
        
        mean_val_balanced_acc = np.zeros(max_epochs)
        std_val_balanced_acc = np.zeros(max_epochs)
        count = np.zeros(max_epochs)
        
        for history in fold_histories:
            for i, acc in enumerate(history['val_balanced_accuracy']):
                mean_val_balanced_acc[i] += acc
                count[i] += 1
        
        for i in range(max_epochs):
            if count[i] > 0:
                mean_val_balanced_acc[i] /= count[i]
        
        
        for history in fold_histories:
            for i, acc in enumerate(history['val_balanced_accuracy']):
                if i < max_epochs and count[i] > 0:
                    std_val_balanced_acc[i] += (acc - mean_val_balanced_acc[i]) ** 2 / count[i]
        
        std_val_balanced_acc = np.sqrt(std_val_balanced_acc)
        
        
        plt.plot(epochs, mean_val_balanced_acc, 'b-', linewidth=2, label='Mean validation balanced accuracy')
        plt.fill_between(epochs, mean_val_balanced_acc - std_val_balanced_acc, 
                         mean_val_balanced_acc + std_val_balanced_acc, color='b', alpha=0.2)
        
        
        if self.optimal_epochs:
            plt.axvline(x=self.optimal_epochs, color='r', linestyle='--', 
                        label=f'Optimal epochs: {self.optimal_epochs}')
            
            
            optimal_value = mean_val_balanced_acc[self.optimal_epochs - 1]
            plt.plot([self.optimal_epochs], [optimal_value], 'ro', markersize=8)
            plt.annotate(f'({self.optimal_epochs}, {optimal_value:.4f})', 
                        xy=(self.optimal_epochs, optimal_value),
                        xytext=(self.optimal_epochs + 1, optimal_value),
                        fontsize=10)
        
        plt.title('FEMNIST Validation Balanced Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Balanced Accuracy')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'validation_balanced_accuracy.png'))
        plt.close()
        
        
        plt.figure(figsize=(12, 7))
        
        
        for i, history in enumerate(fold_histories):
            fold_epochs = list(range(1, len(history['val_accuracy']) + 1))
            plt.plot(fold_epochs, history['val_accuracy'], 'g--', alpha=0.3, 
                     label=f'Fold {i+1}' if i == 0 else None)
        
        
        mean_val_acc = np.zeros(max_epochs)
        std_val_acc = np.zeros(max_epochs)
        count = np.zeros(max_epochs)
        
        for history in fold_histories:
            for i, acc in enumerate(history['val_accuracy']):
                mean_val_acc[i] += acc
                count[i] += 1
        
        for i in range(max_epochs):
            if count[i] > 0:
                mean_val_acc[i] /= count[i]
        
        
        for history in fold_histories:
            for i, acc in enumerate(history['val_accuracy']):
                if i < max_epochs and count[i] > 0:
                    std_val_acc[i] += (acc - mean_val_acc[i]) ** 2 / count[i]
        
        std_val_acc = np.sqrt(std_val_acc)
        
        
        plt.plot(epochs, mean_val_acc, 'g-', linewidth=2, label='Mean validation accuracy')
        plt.fill_between(epochs, mean_val_acc - std_val_acc, 
                         mean_val_acc + std_val_acc, color='g', alpha=0.2)
        
        
        if self.optimal_epochs:
            plt.axvline(x=self.optimal_epochs, color='r', linestyle='--', 
                        label=f'Optimal epochs: {self.optimal_epochs}')
            
            
            optimal_value = mean_val_acc[self.optimal_epochs - 1]
            plt.plot([self.optimal_epochs], [optimal_value], 'ro', markersize=8)
            plt.annotate(f'({self.optimal_epochs}, {optimal_value:.4f})', 
                        xy=(self.optimal_epochs, optimal_value),
                        xytext=(self.optimal_epochs + 1, optimal_value),
                        fontsize=10)
        
        plt.title('FEMNIST Validation Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'validation_accuracy.png'))
        plt.close()
        
        
        plt.figure(figsize=(12, 7))
        
        
        for i, history in enumerate(fold_histories):
            fold_epochs = list(range(1, len(history['val_loss']) + 1))
            plt.plot(fold_epochs, history['val_loss'], 'r--', alpha=0.3, 
                     label=f'Fold {i+1}' if i == 0 else None)
        
        
        mean_val_loss = np.zeros(max_epochs)
        std_val_loss = np.zeros(max_epochs)
        count = np.zeros(max_epochs)
        
        for history in fold_histories:
            for i, loss in enumerate(history['val_loss']):
                mean_val_loss[i] += loss
                count[i] += 1
        
        for i in range(max_epochs):
            if count[i] > 0:
                mean_val_loss[i] /= count[i]
        
        
        for history in fold_histories:
            for i, loss in enumerate(history['val_loss']):
                if i < max_epochs and count[i] > 0:
                    std_val_loss[i] += (loss - mean_val_loss[i]) ** 2 / count[i]
        
        std_val_loss = np.sqrt(std_val_loss)
        
        
        plt.plot(epochs, mean_val_loss, 'r-', linewidth=2, label='Mean validation loss')
        plt.fill_between(epochs, mean_val_loss - std_val_loss, 
                         mean_val_loss + std_val_loss, color='r', alpha=0.2)
        
        
        if self.optimal_epochs:
            plt.axvline(x=self.optimal_epochs, color='b', linestyle='--', 
                        label=f'Optimal epochs: {self.optimal_epochs}')
            
            
            optimal_value = mean_val_loss[self.optimal_epochs - 1]
            plt.plot([self.optimal_epochs], [optimal_value], 'bo', markersize=8)
            plt.annotate(f'({self.optimal_epochs}, {optimal_value:.4f})', 
                        xy=(self.optimal_epochs, optimal_value),
                        xytext=(self.optimal_epochs + 1, optimal_value),
                        fontsize=10)
        
        plt.title('FEMNIST Validation Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'validation_loss.png'))
        plt.close()
        
        
        plt.figure(figsize=(14, 10))
        
        
        ax1 = plt.subplot(3, 1, 1)
        ax2 = plt.subplot(3, 1, 2)
        ax3 = plt.subplot(3, 1, 3)
        
        
        ax1.plot(epochs, mean_val_balanced_acc, 'b-', linewidth=2, label='Balanced Accuracy')
        ax1.fill_between(epochs, mean_val_balanced_acc - std_val_balanced_acc, 
                        mean_val_balanced_acc + std_val_balanced_acc, color='b', alpha=0.2)
        
        
        ax2.plot(epochs, mean_val_acc, 'g-', linewidth=2, label='Accuracy')
        ax2.fill_between(epochs, mean_val_acc - std_val_acc,
                         mean_val_acc + std_val_acc, color='g', alpha=0.2)
        
        
        ax3.plot(epochs, mean_val_loss, 'r-', linewidth=2, label='Loss')
        ax3.fill_between(epochs, mean_val_loss - std_val_loss,
                         mean_val_loss + std_val_loss, color='r', alpha=0.2)
        
        
        if self.optimal_epochs:
            for ax in [ax1, ax2, ax3]:
                ax.axvline(x=self.optimal_epochs, color='k', linestyle='--',
                           label=f'Optimal epochs: {self.optimal_epochs}')
        
        
        ax1.set_title('Validation Balanced Accuracy')
        ax1.set_ylabel('Balanced Accuracy')
        ax1.legend(loc='lower right')
        ax1.grid(True)
        
        ax2.set_title('Validation Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.legend(loc='lower right')
        ax2.grid(True)
        
        ax3.set_title('Validation Loss')
        ax3.set_xlabel('Epochs')
        ax3.set_ylabel('Loss')
        ax3.legend(loc='upper right')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, 'validation_metrics_summary.png'))
        plt.close()
    
    def run(self) -> int:
        """
        Run the FEMNIST epoch finder process end-to-end.
        
        Returns:
            Optimal number of epochs for training
        """
        
        train_dataset, _ = self.setup_data()
        
        
        optimal_epochs = self.run_kfold(train_dataset)
        
        logger.info(f"FEMNIST epoch finding complete. Optimal epochs: {optimal_epochs}")
        
        return optimal_epochs


def run_femnist_epoch_finder(config: Optional[FEMNISTEpochFinderConfig] = None) -> int:
    """
    Convenience function to run the FEMNIST epoch finder.
    
    Args:
        config: Optional configuration for the epoch finder.
            If not provided, a default configuration will be used.
        
    Returns:
        Optimal number of epochs for training
    """
    
    if config is None:
        config = FEMNISTEpochFinderConfig()
    
    
    finder = FEMNISTEpochFinder(config)
    optimal_epochs = finder.run()
    
    return optimal_epochs


def run_experiment_with_optimal_epochs(
    optimal_epochs: int, 
    config: Optional[FEMNISTEpochFinderConfig] = None,
    num_models: int = 3,
    merging_methods: List[str] = None
):
    """
    Run a merge experiment with the determined optimal epochs.
    
    Args:
        optimal_epochs: Number of epochs to train each model
        config: Optional configuration for the experiment
        num_models: Number of models to train and merge
        merging_methods: List of merging methods to use
    """
    from c2m3.experiments.merge_experiment import MergeExperimentRunner
    
    if config is None:
        config = FEMNISTEpochFinderConfig()
    
    
    merge_config = config.to_merge_experiment_config(epochs_per_model=optimal_epochs)
    
    
    merge_config.num_models = num_models
    if merging_methods is not None:
        merge_config.merging_methods = merging_methods
    
    
    merge_config.experiment_name = f"femnist_{merge_config.data_distribution}_epochs_{optimal_epochs}"
    
    
    logger.info(f"Running FEMNIST merge experiment with {optimal_epochs} epochs per model...")
    experiment_runner = MergeExperimentRunner(merge_config)
    experiment_runner.setup()
    experiment_runner.run()
    experiment_runner.visualize_results()
    
    logger.info(f"Experiment complete. Results saved to {merge_config.output_dir}")


if __name__ == "__main__":
    import argparse
    
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    
    parser = argparse.ArgumentParser(description="Find optimal epochs for FEMNIST training")
    
    
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).parent.parent / "data"),
                        help="Directory containing the FEMNIST dataset")
    parser.add_argument("--data-distribution", type=str, default="iid",
                        choices=["iid", "dirichlet", "pathological", "natural"],
                        help="Data distribution type")
    parser.add_argument("--samples-per-partition", type=int, default=4000, 
                        help="Number of samples per partition for IID FEMNIST")
    
    
    parser.add_argument("--num-folds", type=int, default=10, 
                        help="Number of folds for cross-validation")
    parser.add_argument("--max-epochs", type=int, default=100, 
                        help="Maximum number of epochs for k-fold training")
    parser.add_argument("--patience", type=int, default=10, 
                        help="Early stopping patience for k-fold training")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    
    parser.add_argument("--batch-size", type=int, default=64, 
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.01, 
                        help="Learning rate for training")
    
    
    parser.add_argument("--output-dir", type=str, default="./femnist_epoch_finder_results",
                        help="Directory to save results")
    
    
    parser.add_argument("--run-experiment", action="store_true",
                        help="Run merge experiment with optimal epochs")
    parser.add_argument("--num-models", type=int, default=3,
                        help="Number of models to train and merge (only with --run-experiment)")
    
    
    parser.add_argument("--skip-epoch-finding", action="store_true",
                        help="Skip epoch finding and use provided epochs")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Use this many epochs (only if skip-epoch-finding is True)")
    
    args = parser.parse_args()
    
    
    config = FEMNISTEpochFinderConfig(
        data_dir=args.data_dir,
        data_distribution=args.data_distribution,
        samples_per_partition=args.samples_per_partition,
        num_folds=args.num_folds,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    
    optimal_epochs = None
    if not args.skip_epoch_finding:
        optimal_epochs = run_femnist_epoch_finder(config)
    else:
        if args.epochs is None:
            
            epochs_file = os.path.join(args.output_dir, 'optimal_epochs.txt')
            if os.path.exists(epochs_file):
                with open(epochs_file, 'r') as f:
                    optimal_epochs = int(f.read().strip())
                logger.info(f"Using optimal epochs from file: {optimal_epochs}")
            else:
                raise ValueError("--skip-epoch-finding requires either --epochs or an existing optimal_epochs.txt file")
        else:
            optimal_epochs = args.epochs
            logger.info(f"Using provided epochs: {optimal_epochs}")
    
    
    if args.run_experiment:
        run_experiment_with_optimal_epochs(
            optimal_epochs=optimal_epochs,
            config=config,
            num_models=args.num_models
        ) 