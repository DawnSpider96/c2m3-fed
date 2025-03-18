#!/usr/bin/env python
"""
Script to run model merging experiments with C2M3, FedAvg, and TIES.
This script can be used to conduct the experiments required for the project.
"""

import argparse
import logging
from pathlib import Path
import os
import sys

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from c2m3.experiments.merge_experiment import (
    MergeExperimentConfig,
    MergeExperimentRunner,
    run_parameter_sweep
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_model_initialization_experiments(seed=42):
    """
    Run experiments for Model Initialization Consistency:
    - Train models on IID data for an increasing number of epochs/steps and merge them
    - Repeat with non-IID data distributions
    - Compare with standard methods like model averaging and TIES
    """
    logger.info("Running Model Initialization Consistency experiments")
    
    # Base configuration
    base_config = MergeExperimentConfig(
        experiment_name="model_init_consistency",
        dataset_name="cifar10",
        model_name="resnet18",
        num_models=3,
        batch_size=64,
        learning_rate=0.01,
        initialization_type="identical",
        merging_methods=["c2m3", "fedavg", "ties"],
        save_results=True,
        seed=seed
    )
    
    # Define parameter grid for sweep
    parameter_grid = {
        "epochs_per_model": [[5, 5, 5], [10, 10, 10], [20, 20, 20], [30, 30, 30]],
        "data_distribution": ["iid", "dirichlet", "pathological"],
        "non_iid_alpha": [0.5, 0.1],  # More heterogeneous with lower alpha
    }
    
    # Run parameter sweep
    results = run_parameter_sweep(base_config, parameter_grid)
    
    return results

def run_varying_initializations_experiments(seed=42):
    """
    Run experiments for Varying Initializations:
    - Train models starting with different random initializations
    - Merge models trained on IID and non-IID data distributions
    - Compare performance with identical initialization experiments
    """
    logger.info("Running Varying Initializations experiments")
    
    # Base configuration
    base_config = MergeExperimentConfig(
        experiment_name="varying_initializations",
        dataset_name="cifar10",
        model_name="resnet18",
        num_models=3,
        epochs_per_model=15,
        batch_size=64,
        learning_rate=0.01,
        initialization_type="diverse",  # Use diverse initializations
        merging_methods=["c2m3", "fedavg", "ties"],
        save_results=True,
        seed=seed
    )
    
    # Define parameter grid for sweep
    parameter_grid = {
        "data_distribution": ["iid", "dirichlet", "pathological"],
        "non_iid_alpha": [0.5, 0.1],  # Test with different levels of heterogeneity
        "classes_per_partition": [1, 2]  # For pathological partitioning
    }
    
    # Run parameter sweep
    results = run_parameter_sweep(base_config, parameter_grid)
    
    return results

def run_single_experiment(args):
    """Run a single experiment with provided arguments"""
    logger.info(f"Running single experiment: {args.experiment_name}")
    
    # Create configuration
    config = MergeExperimentConfig(
        experiment_name=args.experiment_name,
        dataset_name=args.dataset,
        model_name=args.model,
        num_models=args.num_models,
        epochs_per_model=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        initialization_type=args.initialization,
        data_distribution=args.data_distribution,
        non_iid_alpha=args.non_iid_alpha,
        classes_per_partition=args.classes_per_partition,
        merging_methods=args.merging_methods.split(','),
        c2m3_max_iter=args.c2m3_max_iter,
        c2m3_score_tolerance=args.c2m3_score_tolerance,
        ties_alpha=args.ties_alpha,
        output_dir=args.output_dir,
        save_models=args.save_models,
        save_results=True,
        seed=args.seed
    )
    
    # Run experiment
    runner = MergeExperimentRunner(config)
    runner.setup()
    results = runner.run()
    runner.visualize_results()
    
    return results

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run model merging experiments")
    
    # Experiment type
    parser.add_argument('--run-type', type=str, default='single',
                        choices=['single', 'model_init', 'varying_init'],
                        help='Type of experiment to run')
    
    # Basic experiment settings
    parser.add_argument('--experiment-name', type=str, default='experiment',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['femnist', 'cifar10', 'cifar100', 'shakespeare'],
                        help='Dataset to use')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model architecture to use')
    
    # Training configuration
    parser.add_argument('--num-models', type=int, default=3,
                        help='Number of models to train and merge')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs to train each model')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate for training')
    
    # Data distribution
    parser.add_argument('--data-distribution', type=str, default='iid',
                        choices=['iid', 'dirichlet', 'pathological', 'natural', 'by_writer', 'by_character'],
                        help='Data distribution type')
    parser.add_argument('--non-iid-alpha', type=float, default=0.5,
                        help='Dirichlet alpha parameter for non-IID distribution')
    parser.add_argument('--classes-per-partition', type=int, default=1,
                        help='Number of classes per partition for pathological partitioning')
    
    # Initialization
    parser.add_argument('--initialization', type=str, default='identical',
                        choices=['identical', 'diverse'],
                        help='Model initialization type')
    
    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (controls all random operations)')
    
    # Merging configuration
    parser.add_argument('--merging-methods', type=str, default='c2m3,fedavg,ties',
                        help='Comma-separated list of merging methods to use')
    parser.add_argument('--c2m3-max-iter', type=int, default=100,
                        help='Maximum iterations for C2M3 Frank-Wolfe algorithm')
    parser.add_argument('--c2m3-score-tolerance', type=float, default=1e-6,
                        help='Convergence threshold for C2M3 Frank-Wolfe algorithm')
    parser.add_argument('--ties-alpha', type=float, default=0.5,
                        help='Interpolation parameter for TIES merging')
    
    # Output configuration
    parser.add_argument('--output-dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--save-models', action='store_true',
                        help='Whether to save the trained and merged models')
    
    return parser.parse_args()

def main():
    """Main function to run experiments"""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    # Run experiments based on type
    if args.run_type == 'single':
        results = run_single_experiment(args)
    elif args.run_type == 'model_init':
        results = run_model_initialization_experiments(args.seed)
    elif args.run_type == 'varying_init':
        results = run_varying_initializations_experiments(args.seed)
    else:
        raise ValueError(f"Unknown run type: {args.run_type}")
    
    logger.info("Experiments completed successfully")
    
    return results

if __name__ == "__main__":
    main() 