"""
Run a merge experiment with optimal epochs determined through k-fold cross-validation.

This script:
1. Runs k-fold cross-validation to determine the optimal number of epochs
2. Uses the determined epochs to run the actual merge experiment
"""

import os
import logging
import argparse
from pathlib import Path

from c2m3.experiments.merge_experiment import MergeExperimentConfig, MergeExperimentRunner
from c2m3.experiments.epoch_finder import run_epoch_finder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run merge experiments with optimal epochs")
    
    # Dataset and model
    parser.add_argument("--dataset", type=str, default="cifar10", 
                        choices=["femnist", "cifar10", "cifar100", "shakespeare"],
                        help="Dataset name")
    parser.add_argument("--model", type=str, default="resnet18", 
                        choices=["resnet18", "cnn", "lstm"],
                        help="Model name")
    
    # Experiment configuration
    parser.add_argument("--experiment-name", type=str, default="optimal_epochs_experiment",
                        help="Name of the experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-models", type=int, default=3, 
                        help="Number of models to train and merge")
    parser.add_argument("--data-distribution", type=str, default="iid",
                        choices=["iid", "dirichlet", "pathological", "natural"],
                        help="Data distribution type")
    
    # K-fold configuration
    parser.add_argument("--num-folds", type=int, default=5, 
                        help="Number of folds for cross-validation")
    parser.add_argument("--max-epochs", type=int, default=100, 
                        help="Maximum number of epochs for k-fold training")
    parser.add_argument("--patience", type=int, default=5, 
                        help="Early stopping patience for k-fold training")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--finder-output-dir", type=str, default=None,
                        help="Directory to save epoch finder results (default: inside output_dir)")
    
    # Skip epoch finding if already done
    parser.add_argument("--skip-epoch-finding", action="store_true",
                        help="Skip epoch finding and use provided epochs")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Use this many epochs (only if skip-epoch-finding is True)")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    finder_output_dir = args.finder_output_dir
    if finder_output_dir is None:
        finder_output_dir = os.path.join(args.output_dir, "epoch_finder")
    os.makedirs(finder_output_dir, exist_ok=True)
    
    # Create base configuration for both epoch finder and merge experiment
    base_config = MergeExperimentConfig(
        experiment_name=args.experiment_name,
        dataset_name=args.dataset,
        model_name=args.model,
        seed=args.seed,
        num_models=args.num_models,
        data_distribution=args.data_distribution,
        output_dir=args.output_dir
    )
    
    # Step 1: Determine optimal epochs through k-fold cross-validation
    optimal_epochs = None
    if not args.skip_epoch_finding:
        logger.info("Running k-fold cross-validation to determine optimal epochs...")
        optimal_epochs = run_epoch_finder(
            base_config=base_config,
            num_folds=args.num_folds,
            max_epochs=args.max_epochs,
            patience=args.patience,
            output_dir=finder_output_dir
        )
    else:
        if args.epochs is None:
            # Try to read from previous run
            epochs_file = os.path.join(finder_output_dir, 'optimal_epochs.txt')
            if os.path.exists(epochs_file):
                with open(epochs_file, 'r') as f:
                    optimal_epochs = int(f.read().strip())
                logger.info(f"Using optimal epochs from file: {optimal_epochs}")
            else:
                raise ValueError("--skip-epoch-finding requires either --epochs or an existing optimal_epochs.txt file")
        else:
            optimal_epochs = args.epochs
            logger.info(f"Using provided epochs: {optimal_epochs}")
    
    # Step 2: Run merge experiment with determined epochs
    logger.info(f"Running merge experiment with {optimal_epochs} epochs per model...")
    
    # Update configuration with determined epochs
    experiment_config = MergeExperimentConfig(
        experiment_name=f"{args.experiment_name}_epochs_{optimal_epochs}",
        dataset_name=args.dataset,
        model_name=args.model,
        seed=args.seed,
        num_models=args.num_models,
        epochs_per_model=optimal_epochs,
        data_distribution=args.data_distribution,
        output_dir=args.output_dir,
        early_stopping=False  # Disable early stopping since we're using fixed epochs
    )
    
    # Run the experiment
    experiment_runner = MergeExperimentRunner(experiment_config)
    experiment_runner.setup()
    experiment_runner.run()
    experiment_runner.visualize_results()
    
    logger.info(f"Experiment complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main() 