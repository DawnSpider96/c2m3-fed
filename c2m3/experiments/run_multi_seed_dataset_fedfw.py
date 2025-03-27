#!/usr/bin/env python
"""
Script to run multi-seed learning rate experiments using the DatasetFrankWolfeSync strategy.
This script will run experiments with multiple random seeds and learning rates
to evaluate the robustness of the federated learning approach.
"""
import argparse
import logging
from pathlib import Path

from c2m3.experiments.federated_experiment import (
    FederatedExperimentConfig, 
    run_multi_seed_learning_rates_experiment
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run multi-seed learning rate experiments with DatasetFrankWolfeSync')
    
    # Required arguments
    parser.add_argument('--experiment-name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='femnist', choices=['femnist'], 
                        help='Dataset name (only femnist supported for DatasetFrankWolfeSync)')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn'], 
                        help='Model name (only CNN supported for FEMNIST)')
    
    # Optional arguments
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 101], 
                        help='Random seeds to use')
    parser.add_argument('--data-dir', type=str, default=None, help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./results/multi_seed_lr', help='Output directory')
    
    # Training configuration
    parser.add_argument('--num-rounds', type=int, default=10, help='Number of federation rounds')
    parser.add_argument('--local-epochs', type=int, default=1, help='Number of local epochs per round')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rates', type=float, nargs='+', default=[0.001, 0.01, 0.1], 
                        help='Learning rates to evaluate')
    parser.add_argument('--weight-decay', type=float, default=0.0001, help='Weight decay')
    
    # Client configuration
    parser.add_argument('--num-total-clients', type=int, default=20, help='Total number of available clients')
    parser.add_argument('--num-clients-per-round', type=int, default=5, help='Number of clients sampled each round')
    parser.add_argument('--min-train-samples', type=int, default=10, help='Minimum number of training samples per client')
    parser.add_argument('--initialization-type', type=str, default='identical', 
                        choices=['identical', 'random'], help='How to initialize client models')
    
    # Data distribution
    parser.add_argument('--data-distribution', type=str, default='natural', 
                        choices=['iid', 'dirichlet', 'pathological', 'natural'],
                        help='How to distribute data among clients')
    parser.add_argument('--non-iid-alpha', type=float, default=0.5, 
                        help='Dirichlet alpha parameter for non-IID distribution')
    
    # Output options
    parser.add_argument('--save-models', action='store_true', help='Whether to save models')
    
    return parser.parse_args()

def main():
    """Run multi-seed learning rate experiments with DatasetFrankWolfeSync."""
    args = parse_args()
    
    # Set data directory if not provided
    if args.data_dir is None:
        args.data_dir = str(Path(__file__).parent.parent / "data")
    
    # Create base configuration
    base_config = FederatedExperimentConfig(
        experiment_name=args.experiment_name,
        dataset_name=args.dataset,
        model_name=args.model,
        seed=args.seeds[0],  # This will be overridden in multi-seed experiment
        data_dir=args.data_dir,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rates[0],  # This will be overridden in multi-LR experiment
        weight_decay=args.weight_decay,
        num_total_clients=args.num_total_clients,
        num_clients_per_round=args.num_clients_per_round,
        min_train_samples=args.min_train_samples,
        initialization_type=args.initialization_type,
        data_distribution=args.data_distribution,
        non_iid_alpha=args.non_iid_alpha,
        output_dir=args.output_dir,
        save_models=args.save_models
    )
    
    # Run experiments with multiple seeds and learning rates
    logger.info(f"Running multi-seed learning rate experiments for {args.experiment_name}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Learning rates: {args.learning_rates}")
    
    results = run_multi_seed_learning_rates_experiment(
        base_config=base_config,
        learning_rates=args.learning_rates,
        seeds=args.seeds,
        output_dir=args.output_dir
    )
    
    logger.info(f"Multi-seed learning rate experiments for {args.experiment_name} completed successfully.")
    
if __name__ == "__main__":
    main() 