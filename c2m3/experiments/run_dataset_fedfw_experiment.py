#!/usr/bin/env python
"""
Script to run federated learning experiments using the DatasetFrankWolfeSync strategy.
This strategy works directly with dataset objects instead of file paths.
"""
import argparse
import logging
from pathlib import Path

from c2m3.experiments.federated_experiment import (
    FederatedExperimentConfig, 
    FederatedExperimentRunner
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run federated learning experiments with DatasetFrankWolfeSync')
    
    # Required arguments
    parser.add_argument('--experiment-name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='femnist', choices=['femnist'], 
                        help='Dataset name (only femnist supported for DatasetFrankWolfeSync)')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn'], 
                        help='Model name (only CNN supported for FEMNIST)')
    
    # Optional arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--data-dir', type=str, default=None, help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    
    # Training configuration
    parser.add_argument('--num-rounds', type=int, default=10, help='Number of federation rounds')
    parser.add_argument('--local-epochs', type=int, default=1, help='Number of local epochs per round')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate')
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
    """Run the federated learning experiment with DatasetFrankWolfeSync."""
    args = parse_args()
    
    # Set data directory if not provided
    if args.data_dir is None:
        args.data_dir = str(Path(__file__).parent.parent / "data")
    
    # Create base configuration
    config = FederatedExperimentConfig(
        experiment_name=args.experiment_name,
        dataset_name=args.dataset,
        model_name=args.model,
        seed=args.seed,
        data_dir=args.data_dir,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
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
    
    # Create and run experiment (uses DatasetFrankWolfeSync automatically for FEMNIST)
    runner = FederatedExperimentRunner(config, use_direct_partitioning=True)
    runner.setup()
    runner.run()
    
    # Visualize results
    runner.visualize_results()
    
    logger.info(f"Experiment {args.experiment_name} completed successfully.")
    
if __name__ == "__main__":
    main() 