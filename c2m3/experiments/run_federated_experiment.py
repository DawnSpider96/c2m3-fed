#!/usr/bin/env python
"""
Script to run federated learning experiments.
"""
import argparse
import logging
from pathlib import Path

from c2m3.experiments.federated_experiment import (
    FederatedExperimentConfig, 
    FederatedExperimentRunner,
    run_learning_rate_experiment,
    run_aggregation_frequency_experiment,
    run_server_lr_experiment,
    run_local_epochs_experiment,
    run_multi_seed_experiment
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run federated learning experiments')
    
    # Required arguments
    parser.add_argument('--experiment-name', type=str, required=True, help='Name of the experiment')
    parser.add_argument('--dataset', type=str, required=True, choices=['femnist'], help='Dataset name')
    parser.add_argument('--model', type=str, required=True, choices=['cnn'], help='Model name')
    
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
    
    # Aggregation configuration
    parser.add_argument('--server-learning-rate', type=float, default=1.0, help='Server learning rate')
    parser.add_argument('--server-momentum', type=float, default=0.0, help='Server momentum')
    parser.add_argument('--aggregation-frequency', type=int, default=1, help='How often to aggregate (every X rounds)')
    
    # Data distribution
    parser.add_argument('--data-distribution', type=str, default='natural', 
                        choices=['iid', 'dirichlet', 'pathological', 'natural'],
                        help='Data distribution type')
    parser.add_argument('--non-iid-alpha', type=float, default=0.5, help='Dirichlet alpha parameter for non-IID distribution')
    
    # Experiment type
    parser.add_argument('--experiment-type', type=str, default='single',
                        choices=['single', 'learning-rate', 'server-lr', 'aggregation-frequency', 'local-epochs', 'multi-seed'],
                        help='Type of experiment to run')
    
    # Parameter sweep values (comma-separated)
    parser.add_argument('--learning-rates', type=str, default='0.001,0.01,0.1', help='Learning rates to try (comma-separated)')
    parser.add_argument('--server-learning-rates', type=str, default='0.1,0.5,1.0', help='Server learning rates to try (comma-separated)')
    parser.add_argument('--aggregation-frequencies', type=str, default='1,2,5', help='Aggregation frequencies to try (comma-separated)')
    parser.add_argument('--local-epoch-values', type=str, default='1,2,5,10', help='Local epoch values to try (comma-separated)')
    parser.add_argument('--seeds', type=str, default='42,123,456,789,101', help='Random seeds to try (comma-separated)')
    
    # Other options
    parser.add_argument('--save-models', action='store_true', help='Save the trained models')
    
    return parser.parse_args()

def main():
    """Run the federated learning experiment."""
    args = parse_args()
    
    # Set data directory if not provided
    if args.data_dir is None:
        args.data_dir = str(Path(__file__).parent.parent / "data")
    
    # Create base configuration
    base_config = FederatedExperimentConfig(
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
        server_learning_rate=args.server_learning_rate,
        server_momentum=args.server_momentum,
        aggregation_frequency=args.aggregation_frequency,
        data_distribution=args.data_distribution,
        non_iid_alpha=args.non_iid_alpha,
        output_dir=args.output_dir,
        save_models=args.save_models
    )
    
    # Run the appropriate experiment
    if args.experiment_type == 'single':
        # Run a single experiment with the specified configuration
        logger.info(f"Running single experiment: {args.experiment_name}")
        runner = FederatedExperimentRunner(base_config)
        runner.setup()
        results = runner.run()
        runner.visualize_results()
        
    elif args.experiment_type == 'learning-rate':
        # Run learning rate sweep
        learning_rates = [float(lr) for lr in args.learning_rates.split(',')]
        logger.info(f"Running learning rate experiment with values: {learning_rates}")
        run_learning_rate_experiment(base_config, learning_rates, args.output_dir)
        
    elif args.experiment_type == 'server-lr':
        # Run server learning rate sweep
        server_lrs = [float(lr) for lr in args.server_learning_rates.split(',')]
        logger.info(f"Running server learning rate experiment with values: {server_lrs}")
        run_server_lr_experiment(base_config, server_lrs, args.output_dir)
        
    elif args.experiment_type == 'aggregation-frequency':
        # Run aggregation frequency sweep
        frequencies = [int(f) for f in args.aggregation_frequencies.split(',')]
        logger.info(f"Running aggregation frequency experiment with values: {frequencies}")
        run_aggregation_frequency_experiment(base_config, frequencies, args.output_dir)
        
    elif args.experiment_type == 'local-epochs':
        # Run local epochs sweep
        epoch_values = [int(e) for e in args.local_epoch_values.split(',')]
        logger.info(f"Running local epochs experiment with values: {epoch_values}")
        run_local_epochs_experiment(base_config, epoch_values, args.output_dir)
        
    elif args.experiment_type == 'multi-seed':
        # Run multi-seed experiment
        seed_values = [int(s) for s in args.seeds.split(',')]
        logger.info(f"Running multi-seed experiment with values: {seed_values}")
        run_multi_seed_experiment(base_config, seed_values, args.output_dir)
    
    logger.info("Experiment completed successfully")

if __name__ == "__main__":
    main() 