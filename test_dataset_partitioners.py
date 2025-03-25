#!/usr/bin/env python
from c2m3.experiments.merge_experiment import MergeExperimentConfig, MergeExperimentRunner
import logging
import sys

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)

def test_dataset(dataset_name):
    """Test the partitioning for a specific dataset"""
    print(f"\n--- Testing {dataset_name.upper()} dataset ---")
    
    # Create a basic configuration for testing
    config = MergeExperimentConfig(
        experiment_name=f'test_{dataset_name}_iid', 
        dataset_name=dataset_name,
        model_name='cnn',
        data_distribution='iid', 
        num_models=10, 
        samples_per_partition=2000 if dataset_name == 'femnist' else None
    )
    
    try:
        # Initialize the experiment runner
        runner = MergeExperimentRunner(config)
        
        # Setup the experiment (this will create the partitions)
        runner.setup()
        
        print(f"{dataset_name.upper()} setup completed successfully!")
        
        # Show info about the created partitions
        for i, loader in enumerate(runner.train_loaders):
            print(f"Train loader {i}: {len(loader.dataset)} samples")
        print(f"Test loader: {len(runner.test_loader.dataset)} samples")
        
        return True
    except Exception as e:
        import traceback
        print(f'Error with {dataset_name}: {e}')
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Get dataset name from command line or use default
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1].lower()
        test_dataset(dataset_name)
    else:
        # Test all supported datasets
        datasets = ['femnist', 'cifar10']  # Can add 'cifar100', 'shakespeare' if dependencies are installed
        
        results = {}
        for dataset in datasets:
            results[dataset] = test_dataset(dataset)
        
        # Print summary
        print("\n--- Summary ---")
        for dataset, success in results.items():
            print(f"{dataset.upper()}: {'✓ Success' if success else '✗ Failed'}") 