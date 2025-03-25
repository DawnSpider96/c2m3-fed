#!/usr/bin/env python
"""
Test script for fixed-size partitioning in FEMNIST dataset.
This script tests whether the samples_per_partition parameter correctly creates partitions of the specified size.
"""

import logging
from pathlib import Path
from c2m3.experiments.merge_experiment import MergeExperimentConfig, MergeExperimentRunner

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fixed_size_partitioning():
    """Test whether the fixed-size partitioning works for FEMNIST dataset"""
    print("\n--- Testing FEMNIST Fixed-Size Partitioning ---")
    
    # Create configurations with different samples_per_partition values
    configs = [
        MergeExperimentConfig(
            experiment_name='test_femnist_default', 
            dataset_name='femnist', 
            model_name='cnn', 
            data_distribution='iid', 
            num_models=3,
            # Default behavior - divide evenly
            samples_per_partition=None  
        ),
        MergeExperimentConfig(
            experiment_name='test_femnist_fixed_1000', 
            dataset_name='femnist', 
            model_name='cnn', 
            data_distribution='iid', 
            num_models=3,
            samples_per_partition=1000  # Fixed size of 1000 samples per partition
        ),
        MergeExperimentConfig(
            experiment_name='test_femnist_fixed_2000', 
            dataset_name='femnist', 
            model_name='cnn', 
            data_distribution='iid', 
            num_models=3,
            samples_per_partition=2000  # Fixed size of 2000 samples per partition
        )
    ]
    
    for config in configs:
        try:
            print(f"\nTesting with samples_per_partition={config.samples_per_partition}")
            
            # Initialize the experiment runner
            runner = MergeExperimentRunner(config)
            
            # Setup the experiment (this will create the partitions)
            runner.setup()
            
            print(f"Setup completed successfully!")
            
            # Show detailed info about the created datasets
            print("\nTRAINING DATASETS:")
            for i, loader in enumerate(runner.train_loaders):
                dataset = loader.dataset
                print(f"Train loader {i}: {len(dataset)} samples")
                
                # Verify dataset type
                print(f"  Dataset type: {type(dataset).__name__}")
                
                # Check if data attribute exists
                if hasattr(dataset, 'data'):
                    print(f"  Dataset.data length: {len(dataset.data)}")
                
            print("\nVALIDATION DATASETS:")
            for i, loader in enumerate(runner.val_loaders):
                dataset = loader.dataset
                print(f"Val loader {i}: {len(dataset)} samples")
                
                # Verify dataset type
                print(f"  Dataset type: {type(dataset).__name__}")
                
                # Check if data attribute exists
                if hasattr(dataset, 'data'):
                    print(f"  Dataset.data length: {len(dataset.data)}")
            
            print("\nTEST DATASET:")
            test_dataset = runner.test_loader.dataset
            print(f"Test loader: {len(test_dataset)} samples")
            print(f"  Dataset type: {type(test_dataset).__name__}")
            
            # Calculate total training samples
            total_train = sum(len(loader.dataset) for loader in runner.train_loaders)
            total_val = sum(len(loader.dataset) for loader in runner.val_loaders)
            print(f"\nTotal training samples: {total_train}")
            print(f"Total validation samples: {total_val}")
            print(f"Total test samples: {len(test_dataset)}")
            
        except Exception as e:
            import traceback
            print(f'Error: {e}')
            traceback.print_exc()

if __name__ == "__main__":
    test_fixed_size_partitioning() 