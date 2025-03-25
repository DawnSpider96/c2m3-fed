#!/usr/bin/env python
from c2m3.experiments.merge_experiment import MergeExperimentConfig, MergeExperimentRunner
import logging

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO)

# Create a basic configuration for testing
config = MergeExperimentConfig(
    experiment_name='test_iid', 
    dataset_name='femnist', 
    model_name='cnn', 
    data_distribution='iid', 
    num_models=3, 
    samples_per_partition=2000
)

try:
    # Initialize the experiment runner
    runner = MergeExperimentRunner(config)
    
    # Setup the experiment (this will create the partitions)
    runner.setup()
    
    print('Setup completed successfully!')
    
    # Optional: show some info about the created partitions
    for i, loader in enumerate(runner.train_loaders):
        print(f"Train loader {i}: {len(loader.dataset)} samples")
    print(f"Test loader: {len(runner.test_loader.dataset)} samples")
    
except Exception as e:
    import traceback
    print(f'Error: {e}')
    traceback.print_exc() 