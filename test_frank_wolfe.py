#!/usr/bin/env python3
"""
Test script to diagnose Frank-Wolfe merging issues.
This script creates multiple models with different weights,
then attempts to merge them using the Frank-Wolfe algorithm.
"""
import argparse
import logging
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append('.')

# Import project modules
from c2m3.common.client_utils import (
    Net, 
    get_network_generator_cnn,
    get_model_parameters,
    set_model_parameters,
    load_femnist_dataset
)
from c2m3.modules.pl_module import MyLightningModule
from c2m3.match.frank_wolfe_sync_merging import frank_wolfe_synchronized_merging
from c2m3.utils.utils import l2_norm_models, get_model, set_seed

def create_models_with_different_weights(num_models=3, randomness=0.1):
    """
    Create multiple CNN models with slightly different weights
    
    Args:
        num_models: Number of models to create
        randomness: Scale of random perturbation to add to weights
        
    Returns:
        List of MyLightningModule models
    """
    logger.info(f"Creating {num_models} models with randomness={randomness}")
    
    # Create a base model
    network_generator = get_network_generator_cnn()
    base_model = network_generator()
    
    # Create models with perturbed weights
    models = []
    for i in range(num_models):
        # Clone the base model
        model = network_generator()
        model.load_state_dict(base_model.state_dict())
        
        # Add random perturbations to weights
        with torch.no_grad():
            for param in model.parameters():
                # Add random noise scaled by randomness
                noise = torch.randn_like(param) * randomness
                param.add_(noise)
        
        # Wrap in Lightning module
        lightning_model = MyLightningModule(model, num_classes=62)
        models.append(lightning_model)
        
    return models

def create_mock_dataloaders(num_loaders=3, batch_size=16):
    """
    Create dummy dataloaders for testing
    
    Args:
        num_loaders: Number of dataloaders to create
        batch_size: Batch size for dataloaders
        
    Returns:
        List of DataLoader objects
    """
    logger.info(f"Creating {num_loaders} mock dataloaders")
    
    loaders = []
    for i in range(num_loaders):
        # Create dummy data: 100 samples, 1 channel, 28x28 images
        data = torch.randn(100, 1, 28, 28)
        labels = torch.randint(0, 62, (100,))
        
        # Create dataset and dataloader
        dataset = TensorDataset(data, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)
    
    return loaders

def test_with_real_femnist_data(data_dir=None, num_models=3, randomness=0.1):
    """
    Test Frank-Wolfe merging with real FEMNIST data
    
    Args:
        data_dir: Directory containing FEMNIST data
        num_models: Number of models to create and merge
        randomness: Scale of random perturbation for model weights
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "c2m3" / "data"
    
    logger.info(f"Testing with real FEMNIST data from {data_dir}")
    
    try:
        # Load FEMNIST dataset
        dataset = load_femnist_dataset(
            mapping=None,  # Use None to load all data
            name="train",
            data_dir=data_dir,
        )
        
        # Split into chunks for different clients
        chunk_size = len(dataset) // num_models
        datasets = []
        for i in range(num_models):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_models - 1 else len(dataset)
            indices = list(range(start_idx, end_idx))
            datasets.append(torch.utils.data.Subset(dataset, indices))
        
        # Create dataloaders
        dataloaders = [
            DataLoader(ds, batch_size=16, shuffle=True, num_workers=0, drop_last=True)
            for ds in datasets
        ]
        
        # Create models with different weights
        models = create_models_with_different_weights(num_models, randomness)
        
        # Check L2 distances between models before merging
        logger.info("L2 distances between models before merging:")
        for i in range(num_models):
            for j in range(i+1, num_models):
                dist = l2_norm_models(
                    get_model(models[i]).state_dict(),
                    get_model(models[j]).state_dict()
                )
                logger.info(f"  Distance between model {i} and model {j}: {dist:.6f}")
        
        # Merge models
        logger.info("Merging models using Frank-Wolfe algorithm")
        merged_model = frank_wolfe_synchronized_merging(models, dataloaders)
        
        # Check L2 distances between merged model and original models
        logger.info("L2 distances between merged model and original models:")
        merged_state_dict = get_model(merged_model).state_dict()
        for i in range(num_models):
            dist = l2_norm_models(
                merged_state_dict,
                get_model(models[i]).state_dict()
            )
            logger.info(f"  Distance between merged model and model {i}: {dist:.6f}")
        
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_with_mock_data(num_models=3, randomness=0.1):
    """
    Test Frank-Wolfe merging with mock data
    
    Args:
        num_models: Number of models to create and merge
        randomness: Scale of random perturbation for model weights
    """
    logger.info(f"Testing with mock data")
    
    try:
        # Create models with different weights
        models = create_models_with_different_weights(num_models, randomness)
        
        # Create mock dataloaders
        dataloaders = create_mock_dataloaders(num_models)
        
        # Check L2 distances between models before merging
        logger.info("L2 distances between models before merging:")
        for i in range(num_models):
            for j in range(i+1, num_models):
                dist = l2_norm_models(
                    get_model(models[i]).state_dict(),
                    get_model(models[j]).state_dict()
                )
                logger.info(f"  Distance between model {i} and model {j}: {dist:.6f}")
        
        # Merge models
        logger.info("Merging models using Frank-Wolfe algorithm")
        merged_model = frank_wolfe_synchronized_merging(models, dataloaders)
        
        # Check L2 distances between merged model and original models
        logger.info("L2 distances between merged model and original models:")
        merged_state_dict = get_model(merged_model).state_dict()
        for i in range(num_models):
            dist = l2_norm_models(
                merged_state_dict,
                get_model(models[i]).state_dict()
            )
            logger.info(f"  Distance between merged model and model {i}: {dist:.6f}")
        
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Frank-Wolfe merging")
    parser.add_argument("--real-data", action="store_true", help="Use real FEMNIST data")
    parser.add_argument("--data-dir", type=str, help="Directory containing FEMNIST data", default=None)
    parser.add_argument("--num-models", type=int, default=3, help="Number of models to create and merge")
    parser.add_argument("--randomness", type=float, default=0.1, help="Scale of random perturbation for model weights")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    if args.real_data:
        test_with_real_femnist_data(args.data_dir, args.num_models, args.randomness)
    else:
        test_with_mock_data(args.num_models, args.randomness) 