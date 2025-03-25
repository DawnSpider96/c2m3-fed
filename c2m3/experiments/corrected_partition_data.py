"""
Corrected data partitioning approach for federated learning experiments.

This file contains a corrected implementation of the partition_data method that:
1. Extracts a universal test set from the full dataset (using stratified sampling)
2. Then partitions the remaining data across models
3. Finally, for each model, splits its portion into training and validation sets
"""

import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from c2m3.common.femnist_dataset import FEMNIST
from c2m3.common.client_utils import load_femnist_dataset, to_tensor_transform
from c2m3.data.partitioners import PartitionerRegistry, DatasetPartitioner

# Configure logger
logger = logging.getLogger(__name__)


class FEMNISTFromSamples(FEMNIST):
    """
    Extension of FEMNIST class that allows creating a dataset directly from samples.
    This is needed for creating stratified test sets.
    """
    def __init__(
        self,
        samples: List[Tuple[str, int]],
        data_dir: Path,
        name: str = "train",
        transform = None,
        target_transform = None,
    ):
        """Initialize FEMNIST dataset directly from samples list.
        
        Args:
            samples: List of tuples (sample_path, label)
            data_dir: Path to the dataset folder
            name: Name of the dataset (train, test, validation)
            transform: Transform to apply to images
            target_transform: Transform to apply to labels
        """
        # Apply default transform if none provided
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        # Copy implementation from load_femnist_dataset
        if target_transform is None:
            target_transform = to_tensor_transform
            
        # Set instance variables directly
        self.data_dir = data_dir
        self.name = name
        self.mapping = None  # No mapping file needed
        self.data = samples
        self.transform = transform
        self.target_transform = target_transform
    
    def _load_dataset(self):
        return self.data


def partition_cifar10(config, data_dir, num_partitions):
    """
    Partition CIFAR10 dataset.
    
    Args:
        config: Configuration object with data_distribution and other parameters
        data_dir: Directory containing the dataset
        num_partitions: Number of partitions to create
        
    Returns:
        Tuple of (train_partitions, val_partitions, test_set)
    """
    # Load the complete dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    standard_test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    
    # Get class labels for stratification
    targets = np.array(full_dataset.targets)
    
    # Extract test set using stratified sampling (20% of training data)
    indices = np.arange(len(full_dataset))
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=0.2,
        random_state=config.seed,
        stratify=targets
    )
    
    # Create train and test datasets
    train_dataset = Subset(full_dataset, train_indices)
    extracted_test_set = Subset(full_dataset, test_indices)
    
    logger.info(f"Extracted stratified test set from CIFAR10 training data: {len(extracted_test_set)} samples")
    
    # Combine extracted test set with standard test set
    test_set = ConcatDataset([extracted_test_set, standard_test_set])
    logger.info(f"Combined test set size: {len(test_set)} samples")
    
    # Select the appropriate partitioner
    if config.data_distribution == "iid":
        partitioner_class = PartitionerRegistry.get("default", "iid")
        partitioner = partitioner_class(seed=config.seed)
        logger.info(f"Using IID partitioner for CIFAR10 with seed={config.seed}")
    else:
        # Try dataset-specific partitioner, fall back to default
        try:
            partitioner_class = PartitionerRegistry.get("cifar10", config.data_distribution)
            logger.info(f"Using CIFAR10-specific {config.data_distribution} partitioner")
        except ValueError:
            partitioner_class = PartitionerRegistry.get("default", config.data_distribution)
            logger.info(f"Using default {config.data_distribution} partitioner for CIFAR10")
        
        # Create partitioner with appropriate parameters
        if config.data_distribution == "dirichlet":
            partitioner = partitioner_class(alpha=config.non_iid_alpha)
        elif config.data_distribution == "pathological":
            partitioner = partitioner_class(classes_per_partition=config.classes_per_partition)
        else:
            partitioner = partitioner_class()
    
    # Partition the remaining training dataset
    partitions = partitioner.partition(train_dataset, num_partitions)
    
    # For each partition, split into train and validation
    train_partitions = []
    val_partitions = []
    
    for partition in partitions:
        # Use 90/10 split for train/validation
        train_size = int(0.9 * len(partition))
        val_size = len(partition) - train_size
        
        # Create the split with fixed seed for reproducibility
        generator = torch.Generator().manual_seed(config.seed)
        train_subset, val_subset = torch.utils.data.random_split(
            partition, [train_size, val_size], generator=generator
        )
        
        train_partitions.append(train_subset)
        val_partitions.append(val_subset)
    
    logger.info(f"Created {len(train_partitions)} CIFAR10 training partitions with validation splits")
    
    return train_partitions, val_partitions, test_set


def partition_cifar100(config, data_dir, num_partitions):
    """
    Partition CIFAR100 dataset.
    
    Args:
        config: Configuration object with data_distribution and other parameters
        data_dir: Directory containing the dataset
        num_partitions: Number of partitions to create
        
    Returns:
        Tuple of (train_partitions, val_partitions, test_set)
    """
    # Load the complete dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    full_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    standard_test_set = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
    
    # Get class labels for stratification
    targets = np.array(full_dataset.targets)
    
    # Extract test set using stratified sampling (20% of training data)
    indices = np.arange(len(full_dataset))
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=0.2,
        random_state=config.seed,
        stratify=targets
    )
    
    # Create train and test datasets
    train_dataset = Subset(full_dataset, train_indices)
    extracted_test_set = Subset(full_dataset, test_indices)
    
    logger.info(f"Extracted stratified test set from CIFAR100 training data: {len(extracted_test_set)} samples")
    
    # Combine extracted test set with standard test set
    test_set = ConcatDataset([extracted_test_set, standard_test_set])
    logger.info(f"Combined test set size: {len(test_set)} samples")
    
    # Select the appropriate partitioner
    if config.data_distribution == "iid":
        partitioner_class = PartitionerRegistry.get("default", "iid")
        partitioner = partitioner_class(seed=config.seed)
        logger.info(f"Using IID partitioner for CIFAR100 with seed={config.seed}")
    else:
        # Try dataset-specific partitioner, fall back to default
        try:
            partitioner_class = PartitionerRegistry.get("cifar100", config.data_distribution)
            logger.info(f"Using CIFAR100-specific {config.data_distribution} partitioner")
        except ValueError:
            partitioner_class = PartitionerRegistry.get("default", config.data_distribution)
            logger.info(f"Using default {config.data_distribution} partitioner for CIFAR100")
        
        # Create partitioner with appropriate parameters
        if config.data_distribution == "dirichlet":
            partitioner = partitioner_class(alpha=config.non_iid_alpha)
        elif config.data_distribution == "pathological":
            partitioner = partitioner_class(classes_per_partition=config.classes_per_partition)
        else:
            partitioner = partitioner_class()
    
    # Partition the remaining training dataset
    partitions = partitioner.partition(train_dataset, num_partitions)
    
    # For each partition, split into train and validation
    train_partitions = []
    val_partitions = []
    
    for partition in partitions:
        # Use 90/10 split for train/validation
        train_size = int(0.9 * len(partition))
        val_size = len(partition) - train_size
        
        # Create the split with fixed seed for reproducibility
        generator = torch.Generator().manual_seed(config.seed)
        train_subset, val_subset = torch.utils.data.random_split(
            partition, [train_size, val_size], generator=generator
        )
        
        train_partitions.append(train_subset)
        val_partitions.append(val_subset)
    
    logger.info(f"Created {len(train_partitions)} CIFAR100 training partitions with validation splits")
    
    return train_partitions, val_partitions, test_set


def partition_shakespeare(config, data_dir, num_partitions):
    """
    Partition Shakespeare dataset.
    
    Args:
        config: Configuration object with data_distribution and other parameters
        data_dir: Directory containing the dataset
        num_partitions: Number of partitions to create
        
    Returns:
        Tuple of (train_partitions, val_partitions, test_set)
    """
    from torchtext.datasets import CharacterLevel
    
    # Create a character-level language modeling dataset
    full_dataset = CharacterLevel('shakespeare', root=data_dir)
    
    # Extract a test set (20% of the data)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # Use seeded generator for reproducibility
    generator = torch.Generator().manual_seed(config.seed)
    train_dataset, test_set = torch.utils.data.random_split(
        full_dataset, 
        [train_size, test_size], 
        generator=generator
    )
    
    logger.info(f"Split Shakespeare dataset: {len(train_dataset)} training samples, {len(test_set)} test samples")
    
    # Select the appropriate partitioner
    if config.data_distribution == "iid":
        partitioner_class = PartitionerRegistry.get("default", "iid")
        partitioner = partitioner_class(seed=config.seed)
        logger.info(f"Using IID partitioner for Shakespeare with seed={config.seed}")
    else:
        # Try dataset-specific partitioner, fall back to default
        try:
            partitioner_class = PartitionerRegistry.get("shakespeare", config.data_distribution)
            logger.info(f"Using Shakespeare-specific {config.data_distribution} partitioner")
        except ValueError:
            partitioner_class = PartitionerRegistry.get("default", config.data_distribution)
            logger.info(f"Using default {config.data_distribution} partitioner for Shakespeare")
    
        # Create partitioner with appropriate parameters
        if config.data_distribution == "dirichlet":
            partitioner = partitioner_class(alpha=config.non_iid_alpha)
        elif config.data_distribution == "pathological":
            partitioner = partitioner_class(classes_per_partition=config.classes_per_partition)
        else:
            partitioner = partitioner_class()
    
    # Partition the training dataset
    partitions = partitioner.partition(train_dataset, num_partitions)
    
    # For each partition, split into train and validation
    train_partitions = []
    val_partitions = []
    
    for partition in partitions:
        # Use 90/10 split for train/validation
        train_size = int(0.9 * len(partition))
        val_size = len(partition) - train_size
        
        # Create the split with fixed seed for reproducibility
        generator = torch.Generator().manual_seed(config.seed)
        train_subset, val_subset = torch.utils.data.random_split(
            partition, [train_size, val_size], generator=generator
        )
        
        train_partitions.append(train_subset)
        val_partitions.append(val_subset)
    
    logger.info(f"Created {len(train_partitions)} Shakespeare training partitions with validation splits")
    
    return train_partitions, val_partitions, test_set


def partition_femnist_iid(config, data_dir, num_partitions):
    """
    Partition FEMNIST dataset using IID partitioning.
    
    This function:
    1. Extracts a universal test set using stratified sampling
    2. Partitions the remaining training data
    3. Creates separate datasets for each partition's train and validation sets
    
    Args:
        config: Configuration object with seed and other parameters
        data_dir: Directory containing the dataset
        num_partitions: Number of partitions to create
        
    Returns:
        Tuple of (train_partitions, val_partitions, test_set)
    """
    # Use centralized mapping
    centralized_mapping = data_dir / "femnist" / "client_data_mappings" / "centralized" / "0"
    
    if not centralized_mapping.exists():
        raise FileNotFoundError(f"Centralized mapping not found at {centralized_mapping}")
    
    # Load the centralized dataset
    centralized_dataset = load_femnist_dataset(
        mapping=centralized_mapping,
        data_dir=data_dir / "femnist" / "data",
        name="train"
    )
    
    # Get all samples and their labels for stratification
    all_samples = centralized_dataset._load_dataset()
    
    # Extract labels from samples (second element of each tuple)
    labels = np.array([sample[1] for sample in all_samples])
    
    # Create indices for stratified split
    indices = np.arange(len(all_samples))
    
    # Split into train and test set using stratified sampling
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.007,  # 2400 sample test set
        random_state=config.seed,
        stratify=labels  # Ensure balanced class distribution
    )
    
    # Create test set samples
    test_samples = [all_samples[i] for i in test_indices]
    
    # Create a custom FEMNIST dataset with the test samples
    test_set = FEMNISTFromSamples(
        samples=test_samples,
        data_dir=data_dir / "femnist" / "data",
        name="test"
    )
    
    logger.info(f"Created stratified FEMNIST test set with {len(test_set)} samples")
    
    # Create training samples (remaining samples)
    train_samples = [all_samples[i] for i in train_indices]
    
    # Instead of using the partitioner to create datasets directly,
    # we'll partition the raw samples and then create datasets at the end
    
    # Initialize random generator for reproducibility
    random_generator = random.Random(config.seed)
    
    # Get the samples_per_partition from the config if it exists
    samples_per_partition = getattr(config, "samples_per_partition", None)
    
    # Shuffle samples for IID distribution
    shuffled_samples = list(train_samples)
    random_generator.shuffle(shuffled_samples)
    
    # Determine partition size
    if samples_per_partition is not None:
        # Fixed-size partitioning: Each partition has exactly samples_per_partition samples
        samples_needed = samples_per_partition * num_partitions
        
        # Check if we have enough samples
        if samples_needed > len(shuffled_samples):
            logger.warning(
                f"Not enough samples for {num_partitions} partitions with {samples_per_partition} samples each. "
                f"Need {samples_needed}, but only have {len(shuffled_samples)}. Will use all available samples."
            )
            # Fall back to even division
            partition_size = len(shuffled_samples) // num_partitions
        else:
            partition_size = samples_per_partition
            # We may not use all samples if fixed size is specified
            shuffled_samples = shuffled_samples[:samples_needed]
    else:
        # Even division: Divide all samples evenly among partitions
        partition_size = len(shuffled_samples) // num_partitions
    
    # Create train and validation partitions directly
    train_partitions = []
    val_partitions = []
    
    # Calculate validation proportion (10%)
    val_proportion = 0.1
    
    for i in range(num_partitions):
        start_idx = i * partition_size
        end_idx = start_idx + partition_size
        
        # Make sure we don't go beyond available samples
        if end_idx <= len(shuffled_samples):
            partition_samples = shuffled_samples[start_idx:end_idx]
            
            # Calculate split sizes for this partition
            val_size = int(len(partition_samples) * val_proportion)
            train_size = len(partition_samples) - val_size
            
            # Split into train and validation samples
            train_samples_i = partition_samples[:train_size]
            val_samples_i = partition_samples[train_size:]
            
            # Create datasets directly from samples
            train_dataset = FEMNISTFromSamples(
                samples=train_samples_i,
                data_dir=data_dir / "femnist" / "data",
                name="train"
            )
            
            val_dataset = FEMNISTFromSamples(
                samples=val_samples_i,
                data_dir=data_dir / "femnist" / "data",
                name="val"
            )
            
            train_partitions.append(train_dataset)
            val_partitions.append(val_dataset)
    
    # Log information about the partitioning
    if samples_per_partition is not None:
        logger.info(f"Created {len(train_partitions)} IID FEMNIST partitions with exactly {partition_size} samples each")
        logger.info(f"Each partition: {int(partition_size * (1-val_proportion))} training samples, {int(partition_size * val_proportion)} validation samples")
    else:
        logger.info(f"Created {len(train_partitions)} IID FEMNIST partitions with ~{partition_size} samples each")
    
    return train_partitions, val_partitions, test_set


def partition_femnist_natural(config, data_dir, available_client_ids, num_partitions):
    """
    Partition FEMNIST dataset using natural partitioning.
    
    This function:
    1. Extracts a universal test set using stratified sampling
    2. Uses FEMNISTNaturalPartitioner to distribute remaining samples by writer
    3. Creates validation splits for each partition
    
    Args:
        config: Configuration object with seed and other parameters
        data_dir: Directory containing the dataset
        available_client_ids: List of available client IDs
        num_partitions: Number of partitions to create
        
    Returns:
        Tuple of (train_partitions, val_partitions, test_set)
    """
    # Step 1: Create a universal test set from the centralized dataset
    # This ensures consistent test set regardless of partitioning strategy
    centralized_mapping = data_dir / "femnist" / "client_data_mappings" / "centralized" / "0"
    
    if not centralized_mapping.exists():
        raise FileNotFoundError(f"Centralized mapping not found at {centralized_mapping}")
    
    # Load the centralized dataset
    centralized_dataset = load_femnist_dataset(
        mapping=centralized_mapping,
        data_dir=data_dir / "femnist" / "data",
        name="train"
    )
    
    # Get all samples and their labels for stratification
    all_centralized_samples = centralized_dataset._load_dataset()
    
    # Extract labels from samples (second element of each tuple)
    centralized_labels = np.array([sample[1] for sample in all_centralized_samples])
    
    # Create indices for stratified split
    centralized_indices = np.arange(len(all_centralized_samples))
    
    # Split into train and test set using stratified sampling
    _, test_indices = train_test_split(
        centralized_indices,
        test_size=0.2,  # 20% for test set
        random_state=config.seed,
        stratify=centralized_labels  # Ensure balanced class distribution
    )
    
    # Create test set samples
    test_samples = [all_centralized_samples[i] for i in test_indices]
    
    # Create a test dataset
    test_set = FEMNISTFromSamples(
        samples=test_samples,
        data_dir=data_dir / "femnist" / "data",
        name="test"
    )
    
    logger.info(f"Created stratified FEMNIST test set with {len(test_set)} samples from centralized dataset")
    
    # Create a set of test sample paths for efficient lookup
    test_sample_paths = {sample[0] for sample in test_samples}
    
    # Step 2: Use the FEMNISTNaturalPartitioner to create partitions, filtering out test samples
    from c2m3.data.partitioners import FEMNISTNaturalPartitioner
    
    partitioner = FEMNISTNaturalPartitioner(data_dir=data_dir, seed=config.seed)
    partitions = partitioner.partition(
        available_client_ids=available_client_ids,
        num_partitions=num_partitions,
        test_sample_paths=test_sample_paths
    )
    
    # Step 3: For each partition, split into train and validation
    train_partitions = []
    val_partitions = []
    
    for partition in partitions:
        # Skip partitions with too few samples
        if len(partition) < 2:
            logger.warning(f"Skipping partition with only {len(partition)} samples")
            continue
            
        # Use 90/10 split for train/validation
        train_size = int(0.9 * len(partition))
        val_size = len(partition) - train_size
        
        # Create the split with fixed seed for reproducibility
        generator = torch.Generator().manual_seed(config.seed)
        train_subset, val_subset = torch.utils.data.random_split(
            partition, [train_size, val_size], generator=generator
        )
        
        train_partitions.append(train_subset)
        val_partitions.append(val_subset)
    
    logger.info(f"Created {len(train_partitions)} natural FEMNIST partitions with train/val splits")
    
    return train_partitions, val_partitions, test_set


def partition_data(config, data_refs, num_partitions, available_client_ids=None):
    """
    Create data partitions based on the dataset type and distribution.
    
    The correct approach is:
    1. First extract a universal test set from the full dataset (using stratified sampling)
    2. Then partition the remaining data (IID or otherwise) across models
    3. Finally, for each model, split its portion into training and validation sets
    
    Args:
        config: Configuration with data_distribution, seed, etc.
        data_refs: References to data depending on the dataset
        num_partitions: Number of partitions to create
        available_client_ids: List of available client IDs (for FEMNIST natural partitioning)
        
    Returns:
        Tuple of (train_partitions, val_partitions, test_set) for all datasets
    """
    dataset_name = data_refs.get("dataset_name", "unknown")
    data_dir = Path(data_refs.get("data_dir", "."))
    
    if dataset_name == "cifar10":
        return partition_cifar10(config, data_dir, num_partitions)
    elif dataset_name == "cifar100":
        return partition_cifar100(config, data_dir, num_partitions)
    elif dataset_name == "shakespeare":
        return partition_shakespeare(config, data_dir, num_partitions)
    elif dataset_name == "femnist":
        if config.data_distribution == "iid":
            return partition_femnist_iid(config, data_dir, num_partitions)
        else:
            return partition_femnist_natural(config, data_dir, available_client_ids, num_partitions)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}") 