"""
Corrected data partitioning approach for federated learning experiments.

This file contains a corrected implementation of the partition_data method that:
1. Extracts a universal test set from the full dataset (using stratified sampling)
2. Partitions the remaining data across models
3. For each model, splits its portion into training and validation sets
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
from c2m3.data.partitioners import (
    FEMNISTPathologicalPartitioner,
    FEMNISTHeldOutClassPartitioner
)

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
        
        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        
        if target_transform is None:
            target_transform = to_tensor_transform
            
        
        self.data_dir = data_dir
        self.name = name
        self.mapping = None  
        self.data = samples
        self.transform = transform
        self.target_transform = target_transform
    
    def _load_dataset(self):
        return self.data





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
    
    centralized_mapping = data_dir / "femnist" / "client_data_mappings" / "centralized" / "0"
    
    if not centralized_mapping.exists():
        raise FileNotFoundError(f"Centralized mapping not found at {centralized_mapping}")
    
    
    centralized_dataset = load_femnist_dataset(
        mapping=centralized_mapping,
        data_dir=data_dir / "femnist" / "data",
        name="train"
    )
    
    all_samples = centralized_dataset._load_dataset()
    
    labels = np.array([sample[1] for sample in all_samples])
    
    indices = np.arange(len(all_samples))
    
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.007,  
        random_state=config.seed,
        stratify=labels 
    )
    
    test_samples = [all_samples[i] for i in test_indices]
    
    test_set = FEMNISTFromSamples(
        samples=test_samples,
        data_dir=data_dir / "femnist" / "data",
        name="test"
    )
    
    logger.info(f"Created stratified FEMNIST test set with {len(test_set)} samples")
    
    train_samples = [all_samples[i] for i in train_indices]
    
    random_generator = random.Random(config.seed)
    
    samples_per_partition = getattr(config, "samples_per_partition", None)
    
    shuffled_samples = list(train_samples)
    random_generator.shuffle(shuffled_samples)
    
    train_partitions = []
    val_partitions = []
    
    val_proportion = 0.1
    
    if isinstance(samples_per_partition, list):
        if len(samples_per_partition) != num_partitions:
            logger.warning(
                f"Length of samples_per_partition list ({len(samples_per_partition)}) doesn't match num_partitions ({num_partitions}). "
                f"Using the first {num_partitions} values or repeating the last value as needed."
            )
            if len(samples_per_partition) < num_partitions:
                samples_per_partition.extend([samples_per_partition[-1]] * (num_partitions - len(samples_per_partition)))
            else:
                samples_per_partition = samples_per_partition[:num_partitions]
        
        total_samples_needed = sum(samples_per_partition)
        
        if total_samples_needed > len(shuffled_samples):
            logger.warning(
                f"Not enough samples for partitions with specified sizes. Need {total_samples_needed}, "
                f"but only have {len(shuffled_samples)}. Will scale down partition sizes proportionally."
            )
            scale_factor = len(shuffled_samples) / total_samples_needed
            samples_per_partition = [int(s * scale_factor) for s in samples_per_partition]
            remaining = len(shuffled_samples) - sum(samples_per_partition)
            samples_per_partition[-1] += remaining
        
        start_idx = 0
        for i in range(num_partitions):
            partition_size = samples_per_partition[i]
            end_idx = start_idx + partition_size
            
            if end_idx <= len(shuffled_samples):
                partition_samples = shuffled_samples[start_idx:end_idx]
                
                val_size = int(len(partition_samples) * val_proportion)
                train_size = len(partition_samples) - val_size
                
                train_samples_i = partition_samples[:train_size]
                val_samples_i = partition_samples[train_size:]
                
                train_dataset = FEMNISTFromSamples(
                    samples=train_samples_i,
                    data_dir=data_dir / "femnist" / "data",
                    name="train"
                )
                
                val_dataset = FEMNISTFromSamples(
                    samples=val_samples_i,
                    data_dir=data_dir / "femnist" / "data",
                    name="test"
                )
                
                train_partitions.append(train_dataset)
                val_partitions.append(val_dataset)
                
                start_idx = end_idx
            else:
                logger.warning(f"Not enough samples for partition {i}. Skipping.")
        
        logger.info(f"Created {len(train_partitions)} IID FEMNIST partitions with varied sizes:")
        for i, size in enumerate(samples_per_partition[:len(train_partitions)]):
            logger.info(f"  Partition {i}: {size} samples ({int(size * (1-val_proportion))} train, {int(size * val_proportion)} val)")
    
    else:
        if samples_per_partition is not None:
            samples_needed = samples_per_partition * num_partitions
            
            if samples_needed > len(shuffled_samples):
                logger.warning(
                    f"Not enough samples for {num_partitions} partitions with {samples_per_partition} samples each. "
                    f"Need {samples_needed}, but only have {len(shuffled_samples)}. Will use all available samples."
                )
                partition_size = len(shuffled_samples) // num_partitions
            else:
                partition_size = samples_per_partition
                shuffled_samples = shuffled_samples[:samples_needed]
        else:
            partition_size = len(shuffled_samples) // num_partitions
        
        for i in range(num_partitions):
            start_idx = i * partition_size
            end_idx = start_idx + partition_size
            
            if end_idx <= len(shuffled_samples):
                partition_samples = shuffled_samples[start_idx:end_idx]
                
                val_size = int(len(partition_samples) * val_proportion)
                train_size = len(partition_samples) - val_size
                
                train_samples_i = partition_samples[:train_size]
                val_samples_i = partition_samples[train_size:]
                
                train_dataset = FEMNISTFromSamples(
                    samples=train_samples_i,
                    data_dir=data_dir / "femnist" / "data",
                    name="train"
                )
                
                val_dataset = FEMNISTFromSamples(
                    samples=val_samples_i,
                    data_dir=data_dir / "femnist" / "data",
                    name="test"
                )
                
                train_partitions.append(train_dataset)
                val_partitions.append(val_dataset)
        
        if samples_per_partition is not None:
            logger.info(f"Created {len(train_partitions)} IID FEMNIST partitions with exactly {partition_size} samples each")
            logger.info(f"Each partition: {int(partition_size * (1-val_proportion))} training samples, {int(partition_size * val_proportion)} validation samples")
        else:
            logger.info(f"Created {len(train_partitions)} IID FEMNIST partitions with ~{partition_size} samples each")
    
    
    debug_labels = {}
    
    for i, dataset in enumerate(train_partitions):
        labels = []
        if hasattr(dataset, 'data') and isinstance(dataset.data, list):
            labels = [sample[1] for sample in dataset.data]
        
        debug_labels[i] = labels
    
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
    
    
    centralized_mapping = data_dir / "femnist" / "client_data_mappings" / "centralized" / "0"
    
    if not centralized_mapping.exists():
        raise FileNotFoundError(f"Centralized mapping not found at {centralized_mapping}")
    
    
    centralized_dataset = load_femnist_dataset(
        mapping=centralized_mapping,
        data_dir=data_dir / "femnist" / "data",
        name="train"
    )
    
    all_centralized_samples = centralized_dataset._load_dataset()
    
    centralized_labels = np.array([sample[1] for sample in all_centralized_samples])
    
    centralized_indices = np.arange(len(all_centralized_samples))
    
    _, test_indices = train_test_split(
        centralized_indices,
        test_size=0.007, 
        random_state=config.seed,
        stratify=centralized_labels  
    )
    
    test_samples = [all_centralized_samples[i] for i in test_indices]
    
    test_set = FEMNISTFromSamples(
        samples=test_samples,
        data_dir=data_dir / "femnist" / "data",
        name="test"
    )
    
    logger.info(f"Created stratified FEMNIST test set with {len(test_set)} samples from centralized dataset")
    
    test_sample_paths = {sample[0] for sample in test_samples}
    
    from c2m3.data.partitioners import FEMNISTNaturalPartitioner
    
    partitioner = FEMNISTNaturalPartitioner(data_dir=data_dir, seed=config.seed)
    partitions = partitioner.partition(
        available_client_ids=available_client_ids,
        num_partitions=num_partitions,
        test_sample_paths=test_sample_paths
    )
    
    
    train_partitions = []
    val_partitions = []
    
    for partition in partitions:
        
        if len(partition) < 2:
            logger.warning(f"Skipping partition with only {len(partition)} samples")
            continue
            
        
        train_size = int(0.9 * len(partition))
        val_size = len(partition) - train_size
        
        
        generator = torch.Generator().manual_seed(config.seed)
        train_subset, val_subset = torch.utils.data.random_split(
            partition, [train_size, val_size], generator=generator
        )
        
        train_partitions.append(train_subset)
        val_partitions.append(val_subset)
    
    logger.info(f"Created {len(train_partitions)} natural FEMNIST partitions with train/val splits")
    
    return train_partitions, val_partitions, test_set


def partition_femnist(config, data_dir, available_client_ids, num_partitions):
    """
    Partition FEMNIST dataset based on the specified data distribution.
    
    This function:
    1. Extracts a universal test set using stratified sampling
    2. Uses the appropriate partitioner based on config.data_distribution
    3. Creates validation splits for each partition
    
    Args:
        config: Configuration object with seed and other parameters
        data_dir: Directory containing the dataset
        available_client_ids: List of available client IDs
        num_partitions: Number of partitions to create
        
    Returns:
        Tuple of (train_partitions, val_partitions, test_set)
    """
    
    centralized_mapping = data_dir / "femnist" / "client_data_mappings" / "centralized" / "0"
    
    if not centralized_mapping.exists():
        raise FileNotFoundError(f"Centralized mapping not found at {centralized_mapping}")
    
    
    centralized_dataset = load_femnist_dataset(
        mapping=centralized_mapping,
        data_dir=data_dir / "femnist" / "data",
        name="train"
    )
    
    
    all_centralized_samples = centralized_dataset._load_dataset()
    
    
    centralized_labels = np.array([sample[1] for sample in all_centralized_samples])
    
    
    centralized_indices = np.arange(len(all_centralized_samples))
    
    
    train_indices, test_indices = train_test_split(
        centralized_indices,
        test_size=0.007,
        random_state=config.seed,
        stratify=centralized_labels  
    )
    
    
    test_samples = [all_centralized_samples[i] for i in test_indices]
    
    
    test_set = FEMNISTFromSamples(
        samples=test_samples,
        data_dir=data_dir / "femnist" / "data",
        name="test"
    )
    
    logger.info(f"Created stratified FEMNIST test set with {len(test_set)} samples from centralized dataset")
    
    
    test_sample_paths = {sample[0] for sample in test_samples}
    
    
    training_samples = [all_centralized_samples[i] for i in train_indices]
    
    if config.data_distribution == "iid":
        return partition_femnist_iid(config, data_dir, num_partitions)
    elif config.data_distribution == "natural":
        return partition_femnist_natural(config, data_dir, available_client_ids, num_partitions)
    elif config.data_distribution == "pathological":
        samples_per_partition = getattr(config, "samples_per_partition")
        partitioner = FEMNISTPathologicalPartitioner(
            data_dir=data_dir,
            seed=config.seed,
            samples_per_partition=samples_per_partition
        )
        partitions = partitioner.partition(training_samples, num_partitions)
    elif config.data_distribution in ["lda", "dirichlet"]:
        alpha = getattr(config, "non_iid_alpha", 5.0)
        samples_per_partition = getattr(config, "samples_per_partition", None)
        
        
        from c2m3.common.lda_utils import create_lda_partitions
        
        
        sample_paths = np.array([path for path, _ in training_samples])
        labels = np.array([label for _, label in training_samples])
        
        np.random.seed(config.seed)
        
        accept_imbalanced = True if samples_per_partition is not None else False
        
        partitions_xy, dirichlet_dist = create_lda_partitions(
            dataset=(sample_paths, labels),
            num_partitions=num_partitions,
            concentration=alpha,
            accept_imbalanced=accept_imbalanced,
            seed=config.seed
        )
        
        if samples_per_partition is not None:
            
            rng = random.Random(config.seed)
            
            adjusted_partitions_xy = []
            for x_paths, y_labels in partitions_xy:
                current_size = len(x_paths)
                
                if current_size > samples_per_partition:
                    
                    indices = list(range(current_size))
                    rng.shuffle(indices)
                    selected_indices = indices[:samples_per_partition]
                    
                    adjusted_partitions_xy.append((x_paths[selected_indices], y_labels[selected_indices]))
                else:
                    
                    adjusted_partitions_xy.append((x_paths, y_labels))
                    
                    if current_size < samples_per_partition:
                        logger.warning(
                            f"Partition has only {current_size} samples, "
                            f"less than requested {samples_per_partition}. "
                            f"Using all available samples for this partition."
                        )
            
            partitions_xy = adjusted_partitions_xy
        
        
        partitions = []
        for i, (x_paths, y_labels) in enumerate(partitions_xy):
            
            partition_samples = [(path, int(label)) for path, label in zip(x_paths, y_labels)]
            
            
            partition = FEMNISTFromSamples(
                samples=partition_samples,
                data_dir=data_dir / "femnist" / "data",
                name="train"
            )
            partitions.append(partition)
            
            
            if logger.level <= logging.DEBUG:
                class_counts = {}
                for _, label in partition_samples:
                    class_counts[label] = class_counts.get(label, 0) + 1
                
                
                total = sum(class_counts.values())
                class_percents = {label: 100 * count / total for label, count in class_counts.items()}
                logger.debug(f"Partition {i} has {len(partition_samples)} samples with class distribution: {class_percents}")
        
        if samples_per_partition is not None:
            logger.info(f"Created {len(partitions)} LDA-based FEMNIST partitions with alpha={alpha} and approximately {samples_per_partition} samples per partition")
        else:
            logger.info(f"Created {len(partitions)} LDA-based FEMNIST partitions with alpha={alpha}")
            
        
        logger.info(f"Used create_lda_partitions from lda_utils module with concentration={alpha}")
    elif config.data_distribution == "held_out_class":
        
        samples_per_partition = getattr(config, "samples_per_partition", None)
        partitioner = FEMNISTHeldOutClassPartitioner(
            data_dir=data_dir,
            seed=config.seed,
            samples_per_partition=samples_per_partition
        )
        partitions = partitioner.partition(training_samples, num_partitions)
    else:
        raise ValueError(f"Unsupported FEMNIST partitioning strategy: {config.data_distribution}")
    
    
    train_partitions = []
    val_partitions = []
    
    for partition in partitions:
        
        if len(partition) < 2:
            logger.warning(f"Skipping partition with only {len(partition)} samples")
            continue
            
        
        train_size = int(0.9 * len(partition))
        val_size = len(partition) - train_size
        
        
        generator = torch.Generator().manual_seed(config.seed)
        train_subset, val_subset = torch.utils.data.random_split(
            partition, [train_size, val_size], generator=generator
        )
        
        train_partitions.append(train_subset)
        val_partitions.append(val_subset)
    
    logger.info(f"Created {len(train_partitions)} FEMNIST partitions with train/val splits using {config.data_distribution} strategy")
    
    
    logger.info(f"Analyzing {num_partitions} partitions with {config.data_distribution} distribution...")
    train_partition_analysis = analyze_partition_classes(train_partitions)
    summary = summarize_partitions(train_partition_analysis)
    
    logger.info(f"Partition Summary for {config.data_distribution} distribution:")
    logger.info(f"  Number of clients: {summary['num_clients']}")
    logger.info(f"  Avg classes per client: {summary['avg_classes_per_client']:.2f}")
    logger.info(f"  Min/Max classes: {summary['min_classes']}/{summary['max_classes']}")
    logger.info(f"  Avg samples per client: {summary['avg_samples_per_client']:.2f}")
    
    if config.data_distribution.lower() == "iid":
        low_coverage = [cls for cls, pct in summary['class_coverage'].items() if pct < 90]
        if low_coverage:
            logger.warning(f"Some classes have low coverage across clients: {low_coverage}")
        else:
            logger.info("All classes are well-distributed across clients (IID confirmed)")
    
    return train_partitions, val_partitions, test_set


def analyze_partition_classes(partitions):
    """
    Analyze the class distribution in a list of partitions.
    
    Args:
        partitions: List of FEMNISTFromSamples objects or Subset objects
        
    Returns:
        list: A list where each element is a dict containing:
            - 'client_id': Index of the partition
            - 'num_samples': Total number of samples in the partition
            - 'classes': Sorted list of unique class labels in the partition
            - 'class_counts': Dictionary with count of samples per class
    """
    partition_analysis = []
    
    for i, partition in enumerate(partitions):
        
        if isinstance(partition, torch.utils.data.Subset):
            
            dataset = partition.dataset
            indices = partition.indices
            
            
            if hasattr(dataset, 'data') and isinstance(dataset.data, list):
                
                labels = [dataset.data[idx][1] for idx in indices]
            else:
                
                try:
                    labels = [dataset[idx][1] if isinstance(dataset[idx][1], int) 
                             else dataset[idx][1].item() for idx in indices]
                except:
                    labels = []
                    print(f"Could not extract labels for partition {i}")
        
        elif hasattr(partition, 'data') and isinstance(partition.data, list):
            
            labels = [sample[1] for sample in partition.data]
        
        else:
            
            try:
                labels = [partition[j][1] if isinstance(partition[j][1], int)
                         else partition[j][1].item() for j in range(len(partition))]
            except:
                labels = []
                print(f"Could not extract labels for partition {i}")
        
        
        unique_classes = sorted(set(labels))
        class_counts = {cls: labels.count(cls) for cls in unique_classes}
        
        
        partition_analysis.append({
            'client_id': i,
            'num_samples': len(labels),
            'classes': unique_classes,
            'class_counts': class_counts
        })
    
    return partition_analysis

def summarize_partitions(partitions_analysis):
    """
    Create a concise summary of partition class distributions.
    
    Args:
        partitions_analysis: Output from analyze_partition_classes
        
    Returns:
        dict: Summary statistics including:
            - avg_classes_per_client: Average number of unique classes per client
            - min_classes: Minimum number of classes in any client
            - max_classes: Maximum number of classes in any client
            - class_coverage: Percentage of clients that have each class
    """
    if not partitions_analysis:
        return {"error": "No partitions to analyze"}
    
    
    classes_per_client = [len(p['classes']) for p in partitions_analysis]
    
    
    all_classes = set()
    for p in partitions_analysis:
        all_classes.update(p['classes'])
    
    
    num_clients = len(partitions_analysis)
    class_coverage = {}
    for cls in sorted(all_classes):
        clients_with_class = sum(1 for p in partitions_analysis if cls in p['classes'])
        class_coverage[cls] = clients_with_class / num_clients * 100
    
    
    summary = {
        'num_clients': num_clients,
        'avg_classes_per_client': sum(classes_per_client) / num_clients,
        'min_classes': min(classes_per_client),
        'max_classes': max(classes_per_client),
        'class_coverage': class_coverage,
        'avg_samples_per_client': sum(p['num_samples'] for p in partitions_analysis) / num_clients
    }
    
    return summary

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
    
    
        
    if dataset_name == "femnist":
        return partition_femnist(config, data_dir, available_client_ids, num_partitions)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")