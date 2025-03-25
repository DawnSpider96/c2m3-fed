"""
Partitioning strategies for dataset splitting.
Implements various ways to partition datasets for federated learning simulations.
"""

import random
import logging
from typing import List, Dict, Type, Union, Optional, Tuple, Any, ClassVar
from pathlib import Path
from abc import ABC, abstractmethod
import csv

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from c2m3.common.client_utils import load_femnist_dataset
from c2m3.common.femnist_dataset import FEMNIST

logger = logging.getLogger(__name__)


class DatasetPartitioner(ABC):
    """
    Base abstract class for dataset partitioning strategies.
    All partitioning strategies should inherit from this class.
    """
    @abstractmethod
    def partition(self, dataset: Dataset, num_partitions: int) -> List[Dataset]:
        """
        Partition the dataset into num_partitions subsets.
        
        Args:
            dataset: The dataset to partition
            num_partitions: The number of partitions to create
            
        Returns:
            A list of dataset partitions
        """
        raise NotImplementedError("Subclasses must implement partition method")


class PartitionerRegistry:
    """
    Registry for dataset partitioning strategies.
    Stores and retrieves partitioning strategies by dataset name and partition type.
    """
    _registry: ClassVar[Dict[str, Dict[str, Type[DatasetPartitioner]]]] = {}
    
    @classmethod
    def register(cls, dataset_name: str, partition_type: str, partitioner_class: Type[DatasetPartitioner]) -> None:
        """
        Register a partitioning strategy for a dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'femnist', 'cifar10', or 'default')
            partition_type: Type of partitioning (e.g., 'iid', 'dirichlet', 'pathological')
            partitioner_class: The partitioner class to register
        """
        if dataset_name not in cls._registry:
            cls._registry[dataset_name] = {}
        cls._registry[dataset_name][partition_type] = partitioner_class
        logger.debug(f"Registered partitioner {partitioner_class.__name__} for {dataset_name}, {partition_type}")
    
    @classmethod
    def get(cls, dataset_name: str, partition_type: str) -> Type[DatasetPartitioner]:
        """
        Get a partitioning strategy for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            partition_type: Type of partitioning
            
        Returns:
            A partitioner class
            
        Raises:
            ValueError: If no partitioner is found for the given dataset and partition type
        """
        try:
            return cls._registry[dataset_name][partition_type]
        except KeyError:
            # Fallback to default partitioners if dataset-specific one not found
            try:
                return cls._registry["default"][partition_type]
            except KeyError:
                raise ValueError(f"No partitioner found for {dataset_name}, {partition_type}")


class IIDPartitioner(DatasetPartitioner):
    """
    IID (Independent and Identically Distributed) partitioning strategy.
    Randomly shuffles and splits the dataset into equal-sized partitions.
    """
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the IID partitioner.
        
        Args:
            seed: Optional random seed for reproducibility. If provided, 
                 uses a dedicated random generator for consistent results.
        """
        self.seed = seed
        self.random_generator = random.Random(seed) if seed is not None else random
    
    def partition(self, dataset: Dataset, num_partitions: int) -> List[Dataset]:
        """
        Partition the dataset in an IID fashion.
        
        Args:
            dataset: The dataset to partition
            num_partitions: The number of partitions to create
            
        Returns:
            A list of dataset partitions
        """
        # Generate random indices
        indices = list(range(len(dataset)))
        self.random_generator.shuffle(indices)
        
        # Calculate partition size
        partition_size = len(dataset) // num_partitions
        
        # Create partitions
        partitions = []
        for i in range(num_partitions):
            start_idx = i * partition_size
            end_idx = start_idx + partition_size if i < num_partitions - 1 else len(dataset)
            partition_indices = indices[start_idx:end_idx]
            partitions.append(Subset(dataset, partition_indices))
        
        seed_info = f" with seed={self.seed}" if self.seed is not None else ""
        logger.info(f"Created {num_partitions} IID partitions{seed_info} with ~{partition_size} samples each")
        return partitions


class SeededIIDPartitioner(DatasetPartitioner):
    """
    DEPRECATED: Use IIDPartitioner with a seed parameter instead.
    This class is maintained for backward compatibility.
    """
    def __init__(self, seed: int):
        """
        Initialize the seeded IID partitioner.
        
        Args:
            seed: Random seed for reproducibility
        """
        logger.warning("SeededIIDPartitioner is deprecated. Use IIDPartitioner with seed parameter instead.")
        self.random_generator = random.Random(seed)
        self.seed = seed
    
    def partition(self, dataset: Dataset, num_partitions: int) -> List[Dataset]:
        """
        Partition the dataset in an IID fashion with reproducible randomness.
        
        Args:
            dataset: The dataset to partition
            num_partitions: The number of partitions to create
            
        Returns:
            A list of dataset partitions
        """
        # Create and use an IIDPartitioner instead
        partitioner = IIDPartitioner(seed=self.seed)
        return partitioner.partition(dataset, num_partitions)


class DirichletPartitioner(DatasetPartitioner):
    """
    Non-IID partitioning using Dirichlet distribution (LDA).
    Creates partitions where the class distribution follows a Dirichlet distribution.
    """
    def __init__(self, alpha: float = 0.5):
        """
        Initialize the Dirichlet partitioner.
        
        Args:
            alpha: Concentration parameter for Dirichlet distribution.
                  Lower values create more non-IID partitions.
        """
        self.alpha = alpha
    
    def partition(self, dataset: Dataset, num_partitions: int) -> List[Dataset]:
        """
        Partition the dataset using a Dirichlet distribution on class labels.
        
        Args:
            dataset: The dataset to partition
            num_partitions: The number of partitions to create
            
        Returns:
            A list of dataset partitions
        """
        # Get class labels for all examples
        targets = self._get_targets(dataset)
        num_classes = len(np.unique(targets))
        
        # Create list of indices for each class
        class_indices = [np.where(targets == class_idx)[0] for class_idx in range(num_classes)]
        
        # Sample proportion of each class for each partition using Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(self.alpha, num_partitions), size=num_classes)
        
        # Initialize partitions
        partition_indices = [[] for _ in range(num_partitions)]
        
        # For each class, distribute indices according to proportions
        for class_idx, class_indices_list in enumerate(class_indices):
            num_class_samples = len(class_indices_list)
            num_samples_per_partition = (proportions[class_idx] * num_class_samples).astype(int)
            
            # Adjust to ensure we assign all samples
            num_samples_per_partition[-1] = num_class_samples - np.sum(num_samples_per_partition[:-1])
            
            # Shuffle indices for this class
            np.random.shuffle(class_indices_list)
            
            # Distribute indices to partitions
            start_idx = 0
            for partition_idx, num_samples in enumerate(num_samples_per_partition):
                partition_indices[partition_idx].extend(
                    class_indices_list[start_idx:start_idx + num_samples].tolist()
                )
                start_idx += num_samples
        
        # Create DataLoader subsets
        partitions = [Subset(dataset, indices) for indices in partition_indices]
        
        # Log partition statistics
        class_dist = self._get_class_distribution(partitions, targets, num_classes)
        logger.info(f"Created {num_partitions} Dirichlet partitions with alpha={self.alpha}")
        logger.debug(f"Class distribution across partitions: {class_dist}")
        
        return partitions
    
    def _get_targets(self, dataset: Dataset) -> np.ndarray:
        """
        Extract targets from the dataset.
        
        Args:
            dataset: The dataset
            
        Returns:
            NumPy array of targets
        """
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            if isinstance(targets, torch.Tensor):
                return targets.numpy()
            elif isinstance(targets, list):
                return np.array(targets)
            else:
                return targets
        elif isinstance(dataset, Subset):
            # For Subset, extract targets based on indices
            if hasattr(dataset.dataset, 'targets'):
                targets = dataset.dataset.targets
                if isinstance(targets, torch.Tensor):
                    targets = targets.numpy()
                elif isinstance(targets, list):
                    targets = np.array(targets)
                return targets[dataset.indices]
        
        # If we can't extract targets, try iterating (less efficient)
        logger.warning("Extracting targets by iterating through dataset - this may be slow")
        targets = []
        for _, target in dataset:
            targets.append(target)
        return np.array(targets)
    
    def _get_class_distribution(self, partitions: List[Dataset], targets: np.ndarray, 
                               num_classes: int) -> List[Dict[int, float]]:
        """
        Calculate class distribution for each partition.
        
        Args:
            partitions: List of dataset partitions
            targets: Original targets array
            num_classes: Number of classes
            
        Returns:
            List of dictionaries mapping class index to proportion
        """
        result = []
        for partition in partitions:
            if isinstance(partition, Subset):
                partition_targets = targets[partition.indices]
                class_counts = np.bincount(partition_targets, minlength=num_classes)
                class_props = class_counts / len(partition_targets)
                result.append({i: float(prop) for i, prop in enumerate(class_props) if prop > 0})
        return result


class PathologicalPartitioner(DatasetPartitioner):
    """
    Pathological non-IID partitioning where each client gets data from a limited number of classes.
    In the extreme case, each client gets data from only one class.
    """
    def __init__(self, classes_per_partition: int = 1):
        """
        Initialize PathologicalPartitioner.
        
        Args:
            classes_per_partition: Number of classes to assign to each partition
        """
        self.classes_per_partition = classes_per_partition
    
    def partition(self, dataset: Dataset, num_partitions: int) -> List[Dataset]:
        """
        Partition the dataset in a pathological non-IID fashion.
        
        Args:
            dataset: The dataset to partition
            num_partitions: The number of partitions to create
            
        Returns:
            A list of dataset partitions
        """
        # Get class labels for all examples
        targets = self._get_targets(dataset)
        unique_classes = np.unique(targets)
        num_classes = len(unique_classes)
        
        # Check if we have enough classes for the requested partitioning
        if num_classes < self.classes_per_partition:
            raise ValueError(f"Dataset has only {num_classes} classes, but {self.classes_per_partition} classes per partition requested")
        
        # Create list of indices for each class
        class_indices = {c: np.where(targets == c)[0] for c in unique_classes}
        
        # Calculate how many partitions will get each class
        # Each class should be represented in (num_partitions * classes_per_partition / num_classes) partitions
        partitions_per_class = max(1, int(num_partitions * self.classes_per_partition / num_classes))
        
        # Initialize partition indices
        partition_indices = [[] for _ in range(num_partitions)]
        
        # Assign classes to partitions
        class_assignments = []
        for c in unique_classes:
            # Determine which partitions get this class
            partition_ids = np.random.choice(
                range(num_partitions), 
                size=partitions_per_class, 
                replace=False
            )
            for p_id in partition_ids:
                class_assignments.append((c, p_id))
        
        # Ensure each partition gets at least one class
        partition_class_counts = {p_id: 0 for p_id in range(num_partitions)}
        for c, p_id in class_assignments:
            partition_class_counts[p_id] += 1
        
        # Assign additional classes to partitions with no classes
        empty_partitions = [p_id for p_id, count in partition_class_counts.items() if count == 0]
        if empty_partitions:
            logger.warning(f"{len(empty_partitions)} partitions have no classes assigned, adding random classes")
            for p_id in empty_partitions:
                c = np.random.choice(unique_classes)
                class_assignments.append((c, p_id))
        
        # Distribute samples to partitions
        for c, p_id in class_assignments:
            # Get indices for this class
            indices = class_indices[c]
            
            # Determine how many samples to assign to this partition
            # We want to distribute samples of each class equally among the partitions that have it
            count = len(indices)
            partitions_with_this_class = sum(1 for _, p in class_assignments if p == p_id)
            samples_to_assign = count // partitions_with_this_class
            
            # Assign samples
            partition_indices[p_id].extend(indices[:samples_to_assign])
            
            # Remove assigned indices
            class_indices[c] = indices[samples_to_assign:]
        
        # Create Subset instances
        partitions = [Subset(dataset, indices) for indices in partition_indices]
        
        logger.info(f"Created {num_partitions} pathological partitions with {self.classes_per_partition} classes per partition")
        return partitions
    
    def _get_targets(self, dataset: Dataset) -> np.ndarray:
        """
        Extract targets from the dataset.
        
        Args:
            dataset: The dataset
            
        Returns:
            NumPy array of targets
        """
        if hasattr(dataset, 'targets'):
            targets = dataset.targets
            if isinstance(targets, torch.Tensor):
                return targets.numpy()
            elif isinstance(targets, list):
                return np.array(targets)
            else:
                return targets
        elif isinstance(dataset, Subset):
            # For Subset, extract targets based on indices
            if hasattr(dataset.dataset, 'targets'):
                targets = dataset.dataset.targets
                if isinstance(targets, torch.Tensor):
                    targets = targets.numpy()
                elif isinstance(targets, list):
                    targets = np.array(targets)
                return targets[dataset.indices]
        
        # If we can't extract targets, try iterating (less efficient)
        logger.warning("Extracting targets by iterating through dataset - this may be slow")
        targets = []
        for _, target in dataset:
            targets.append(target)
        return np.array(targets)


class FEMNISTNaturalPartitioner(DatasetPartitioner):
    """
    Use FEMNIST's natural partitioning by writer.
    Each partition corresponds to a different writer.
    
    NOTE: Test set extraction should be done before using this partitioner.
    """
    def __init__(self, data_dir: Union[str, Path], seed: int):
        """
        Initialize the FEMNIST natural partitioner.
        
        Args:
            data_dir: Directory containing the FEMNIST dataset
            seed: Random seed for selection of clients
        """
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.seed = seed
        self.random_generator = random.Random(seed)
    
    def partition(self, available_client_ids: List[str], num_partitions: int, test_sample_paths: Optional[set] = None) -> List[Dataset]:
        """
        Use FEMNIST's natural partitioning after test set extraction.
        
        Args:
            available_client_ids: List of available client IDs to choose from
            num_partitions: Maximum number of partitions to create
            test_sample_paths: Set of sample paths that belong to the test set (to be excluded)
            
        Returns:
            A list of FEMNIST datasets, one per writer
        """
        # Get the mapping directory
        mapping_dir = self.data_dir / "femnist" / "client_data_mappings" / "fed_natural"
        
        # Ensure the directory exists
        if not mapping_dir.exists():
            raise FileNotFoundError(f"FEMNIST natural partitioning directory not found at {mapping_dir}")
        
        # Validate available_client_ids
        if not available_client_ids:
            available_client_ids = [d.name for d in mapping_dir.iterdir() if d.is_dir()]
            
        if not available_client_ids:
            raise ValueError(f"No client directories found in {mapping_dir}")
        
        # Select num_partitions clients
        num_to_select = min(num_partitions, len(available_client_ids))
        selected_clients = self.random_generator.sample(available_client_ids, num_to_select)
        
        if len(selected_clients) < num_partitions:
            logger.warning(f"Only {len(selected_clients)} FEMNIST clients available, requested {num_partitions}")
        
        # Create datasets for each selected client, filtering out test samples if needed
        partitions = []
        for cid in selected_clients:
            # Load the client dataset
            client_dataset = load_femnist_dataset(
                data_dir=self.data_dir / "femnist" / "data",
                mapping=mapping_dir / str(cid),
                name="train"
            )
            
            # If we have test sample paths, filter them out
            if test_sample_paths:
                client_samples = client_dataset._load_dataset()
                filtered_samples = [sample for sample in client_samples if sample[0] not in test_sample_paths]
                
                if not filtered_samples:
                    logger.warning(f"Client {cid} has no training samples after filtering out test samples")
                    continue
                
                # Create a filtered dataset
                from c2m3.experiments.corrected_partition_data import FEMNISTFromSamples
                client_dataset = FEMNISTFromSamples(
                    samples=filtered_samples,
                    data_dir=self.data_dir / "femnist" / "data"
                )
            
            partitions.append(client_dataset)
        
        logger.info(f"Created {len(partitions)} natural FEMNIST partitions")
        return partitions


class FEMNISTIIDPartitioner(DatasetPartitioner):
    """
    Create IID partitions for FEMNIST dataset.
    
    This partitioner divides FEMNIST training samples across clients in an IID fashion.
    NOTE: Test set extraction should be done before using this partitioner.
    """
    def __init__(self, data_dir: Union[str, Path], seed: int, samples_per_partition: Optional[int] = None):
        """
        Initialize the FEMNIST IID partitioner.
        
        Args:
            data_dir: Directory containing the FEMNIST dataset
            seed: Random seed for reproducibility
            samples_per_partition: If provided, each partition will have exactly this many samples
                                 If None, divides the dataset evenly among partitions
        """
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.random_generator = random.Random(seed)
        self.seed = seed
        self.samples_per_partition = samples_per_partition
    
    def partition(self, training_samples: List[Tuple[str, int]], num_partitions: int) -> List[Dataset]:
        """
        Create IID partitions for FEMNIST training data.
        
        Args:
            training_samples: List of (sample_path, label) tuples for training
            num_partitions: Number of partitions to create
            
        Returns:
            A list of FEMNIST datasets, one per partition
        """
        # Shuffle samples for IID distribution
        shuffled_samples = list(training_samples)
        self.random_generator.shuffle(shuffled_samples)
        
        partitions = []
        
        if self.samples_per_partition is not None:
            # Fixed-size partitioning: Each partition has exactly samples_per_partition samples
            samples_needed = self.samples_per_partition * num_partitions
            
            # Check if we have enough samples
            if samples_needed > len(shuffled_samples):
                logger.warning(
                    f"Not enough samples for {num_partitions} partitions with {self.samples_per_partition} samples each. "
                    f"Need {samples_needed}, but only have {len(shuffled_samples)}. Will use all available samples."
                )
                # Fall back to even division
                samples_per_partition = len(shuffled_samples) // num_partitions
            else:
                samples_per_partition = self.samples_per_partition
                # We may not use all samples if fixed size is specified
                shuffled_samples = shuffled_samples[:samples_needed]
        else:
            # Even division: Divide all samples evenly among partitions
            samples_per_partition = len(shuffled_samples) // num_partitions
        
        # Create partitions
        for i in range(num_partitions):
            start_idx = i * samples_per_partition
            end_idx = start_idx + samples_per_partition
            
            # Make sure we don't go beyond available samples
            if end_idx <= len(shuffled_samples):
                partition_samples = shuffled_samples[start_idx:end_idx]
                
                # Create a FEMNIST dataset from these samples
                partition = self._create_dataset_from_samples(partition_samples)
                partitions.append(partition)
        
        if self.samples_per_partition is not None:
            logger.info(f"Created {len(partitions)} IID FEMNIST partitions with exactly {samples_per_partition} samples each")
        else:
            logger.info(f"Created {len(partitions)} IID FEMNIST partitions with ~{samples_per_partition} samples each")
        
        return partitions
    
    def _create_dataset_from_samples(self, samples: List[Tuple[str, int]]) -> Dataset:
        """
        Create a FEMNIST dataset from a list of samples.
        
        Args:
            samples: List of (sample_path, label) tuples
            
        Returns:
            A FEMNIST dataset
        """
        # Create a temporary directory to store the mapping file
        import tempfile
        import csv
        import os
        
        temp_dir = Path(tempfile.mkdtemp())
        train_pt_path = temp_dir / "train.pt"
        
        # Save samples to a PT file for faster loading
        torch.save(samples, train_pt_path)
        
        # Create CSV mapping file
        train_csv_path = temp_dir / "train.csv"
        with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "sample_path", "writer_id", "label_id"])
            for sample_path, label in samples:
                writer.writerow([0, sample_path, 0, label])
        
        # Create and return FEMNIST dataset
        return load_femnist_dataset(
            mapping=temp_dir,
            data_dir=self.data_dir / "femnist" / "data",
            name="train"
        )


class FEMNISTByIDPartitioner(DatasetPartitioner):
    """
    Use existing FEMNIST partitions by client ID.
    
    This partitioner:
    1. Directly accesses existing partitions
    2. Selects partitions by specific client IDs
    3. Does not create any new files
    4. Matches the approach used in basic_flower.ipynb
    """
    def __init__(self, data_dir: Union[str, Path], partition_dir_name: str = "fed_natural", client_ids: List[int] = None):
        """
        Initialize the FEMNIST By-ID partitioner.
        
        Args:
            data_dir: Directory containing the FEMNIST dataset
            partition_dir_name: Name of the partition directory to use (e.g., "fed_natural", "fed_iid")
            client_ids: Specific client IDs to include. If None, will use sequential IDs.
        """
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.partition_dir_name = partition_dir_name
        self.client_ids = client_ids
    
    def partition(self, dataset: Dataset, num_partitions: int) -> List[Dataset]:
        """
        Select partitions by client ID.
        
        The function:
        1. Finds existing partition directory
        2. Creates FEMNIST dataset objects for specified client IDs
        3. Returns these datasets
        
        Args:
            dataset: The dataset (used mainly for transforms)
            num_partitions: Maximum number of partitions to create
            
        Returns:
            A list of FEMNIST datasets, one per client ID
        """
        # Get the mapping directory
        mapping_dir = self.data_dir / "femnist" / "client_data_mappings" / self.partition_dir_name
        
        # Ensure the directory exists
        if not mapping_dir.exists():
            raise FileNotFoundError(f"FEMNIST partition directory not found at {mapping_dir}")
        
        # Get transform from the original dataset if available
        transform = getattr(dataset, 'transform', None)
        if transform is None:
            # Create a default transform that converts PIL Images to tensors
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            logger.info("Using default ToTensor transform for FEMNIST")
        else:
            logger.info("Using transform from dataset")
        
        # Determine client IDs to use
        client_ids = self.client_ids
        if client_ids is None:
            logger.error(f"No client IDs provided. Client IDs must be explicitly specified for FEMNIST partitioning.")
            raise ValueError(
                f"Client IDs must be explicitly provided when using FEMNISTByIDPartitioner. "
                f"Please specify {num_partitions} client IDs in the configuration."
            )
        else:
            # Validate that the number of client IDs matches num_partitions
            if len(client_ids) != num_partitions:
                raise ValueError(
                    f"Number of client IDs ({len(client_ids)}) doesn't match requested partitions ({num_partitions}). "
                    f"Please provide exactly {num_partitions} client IDs."
                )
        
        # Verify all client IDs exist
        for cid in client_ids:
            client_dir = mapping_dir / str(cid)
            if not client_dir.exists():
                raise ValueError(f"Client ID {cid} directory not found at {client_dir}")
        
        # Create datasets for each client ID
        partitions = []
        for cid in client_ids:
            client_dir = mapping_dir / str(cid)
            client_dataset = FEMNIST(
                mapping=client_dir,
                data_dir=self.data_dir / "femnist" / "data",
                name="train",  # Default to train, client will specify
                transform=transform
            )
            partitions.append(client_dataset)
            logger.info(f"Client ID {cid} dataset size: {len(client_dataset)} samples")
        
        logger.info(f"Selected {len(partitions)} FEMNIST partitions by client ID from {self.partition_dir_name}")
        return partitions


class ShakespeareCharacterPartitioner(DatasetPartitioner):
    """
    Use Shakespeare's natural partitioning by character.
    Each partition corresponds to a different character.
    """
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the Shakespeare character partitioner.
        
        Args:
            data_dir: Directory containing the Shakespeare dataset
        """
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
    
    def partition(self, dataset: Dataset, num_partitions: int) -> List[Dataset]:
        """
        Partition Shakespeare dataset by character.
        
        Args:
            dataset: The Shakespeare dataset
            num_partitions: Maximum number of partitions to create
            
        Returns:
            A list of Shakespeare datasets, one per character
        """
        # This is a placeholder for the actual implementation
        # In a real implementation, we would:
        # 1. Load character mappings
        # 2. Create subdatasets per character
        logger.warning("ShakespeareCharacterPartitioner is a placeholder and not fully implemented")
        
        # For now, just return dummy partitions
        partitions = []
        for i in range(min(num_partitions, 5)):  # Assume 5 characters max for placeholder
            # Create a subset with random indices (simplified placeholder)
            indices = list(range(i, len(dataset), 5))
            partition = Subset(dataset, indices)
            partitions.append(partition)
        
        return partitions


# Register partitioners with the registry
PartitionerRegistry.register("default", "iid", IIDPartitioner)
PartitionerRegistry.register("default", "seeded_iid", SeededIIDPartitioner)
PartitionerRegistry.register("default", "dirichlet", DirichletPartitioner)
PartitionerRegistry.register("default", "pathological", PathologicalPartitioner)
PartitionerRegistry.register("femnist", "natural", FEMNISTNaturalPartitioner)
PartitionerRegistry.register("femnist", "iid", FEMNISTIIDPartitioner)
PartitionerRegistry.register("femnist", "by_id", FEMNISTByIDPartitioner)
PartitionerRegistry.register("shakespeare", "natural", ShakespeareCharacterPartitioner) 