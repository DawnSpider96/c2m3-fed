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
from torch.utils.data import Dataset, Subset
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
        random.shuffle(indices)
        
        # Calculate partition size
        partition_size = len(dataset) // num_partitions
        
        # Create partitions
        partitions = []
        for i in range(num_partitions):
            start_idx = i * partition_size
            end_idx = start_idx + partition_size if i < num_partitions - 1 else len(dataset)
            partition_indices = indices[start_idx:end_idx]
            partitions.append(Subset(dataset, partition_indices))
        
        logger.info(f"Created {num_partitions} IID partitions with ~{partition_size} samples each")
        return partitions


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
    """
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the FEMNIST natural partitioner.
        
        Args:
            data_dir: Directory containing the FEMNIST dataset
        """
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
    
    def partition(self, dataset: Dataset, num_partitions: int) -> List[Dataset]:
        """
        Use FEMNIST's natural partitioning.
        
        Args:
            dataset: The FEMNIST dataset (only used for transforms)
            num_partitions: Maximum number of partitions to create
            
        Returns:
            A list of FEMNIST datasets, one per writer
        """
        # Get the mapping directory
        mapping_dir = self.data_dir / "femnist" / "client_data_mappings"
        
        # Ensure the directory exists
        if not mapping_dir.exists():
            raise FileNotFoundError(f"FEMNIST client mappings not found at {mapping_dir}")
        
        # Get available client mappings
        client_files = list(mapping_dir.glob("*.csv"))
        if not client_files:
            raise ValueError(f"No client mapping files found in {mapping_dir}")
        
        # Select num_partitions clients
        selected_clients = client_files[:num_partitions]
        if len(selected_clients) < num_partitions:
            logger.warning(f"Only {len(selected_clients)} FEMNIST clients available, requested {num_partitions}")
        
        # Get transform from the original dataset if available
        transform = getattr(dataset, 'transform', None)
        
        # Create datasets for each selected client
        partitions = []
        for client_file in selected_clients:
            client_dataset = FEMNIST(
                mapping=client_file.parent,
                data_dir=self.data_dir / "femnist" / "data",
                name="train",
                transform=transform
            )
            partitions.append(client_dataset)
        
        logger.info(f"Created {len(partitions)} natural FEMNIST partitions")
        return partitions


class FEMNISTIIDPartitioner(DatasetPartitioner):
    """
    Create IID partitions for FEMNIST dataset.
    
    This partitioner uses the centralized mapping file to create IID partitions
    """
    def __init__(self, data_dir: Union[str, Path], partition_dir_name: str = "fed_iid"):
        """
        Initialize the FEMNIST IID partitioner.
        
        Args:
            data_dir: Directory containing the FEMNIST dataset
            partition_dir_name: Name for the IID partition directory
        """
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.partition_dir_name = partition_dir_name
    
    def partition(self, dataset: Dataset, num_partitions: int) -> List[Dataset]:
        """
        Create IID partitions for FEMNIST.
        
        The function:
        1. Loads a centralized dataset
        2. Randomly shuffles and divides data across clients
        3. Creates mapping files for each client
        4. Returns FEMNIST dataset objects for each client
        
        Args:
            dataset: The dataset to partition (used mainly for transforms)
            num_partitions: Number of partitions to create
            
        Returns:
            A list of FEMNIST datasets, one per client
        """
        # Step 1: Set up directories
        root_mapping_dir = self.data_dir / "femnist" / "client_data_mappings"
        partition_dir = root_mapping_dir / self.partition_dir_name
        
        # Create partition directory if it doesn't exist
        partition_dir.mkdir(exist_ok=True, parents=True)
        
        # Step 2: Load the centralized dataset (all samples)
        # We'll use the existing centralized mapping if available
        centralized_mapping = root_mapping_dir / "centralized" / "0"
        if not centralized_mapping.exists():
            raise FileNotFoundError(f"Centralized mapping not found at {centralized_mapping}")
        
        # Load centralized dataset for both train and test
        transform = getattr(dataset, 'transform', None)
        
        # Collect all samples
        all_samples = {}
        for split in ["train", "test"]:
            central_dataset = FEMNIST(
                mapping=centralized_mapping,
                data_dir=self.data_dir / "femnist" / "data",
                name=split,
                transform=transform
            )
            all_samples[split] = central_dataset._load_dataset()
        
        # Step 3: Create IID partitions
        partitions = []
        
        for split in ["train", "test"]:
            samples = all_samples[split]
            # Shuffle samples for IID distribution
            shuffled_samples = list(samples)
            random.shuffle(shuffled_samples)
            
            # Calculate samples per partition
            samples_per_partition = len(shuffled_samples) // num_partitions
            
            # Distribute samples
            for i in range(num_partitions):
                # Create client directory if not exists
                client_dir = partition_dir / str(i)
                client_dir.mkdir(exist_ok=True)
                
                # Get samples for this client
                start_idx = i * samples_per_partition
                end_idx = start_idx + samples_per_partition if i < num_partitions - 1 else len(shuffled_samples)
                client_samples = shuffled_samples[start_idx:end_idx]
                
                # Create CSV mapping file
                csv_path = (client_dir / split).with_suffix(".csv")
                
                # Create CSV mapping
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["user_id", "sample_path", "writer_id", "label_id"])
                    
                    # For each sample, we only have the path and label from FEMNIST._load_dataset()
                    # We'll use placeholder values for user_id and writer_id
                    for sample_path, label in client_samples:
                        writer.writerow([i, sample_path, i, label])
                
                # Create PT file for faster loading
                pt_path = (client_dir / split).with_suffix(".pt")
                torch.save(client_samples, pt_path)
        
        # Step 4: Create and return FEMNIST dataset objects for each partition
        for i in range(num_partitions):
            client_dir = partition_dir / str(i)
            client_dataset = FEMNIST(
                mapping=client_dir,
                data_dir=self.data_dir / "femnist" / "data",
                name="train",  # Default to train, client will specify
                transform=transform
            )
            partitions.append(client_dataset)
        
        logger.info(f"Created {num_partitions} IID FEMNIST partitions with ~{samples_per_partition} samples each")
        return partitions


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
        
        # Determine client IDs to use
        client_ids = self.client_ids
        if client_ids is None:
            # Use all available client directories, up to num_partitions
            client_dirs = sorted([d for d in mapping_dir.iterdir() if d.is_dir()])
            client_ids = [int(d.name) for d in client_dirs[:num_partitions]]
        else:
            # Limit to num_partitions if needed
            client_ids = client_ids[:num_partitions]
        
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
        
        logger.info(f"Selected {len(partitions)} FEMNIST partitions by client ID from {self.partition_dir_name}")
        return partitions


class ShakespeareCharacterPartitioner(DatasetPartitioner):
    """
    Partition Shakespeare dataset by character.
    Each partition corresponds to a different character in the plays.
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
            A list of dataset partitions, one per character
        """
        # Implementation for Shakespeare dataset
        # This is a placeholder - the actual implementation would depend on the structure
        # of your Shakespeare dataset
        logger.warning("Shakespeare character partitioning not fully implemented")
        
        # For now, just return IID partitions
        iid_partitioner = IIDPartitioner()
        return iid_partitioner.partition(dataset, num_partitions)


# Register partitioning strategies
PartitionerRegistry.register("default", "iid", IIDPartitioner)
PartitionerRegistry.register("default", "dirichlet", DirichletPartitioner)
PartitionerRegistry.register("default", "pathological", PathologicalPartitioner)
PartitionerRegistry.register("femnist", "natural", FEMNISTNaturalPartitioner)
PartitionerRegistry.register("femnist", "iid", FEMNISTIIDPartitioner)
PartitionerRegistry.register("femnist", "by_id", FEMNISTByIDPartitioner)
PartitionerRegistry.register("shakespeare", "natural", ShakespeareCharacterPartitioner) 