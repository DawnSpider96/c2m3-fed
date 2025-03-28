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
        mapping_dir = self.data_dir / "femnist" / "client_data_mappings" / "fed_natural"
        
        if not mapping_dir.exists():
            raise FileNotFoundError(f"FEMNIST natural partitioning directory not found at {mapping_dir}")
        
        if not available_client_ids:
            available_client_ids = [d.name for d in mapping_dir.iterdir() if d.is_dir()]
        
        num_to_select = min(num_partitions, len(available_client_ids))
        selected_clients = self.random_generator.sample(available_client_ids, num_to_select)
        
        if len(selected_clients) < num_partitions:
            logger.warning(f"Only {len(selected_clients)} FEMNIST clients available, requested {num_partitions}")
        
        partitions = []
        for cid in selected_clients:
            client_dataset = load_femnist_dataset(
                data_dir=self.data_dir / "femnist" / "data",
                mapping=mapping_dir / str(cid),
                name="train"
            )
            
            if test_sample_paths:
                client_samples = client_dataset._load_dataset()
                filtered_samples = [sample for sample in client_samples if sample[0] not in test_sample_paths]
                
                if not filtered_samples:
                    logger.warning(f"Client {cid} has no training samples after filtering out test samples")
                    continue
                
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
            samples_needed = self.samples_per_partition * num_partitions
            
            if samples_needed > len(shuffled_samples):
                logger.warning(
                    f"Not enough samples for {num_partitions} partitions with {self.samples_per_partition} samples each. "
                    f"Need {samples_needed}, but only have {len(shuffled_samples)}. Will use all available samples."
                )
                samples_per_partition = len(shuffled_samples) // num_partitions
            else:
                samples_per_partition = self.samples_per_partition
                shuffled_samples = shuffled_samples[:samples_needed]
        else:
            samples_per_partition = len(shuffled_samples) // num_partitions
        
        for i in range(num_partitions):
            start_idx = i * samples_per_partition
            end_idx = start_idx + samples_per_partition
            
            if end_idx <= len(shuffled_samples):
                partition_samples = shuffled_samples[start_idx:end_idx]
                
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
        
        torch.save(samples, train_pt_path)
        
        train_csv_path = temp_dir / "train.csv"
        with open(train_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "sample_path", "writer_id", "label_id"])
            for sample_path, label in samples:
                writer.writerow([0, sample_path, 0, label])
        
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
        mapping_dir = self.data_dir / "femnist" / "client_data_mappings" / self.partition_dir_name
        
        if not mapping_dir.exists():
            raise FileNotFoundError(f"FEMNIST partition directory not found at {mapping_dir}")
        
        transform = getattr(dataset, 'transform', None)
        if transform is None:
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
            if len(client_ids) != num_partitions:
                raise ValueError(
                    f"Number of client IDs ({len(client_ids)}) doesn't match requested partitions ({num_partitions}). "
                    f"Please provide exactly {num_partitions} client IDs."
                )
        
        for cid in client_ids:
            client_dir = mapping_dir / str(cid)
            if not client_dir.exists():
                raise ValueError(f"Client ID {cid} directory not found at {client_dir}")
        
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


class FEMNISTPathologicalPartitioner(DatasetPartitioner):
    """
    Pathological non-IID partitioning for FEMNIST dataset where each client gets data from exactly one class.
    
    This partitioner ensures:
    1. Each partition contains samples from exactly one class
    2. Classes with fewer than samples_per_partition samples are ignored
    3. If there aren't enough classes with sufficient samples, an error is raised
    """
    def __init__(self, data_dir: Union[str, Path], seed: int, samples_per_partition: int):
        """
        Initialize the FEMNIST pathological partitioner.
        
        Args:
            data_dir: Directory containing the FEMNIST dataset
            seed: Random seed for reproducibility
            samples_per_partition: Number of samples each partition should have
        """
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.random_generator = random.Random(seed)
        self.seed = seed
        self.samples_per_partition = samples_per_partition
    
    def partition(self, training_samples: List[Tuple[str, int]], num_partitions: int) -> List[Dataset]:
        """
        Create pathological partitions for FEMNIST training data where each partition has samples from exactly one class.
        
        Args:
            training_samples: List of (sample_path, label) tuples for training
            num_partitions: Number of partitions to create
            
        Returns:
            A list of FEMNIST datasets, one per partition
        """
        # Group samples by class
        samples_by_class = {}
        for sample_path, label in training_samples:
            if label not in samples_by_class:
                samples_by_class[label] = []
            samples_by_class[label].append((sample_path, label))
        
        # Shuffle the classes to introduce randomness
        classes = list(samples_by_class.keys())
        self.random_generator.shuffle(classes)
        
        # Shuffle samples within each class
        for class_label in samples_by_class:
            self.random_generator.shuffle(samples_by_class[class_label])
        
        # Handle samples_per_partition as a list
        if isinstance(self.samples_per_partition, int):
            samples_per_partition_list = [self.samples_per_partition] * num_partitions
        else:
            samples_per_partition_list = self.samples_per_partition
            
        if len(samples_per_partition_list) != num_partitions:
            raise ValueError(f"Length of samples_per_partition ({len(samples_per_partition_list)}) "
                            f"doesn't match requested partitions ({num_partitions}).")
        
        # Check if we have enough classes with sufficient samples
        viable_classes = []
        for class_label in classes:
            # A class is viable if it has enough samples for at least one partition
            min_samples_needed = min(samples_per_partition_list)
            if len(samples_by_class[class_label]) >= min_samples_needed:
                viable_classes.append(class_label)
        
        if len(viable_classes) < num_partitions:
            raise ValueError(f"Not enough classes with sufficient samples. "
                            f"Found {len(viable_classes)} viable classes, need {num_partitions}.")
        
        # Select classes for each partition
        selected_classes = viable_classes[:num_partitions]
        
        # Create partitions
        partitions = []
        for i, class_label in enumerate(selected_classes):
            # Get the number of samples needed for this partition
            samples_needed = samples_per_partition_list[i]
            
            # Check if this class has enough samples
            if len(samples_by_class[class_label]) < samples_needed:
                raise ValueError(f"Class {class_label} has only {len(samples_by_class[class_label])} samples, "
                                f"but partition {i} needs {samples_needed}.")
            
            # Take exactly samples_needed samples from this class
            partition_samples = samples_by_class[class_label][:samples_needed]
            
            # Create dataset from these samples
            from c2m3.experiments.corrected_partition_data import FEMNISTFromSamples
            partition = FEMNISTFromSamples(
                samples=partition_samples,
                data_dir=self.data_dir / "femnist" / "data",
                name="train"
            )
            partitions.append(partition)
            
            logger.info(f"Partition {i}: Class {class_label}, {len(partition_samples)} samples")
        
        logger.info(f"Created {len(partitions)} pathological FEMNIST partitions, each with exactly one class")
        
        return partitions


class FEMNISTHeldOutClassPartitioner(DatasetPartitioner):
    """
    Special non-IID partitioning for FEMNIST where:
    1. One class is selected to be the "held out" class
    2. N-1 partitions receive IID samples from all classes EXCEPT the held out class
    3. The Nth partition gets all samples from the held out class, plus additional IID samples if needed
    
    This creates a scenario where most models never see a particular class during training,
    while one model specializes in that class.
    """
    def __init__(self, data_dir: Union[str, Path], seed: int, samples_per_partition: Optional[int] = None):
        """
        Initialize the FEMNIST held-out class partitioner.
        
        Args:
            data_dir: Directory containing the FEMNIST dataset
            seed: Random seed for reproducibility
            samples_per_partition: Optional fixed number of samples per partition.
                                 If None, partitions will be of approximately equal size.
        """
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        self.random_generator = random.Random(seed)
        self.seed = seed
        self.samples_per_partition = samples_per_partition
    
    def partition(self, training_samples: List[Tuple[str, int]], num_partitions: int) -> List[Dataset]:
        """
        Create partitions where N-1 partitions have no samples from one class,
        and the Nth partition has all samples from that class.
        
        Args:
            training_samples: List of (sample_path, label) tuples for training
            num_partitions: Number of partitions to create
            
        Returns:
            A list of FEMNIST datasets, one per partition
        """
        if num_partitions < 2:
            raise ValueError("Need at least 2 partitions for held-out class partitioning")
        
        samples_by_class = {}
        for sample_path, label in training_samples:
            if label not in samples_by_class:
                samples_by_class[label] = []
            samples_by_class[label].append((sample_path, label))
        
        all_classes = list(samples_by_class.keys())
        num_classes = len(all_classes)
        
        self.random_generator.shuffle(all_classes)
        held_out_class = all_classes[0]
        logger.info(f"Selected class {held_out_class} as the held-out class")
        
        held_out_samples = samples_by_class[held_out_class]
        remaining_samples = []
        for cls, samples in samples_by_class.items():
            if cls != held_out_class:
                remaining_samples.extend(samples)
        
        self.random_generator.shuffle(remaining_samples)
        
        if self.samples_per_partition is not None:
            samples_per_regular_partition = self.samples_per_partition
            total_regular_samples_needed = samples_per_regular_partition * (num_partitions - 1)
            
            if total_regular_samples_needed > len(remaining_samples):
                logger.warning(
                    f"Not enough samples for {num_partitions-1} regular partitions with {samples_per_regular_partition} samples each. "
                    f"Need {total_regular_samples_needed}, but only have {len(remaining_samples)}. Will adjust partition sizes."
                )
                samples_per_regular_partition = len(remaining_samples) // (num_partitions - 1)
        else:
            samples_per_regular_partition = len(remaining_samples) // (num_partitions - 1)
        
        regular_partitions = []
        
        for i in range(num_partitions - 1):
            start_idx = i * samples_per_regular_partition
            end_idx = start_idx + samples_per_regular_partition
            
            if end_idx <= len(remaining_samples):
                partition_samples = remaining_samples[start_idx:end_idx]
                
                from c2m3.experiments.corrected_partition_data import FEMNISTFromSamples
                partition = FEMNISTFromSamples(
                    samples=partition_samples,
                    data_dir=self.data_dir / "femnist" / "data",
                    name="train"
                )
                
                regular_partitions.append(partition)
        
        specialist_samples = list(held_out_samples)
        
        if self.samples_per_partition is not None and len(specialist_samples) < self.samples_per_partition:
            additional_needed = self.samples_per_partition - len(specialist_samples)
            
            remaining_idx = (num_partitions - 1) * samples_per_regular_partition
            additional_samples = remaining_samples[remaining_idx:remaining_idx + additional_needed]
            
            if len(additional_samples) < additional_needed:
                more_needed = additional_needed - len(additional_samples)
                
                cycle_start = 0
                while len(additional_samples) < additional_needed and cycle_start < remaining_idx:
                    cycle_end = min(cycle_start + more_needed, remaining_idx)
                    additional_samples.extend(remaining_samples[cycle_start:cycle_end])
                    cycle_start = cycle_end
                
                additional_samples = additional_samples[:additional_needed]
            
            specialist_samples.extend(additional_samples)
        
        from c2m3.experiments.corrected_partition_data import FEMNISTFromSamples
        specialist_partition = FEMNISTFromSamples(
            samples=specialist_samples,
            data_dir=self.data_dir / "femnist" / "data",
            name="train"
        )
        
        partitions = regular_partitions + [specialist_partition]
        
        logger.info(f"Created {len(regular_partitions)} regular partitions with ~{samples_per_regular_partition} samples each")
        logger.info(f"Created 1 specialist partition with {len(specialist_samples)} samples "
                    f"({len(held_out_samples)} from held-out class {held_out_class}, "
                    f"{len(specialist_samples)-len(held_out_samples)} from other classes)")
        
        return partitions


PartitionerRegistry.register("femnist", "natural", FEMNISTNaturalPartitioner)
PartitionerRegistry.register("femnist", "iid", FEMNISTIIDPartitioner)
PartitionerRegistry.register("femnist", "by_id", FEMNISTByIDPartitioner)
PartitionerRegistry.register("femnist", "pathological", FEMNISTPathologicalPartitioner)
PartitionerRegistry.register("femnist", "held_out_class", FEMNISTHeldOutClassPartitioner) 