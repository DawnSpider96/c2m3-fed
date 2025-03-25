import os
import json
import time
import logging
from typing import Dict, List, Union, Tuple, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
import pandas as pd

# Flower imports
import flwr
from flwr.common import NDArrays, Scalar, Parameters, ndarrays_to_parameters
from flwr.client.client import Client
from flwr.server import History, ServerConfig
from flwr.server.strategy import FedAvgM as FedAvg

# C2M3 imports
from c2m3.utils.utils import set_seed
from c2m3.common.client_utils import (
    load_femnist_dataset, 
    get_network_generator_cnn as get_network_generator,
    train_femnist,
    test_femnist,
    save_history,
    get_model_parameters,
    set_model_parameters
)

from c2m3.experiments.corrected_partition_data import partition_data as corrected_partition_data
from c2m3.experiments.corrected_partition_data import FEMNISTFromSamples

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FederatedExperimentConfig:
    """Configuration for federated learning experiments"""
    # Required parameters (no defaults)
    experiment_name: str
    dataset_name: str  # 'femnist', 'cifar10', 'cifar100'
    model_name: str  # 'resnet18', 'cnn'
    
    # Parameters with default values
    seed: int = 42
    data_dir: str = str(Path(__file__).parent.parent / "data")
    
    # Training configuration
    num_rounds: int = 10  # Number of federation rounds
    local_epochs: int = 1  # Number of local epochs per round
    batch_size: int = 64
    learning_rate: float = 0.01
    weight_decay: float = 0.0001
    
    # Client configuration
    num_total_clients: int = 20  # Total number of available clients
    num_clients_per_round: int = 5  # Number of clients sampled each round
    min_train_samples: int = 10  # Minimum number of training samples per client 
    
    # Aggregation configuration
    server_learning_rate: float = 1.0  # Learning rate for server aggregation (FedAvg)
    server_momentum: float = 0.0  # Momentum for server aggregation (FedAvg)
    aggregation_frequency: int = 1  # How often to aggregate (every X rounds)
    
    # Data distribution
    data_distribution: str = "natural"  # 'iid', 'dirichlet', 'pathological', 'natural'
    non_iid_alpha: float = 0.5  # Dirichlet alpha parameter for non-IID distribution
    classes_per_partition: int = 1  # For pathological partitioning, how many classes per client
    samples_per_partition: Optional[int] = None  # Number of samples per partition, None for equal division
    
    # Evaluation
    eval_batch_size: int = 128
    num_evaluation_clients: int = 0  # Number of clients used for validation (0 means centralized evaluation)
    
    # Output
    output_dir: str = "./results"
    save_models: bool = False
    save_results: bool = True
    
    def __post_init__(self):
        """Validate config and set derived parameters"""
        # Validate number of clients
        if self.num_clients_per_round > self.num_total_clients:
            raise ValueError(f"num_clients_per_round ({self.num_clients_per_round}) cannot be greater than num_total_clients ({self.num_total_clients})")
        
        # Validate aggregation frequency
        if self.aggregation_frequency <= 0:
            raise ValueError(f"aggregation_frequency must be positive, got {self.aggregation_frequency}")
        
        # Validate server learning rate
        if self.server_learning_rate <= 0:
            raise ValueError(f"server_learning_rate must be positive, got {self.server_learning_rate}")


class FlowerRayClient(flwr.client.NumPyClient):
    """Flower client for federated learning"""

    def __init__(
        self,
        cid: int,
        partition_dir: Path,
        model_generator: Callable[[], nn.Module],
        data_dir: Path
    ) -> None:
        """Initialize the client with its unique id and the folder to load data from.

        Parameters
        ----------
            cid (int): Unique client id for a client used to map it to its data partition
            partition_dir (Path): The directory containing data for each client/client id
            model_generator (Callable[[], Module]): The model generator function
            data_dir (Path): Directory containing the raw data files
        """
        self.cid = cid
        logger.info(f"Initializing client {self.cid}")
        self.partition_dir = partition_dir
        self.device = str(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.model_generator = model_generator
        self.properties: dict[str, Scalar] = {
            "tensor_type": "numpy.ndarray",
            "partition": str(self.partition_dir),
            "cid": self.cid
        }
        self.data_dir = data_dir

    def set_parameters(self, parameters: NDArrays) -> nn.Module:
        """Load weights inside the network.

        Parameters
        ----------
            parameters (NDArrays): set of weights to be loaded.

        Returns
        -------
            [Module]: Network with new set of weights.
        """
        net = self.model_generator()
        return set_model_parameters(net, parameters)

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return weights from a given model.

        If no model is passed, then a local model is created.
        This can be used to initialise a model in the server.
        The config param is not used but is mandatory in Flower.

        Parameters
        ----------
            config (dict[int, Scalar]): dictionary containing configuration info.

        Returns
        -------
            NDArrays: weights from the model.
        """
        net = self.model_generator()
        return get_model_parameters(net)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict]:
        """Receive and train a model on the local client data.

        Parameters
        ----------
            parameters (NDArrays): Pytorch model parameters
            config (dict[str, Scalar]): dictionary describing the training parameters

        Returns
        -------
            tuple[NDArrays, int, dict]: Returns the updated model, the size of the local
                dataset and other metrics
        """
        # Only create model right before training/testing to lower memory usage when idle
        net = self.set_parameters(parameters)
        net.to(self.device)

        train_loader: DataLoader = self._create_data_loader(config, name="train")
        train_loss = self._train(net, train_loader=train_loader, config=config)
        return get_model_parameters(net), len(train_loader.dataset), {"train_loss": train_loss}

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict]:
        """Receive and test a model on the local client data.

        Parameters
        ----------
            parameters (NDArrays): Pytorch model parameters
            config (dict[str, Scalar]): dictionary describing the testing parameters

        Returns
        -------
            tuple[float, int, dict]: Returns the loss accumulated during testing, the
                size of the local dataset and other metrics such as accuracy
        """
        net = self.set_parameters(parameters)
        net.to(self.device)

        test_loader: DataLoader = self._create_data_loader(config, name="test")
        loss, accuracy = self._test(net, test_loader=test_loader, config=config)
        return loss, len(test_loader.dataset), {"local_accuracy": accuracy}

    def _create_data_loader(self, config: dict[str, Scalar], name: str) -> DataLoader:
        """Create the data loader using the specified config parameters.

        Parameters
        ----------
            config (dict[str, Scalar]): dictionary containing dataloader and dataset parameters
            name (str): Load the training or testing set for the client

        Returns
        -------
            DataLoader: A pytorch dataloader iterable for training/testing
        """
        batch_size = int(config["batch_size"])
        num_workers = int(config.get("num_workers", 0))
        dataset = self._load_dataset(name)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=(name == "train"),
        )

    def _load_dataset(self, name: str) -> Dataset:
        """Load dataset for client from mapping file.
        
        Parameters
        ----------
            name (str): 'train' or 'test' to specify which dataset to load
            
        Returns
        -------
            Dataset: The loaded dataset
        """
        full_file: Path = self.partition_dir / str(self.cid)
        return load_femnist_dataset(
            mapping=full_file,
            name=name,
            data_dir=self.data_dir,
        )

    def _train(
        self, net: nn.Module, train_loader: DataLoader, config: dict[str, Scalar]
    ) -> float:
        """Train the network on local data.
        
        Parameters
        ----------
            net (nn.Module): The neural network to train
            train_loader (DataLoader): DataLoader with training data
            config (dict[str, Scalar]): Configuration parameters
            
        Returns
        -------
            float: The training loss
        """
        return train_femnist(
            net=net,
            train_loader=train_loader,
            epochs=int(config["local_epochs"]),
            device=self.device,
            optimizer=torch.optim.AdamW(
                net.parameters(),
                lr=float(config["learning_rate"]),
                weight_decay=float(config["weight_decay"]),
            ),
            criterion=torch.nn.CrossEntropyLoss(),
            max_batches=int(config.get("max_batches", None)),
        )

    def _test(
        self, net: nn.Module, test_loader: DataLoader, config: dict[str, Scalar]
    ) -> tuple[float, float]:
        """Test the network on local data.
        
        Parameters
        ----------
            net (nn.Module): The neural network to test
            test_loader (DataLoader): DataLoader with test data
            config (dict[str, Scalar]): Configuration parameters
            
        Returns
        -------
            tuple[float, float]: The test loss and accuracy
        """
        return test_femnist(
            net=net,
            test_loader=test_loader,
            device=self.device,
            criterion=torch.nn.CrossEntropyLoss(),
            max_batches=int(config.get("max_batches", None)),
        )

    def get_properties(self, config: dict[str, Scalar]) -> dict[str, Scalar]:
        """Return properties for this client.

        Parameters
        ----------
            config (dict[str, Scalar]): Options to be used for selecting specific properties.

        Returns
        -------
            dict[str, Scalar]: Returned properties.
        """
        return self.properties

    def get_train_set_size(self) -> int:
        """Return the client train set size.

        Returns
        -------
            int: train set size of the client.
        """
        return len(self._load_dataset("train"))

    def get_test_set_size(self) -> int:
        """Return the client test set size.

        Returns
        -------
            int: test set size of the client.
        """
        return len(self._load_dataset("test"))


def get_flower_client_generator(
    model_generator: Callable[[], nn.Module],
    partition_dir: Path,
    data_dir: Path,
    mapping_fn: Callable[[int], int] | None = None,
) -> Callable[[str], FlowerRayClient]:
    """Create a client generator function for Flower simulation.

    Parameters
    ----------
        model_generator (Callable[[], Module]): model generator function.
        partition_dir (Path): directory containing the partition.
        data_dir (Path): directory containing the raw data files.
        mapping_fn (Optional[Callable[[int], int]]): function mapping sorted/filtered ids to real cid.

    Returns
    -------
        Callable[[str], FlowerRayClient]: client generator function.
    """

    def client_fn(cid: str) -> FlowerRayClient:
        """Create a single client instance given the client id `cid`.

        Parameters
        ----------
            cid (str): client id, Flower requires this to be of type str.

        Returns
        -------
            FlowerRayClient: client instance.
        """
        return FlowerRayClient(
            cid=mapping_fn(int(cid)) if mapping_fn is not None else int(cid),
            partition_dir=partition_dir,
            model_generator=model_generator,
            data_dir=data_dir
        )

    return client_fn


def fit_client_seeded(
    client: FlowerRayClient,
    params: NDArrays,
    conf: dict[str, Any],
    seed: int,
    **kwargs: Any,
) -> tuple[NDArrays, int, dict]:
    """Wrap to always seed client training for reproducibility.
    
    Parameters
    ----------
        client (FlowerRayClient): The client to train
        params (NDArrays): Model parameters
        conf (dict[str, Any]): Configuration dictionary
        seed (int): Random seed
        
    Returns
    -------
        tuple[NDArrays, int, dict]: Client fit results
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return client.fit(params, conf, **kwargs)


class FederatedExperimentRunner:
    """Main class for running federated learning experiments"""
    def __init__(self, config: FederatedExperimentConfig):
        self.config = config
        self.experiment_name = config.experiment_name
        
        # Set random seed for reproducibility
        set_seed(config.seed)
        
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create output directory for results
        self.output_dir = Path(config.output_dir) / self.experiment_name
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save configuration
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=4)
            
        # Initialize results dictionary
        self.results = {
            "config": asdict(self.config),
            "rounds": [],
            "final_model": {},
            "runtime": 0
        }
        
        # Initialize components
        self.model_generator = None
        self.federated_partition_dir = None
        self.client_ids = []
        self.test_dataset = None
        self.global_test_loader = None
        self.parameters_for_each_round = None
        self.history = None
        
    def setup(self):
        """Setup experiment - load datasets, prepare client IDs, etc."""
        logger.info(f"Setting up federated experiment '{self.experiment_name}'...")
        
        # 1. Setup data partitioning directory based on dataset and distribution
        data_dir = Path(self.config.data_dir)
        
        if self.config.dataset_name == "femnist":
            dataset_dir = data_dir / "femnist"
            self.federated_partition_dir = dataset_dir / "client_data_mappings" / f"fed_{self.config.data_distribution}"
            logger.info(f"Using FEMNIST partition: {self.federated_partition_dir}")
            
            # Make sure data exists
            if not self.federated_partition_dir.exists():
                raise ValueError(f"Partition directory does not exist: {self.federated_partition_dir}")
            
            # 2. Get model generator based on model name
            if self.config.model_name == "cnn":
                self.model_generator = get_network_generator()
                logger.info("Using CNN model for FEMNIST")
            else:
                raise ValueError(f"Unsupported model for FEMNIST: {self.config.model_name}")
            
            # 3. Sample client IDs
            self.client_ids = self._sample_client_ids()
            logger.info(f"Selected {len(self.client_ids)} client IDs for the experiment")
            
            # 4. Setup global test dataset for central evaluation
            # For now, we'll use a specific client's test data
            # In a real implementation, you'd want a separate held-out test set
            test_client = FlowerRayClient(
                cid=self.client_ids[0],
                partition_dir=self.federated_partition_dir,
                model_generator=self.model_generator,
                data_dir=data_dir
            )
            self.test_dataset = test_client._load_dataset("test")
            self.global_test_loader = DataLoader(
                self.test_dataset,
                batch_size=self.config.eval_batch_size,
                shuffle=False
            )
            logger.info(f"Created global test loader with {len(self.test_dataset)} samples")
            
        else:
            raise ValueError(f"Unsupported dataset: {self.config.dataset_name}")
        
        logger.info("Federated experiment setup complete.")
    
    def _sample_client_ids(self) -> List[int]:
        """Sample client IDs based on configuration.
        
        Returns:
            List[int]: List of client IDs to use in the experiment
        """
        # Get all available client IDs by scanning the partition directory
        all_client_ids = []
        for client_path in self.federated_partition_dir.iterdir():
            if client_path.is_file() and client_path.name.isdigit():
                client_id = int(client_path.name)
                
                # Create a temporary client to check the number of training samples
                temp_client = FlowerRayClient(
                    cid=client_id,
                    partition_dir=self.federated_partition_dir,
                    model_generator=self.model_generator,
                    data_dir=Path(self.config.data_dir)
                )
                
                # Only include clients with at least min_train_samples
                if temp_client.get_train_set_size() >= self.config.min_train_samples:
                    all_client_ids.append(client_id)
        
        logger.info(f"Found {len(all_client_ids)} client IDs with sufficient training data")
        
        # If we don't have enough clients, use all available ones
        if len(all_client_ids) <= self.config.num_total_clients:
            logger.warning(f"Requested {self.config.num_total_clients} clients but only {len(all_client_ids)} available with sufficient data")
            return all_client_ids
        
        # Otherwise, sample the requested number of clients
        random_gen = random.Random(self.config.seed)
        selected_clients = random_gen.sample(all_client_ids, self.config.num_total_clients)
        return selected_clients
    
    def _create_federated_train_config(self) -> Dict[str, Any]:
        """Create training configuration for federated clients.
        
        Returns:
            Dict[str, Any]: Configuration dictionary for client training
        """
        return {
            "local_epochs": self.config.local_epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "num_workers": 0,
            "max_batches": None  # Use all batches
        }
    
    def _create_test_config(self) -> Dict[str, Any]:
        """Create testing configuration for federated clients.
        
        Returns:
            Dict[str, Any]: Configuration dictionary for client evaluation
        """
        return {
            "batch_size": self.config.eval_batch_size,
            "num_workers": 0,
            "max_batches": None  # Use all batches
        }
    
    def _evaluate_global_model(self, parameters: NDArrays) -> Dict[str, float]:
        """Evaluate the global model on the test dataset.
        
        Parameters:
            parameters (NDArrays): Model parameters to evaluate
            
        Returns:
            Dict[str, float]: Dictionary with evaluation metrics
        """
        # Create model and load parameters
        model = self.model_generator()
        model = set_model_parameters(model, parameters)
        model.to(self.device)
        model.eval()
        
        # Evaluate on the test dataset
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.global_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        avg_loss = total_loss / total if total > 0 else float('inf')
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
    
    def _aggregate_weighted_average(self, metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        """Aggregate metrics from multiple clients using weighted averaging.
        
        Parameters:
            metrics (List[Tuple[int, Dict[str, float]]]): List of (num_samples, metrics_dict) tuples
            
        Returns:
            Dict[str, float]: Aggregated metrics
        """
        if not metrics:
            return {}
        
        # Extract all possible metric names
        metric_names = set()
        for _, client_metrics in metrics:
            metric_names.update(client_metrics.keys())
        
        # Calculate weighted average for each metric
        result = {}
        total_samples = sum(num_samples for num_samples, _ in metrics)
        
        for metric_name in metric_names:
            weighted_sum = sum(
                client_metrics.get(metric_name, 0.0) * num_samples
                for num_samples, client_metrics in metrics
                if metric_name in client_metrics
            )
            result[metric_name] = weighted_sum / total_samples if total_samples > 0 else 0.0
        
        return result
    
    def run(self):
        """Run the federated learning experiment."""
        logger.info(f"Running federated experiment '{self.experiment_name}'...")
        start_time = time.time()
        
        # Create client generator
        federated_client_generator = get_flower_client_generator(
            model_generator=self.model_generator,
            partition_dir=self.federated_partition_dir,
            data_dir=Path(self.config.data_dir)
        )
        
        # Create evaluation function for central evaluation
        def federated_evaluation_function(
            server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
            """Evaluate the global model on a central test set."""
            metrics = self._evaluate_global_model(parameters)
            return metrics["loss"], {"accuracy": metrics["accuracy"]}
        
        # Set up training and test configs
        federated_train_config = self._create_federated_train_config()
        test_config = self._create_test_config()
        
        # Create functions to return configs
        def fit_config_fn(server_round: int) -> Dict[str, Scalar]:
            """Return training configuration for clients."""
            return federated_train_config
        
        def eval_config_fn(server_round: int) -> Dict[str, Scalar]:
            """Return evaluation configuration for clients."""
            return test_config
        
        # Get initial parameters from the model
        initial_model = self.model_generator()
        initial_parameters = get_model_parameters(initial_model)
        initial_parameters_flwr = ndarrays_to_parameters(initial_parameters)
        
        # Calculate fractions for client selection
        fraction_fit = self.config.num_clients_per_round / max(self.config.num_total_clients, 1)
        fraction_evaluate = self.config.num_evaluation_clients / max(self.config.num_total_clients, 1)
        
        # Create FedAvg strategy
        strategy = FedAvg(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=self.config.num_clients_per_round,
            min_evaluate_clients=self.config.num_evaluation_clients,
            min_available_clients=self.config.num_total_clients,
            on_fit_config_fn=fit_config_fn,
            on_evaluate_config_fn=eval_config_fn,
            evaluate_fn=federated_evaluation_function,
            initial_parameters=initial_parameters_flwr,
            accept_failures=False,
            server_learning_rate=self.config.server_learning_rate,
            server_momentum=self.config.server_momentum
        )
        
        # Create server config with custom aggregation frequency
        server_config = ServerConfig(num_rounds=self.config.num_rounds)
        
        # Define helper function for starting simulation with fixed seed
        def start_seeded_simulation(
            client_fn: Callable[[str], Client],
            num_clients: int,
            config: ServerConfig,
            strategy,
            seed: int,
            **kwargs: Any
        ) -> Tuple[List[Tuple[int, NDArrays]], History]:
            """Start simulation with fixed seed for reproducibility."""
            # Set seed before simulation
            np.random.seed(seed)
            torch.manual_seed(seed)
            random.seed(seed)
            
            # Start simulation
            parameter_list, hist = flwr.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=num_clients,
                client_resources={"num_cpus": 1},
                config=config,
                strategy=strategy,
                **kwargs
            )
            return parameter_list, hist
        
        # Convert client generator to return Client instances (not NumPyClient)
        def simulator_client_generator(cid: str) -> Client:
            numpy_client = federated_client_generator(cid)
            return numpy_client.to_client()
        
        # Run the simulation
        logger.info(f"Starting federated simulation with {self.config.num_rounds} rounds")
        self.parameters_for_each_round, self.history = start_seeded_simulation(
            client_fn=simulator_client_generator,
            num_clients=len(self.client_ids),
            config=server_config,
            strategy=strategy,
            seed=self.config.seed,
        )
        
        # Extract and store results
        metrics_distributed = self.history.metrics_distributed
        metrics_centralized = self.history.metrics_centralized
        
        # Process results for each round
        for round_idx in range(self.config.num_rounds):
            round_results = {
                "round": round_idx + 1,
                "distributed": {},
                "centralized": {}
            }
            
            # Extract distributed metrics if available
            if metrics_distributed and "fit" in metrics_distributed:
                fit_metrics = metrics_distributed["fit"]
                if len(fit_metrics) > round_idx:
                    round_results["distributed"]["fit"] = fit_metrics[round_idx][1]
            
            if metrics_distributed and "evaluate" in metrics_distributed:
                eval_metrics = metrics_distributed["evaluate"]
                if len(eval_metrics) > round_idx:
                    round_results["distributed"]["evaluate"] = eval_metrics[round_idx][1]
            
            # Extract centralized metrics if available
            if metrics_centralized and "evaluate" in metrics_centralized:
                central_metrics = metrics_centralized["evaluate"]
                if len(central_metrics) > round_idx:
                    round_results["centralized"]["evaluate"] = central_metrics[round_idx][1]
            
            # Store round results
            self.results["rounds"].append(round_results)
        
        # Get final model parameters
        final_parameters = self.parameters_for_each_round[-1][1] if self.parameters_for_each_round else initial_parameters
        
        # Evaluate final model
        final_metrics = self._evaluate_global_model(final_parameters)
        self.results["final_model"] = {
            "metrics": final_metrics,
            "parameters": None  # We don't store the actual parameters in the JSON results
        }
        
        # Save final model if requested
        if self.config.save_models:
            final_model = self.model_generator()
            final_model = set_model_parameters(final_model, final_parameters)
            torch.save(final_model.state_dict(), self.output_dir / "final_model.pt")
        
        # Record total runtime
        self.results["runtime"] = time.time() - start_time
        
        # Save results
        if self.config.save_results:
            with open(self.output_dir / "results.json", "w") as f:
                # Convert numpy values to standard Python types for JSON serialization
                serializable_results = self._make_json_serializable(self.results)
                json.dump(serializable_results, f, indent=4)
            
            # Save the Flower history object
            save_history(self.output_dir, self.history, "flower_history")
        
        logger.info(f"Experiment '{self.experiment_name}' completed in {self.results['runtime']:.2f} seconds.")
        logger.info(f"Final model accuracy: {final_metrics['accuracy']:.4f}")
        
        return self.results
    
    def _make_json_serializable(self, obj):
        """Convert all numpy values to standard Python types for JSON serialization."""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        else:
            return obj

    def visualize_results(self):
        """Visualize the results of the federated learning experiment."""
        if not self.results or not self.results.get("rounds"):
            logger.error("No results to visualize. Run the experiment first.")
            return
        
        # Set up the figure
        plt.figure(figsize=(12, 8))
        
        # Extract round numbers and accuracies
        rounds = [r["round"] for r in self.results["rounds"]]
        accuracies = []
        
        for r in self.results["rounds"]:
            if "centralized" in r and "evaluate" in r["centralized"] and "accuracy" in r["centralized"]["evaluate"]:
                accuracies.append(r["centralized"]["evaluate"]["accuracy"])
            else:
                # If no centralized evaluation, try to get distributed evaluation
                acc = 0.0
                if "distributed" in r and "evaluate" in r["distributed"]:
                    metrics = r["distributed"]["evaluate"]
                    if "local_accuracy" in metrics:
                        acc = metrics["local_accuracy"]
                accuracies.append(acc)
        
        # Plot accuracy over rounds
        plt.plot(rounds, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Round', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.title(f'Federated Learning Experiment: {self.experiment_name}', fontsize=16)
        
        # Add horizontal line for final accuracy
        final_acc = self.results["final_model"]["metrics"]["accuracy"]
        plt.axhline(y=final_acc, color='r', linestyle='--', alpha=0.7)
        plt.text(0.5, final_acc + 0.01, f'Final Accuracy: {final_acc:.4f}', 
                 horizontalalignment='center', color='r')
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.output_dir / "accuracy_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
        
        # If we have train loss data, plot it
        plt.figure(figsize=(12, 8))
        train_losses = []
        
        for r in self.results["rounds"]:
            if "distributed" in r and "fit" in r["distributed"] and "train_loss" in r["distributed"]["fit"]:
                train_losses.append(r["distributed"]["fit"]["train_loss"])
            else:
                train_losses.append(None)  # Handle missing data
        
        # Only plot if we have loss data
        if any(loss is not None for loss in train_losses):
            # Filter out None values for plotting
            valid_rounds = [r for r, loss in zip(rounds, train_losses) if loss is not None]
            valid_losses = [loss for loss in train_losses if loss is not None]
            
            plt.plot(valid_rounds, valid_losses, marker='o', linestyle='-', linewidth=2, markersize=8, color='orange')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlabel('Round', fontsize=14)
            plt.ylabel('Training Loss', fontsize=14)
            plt.title(f'Training Loss: {self.experiment_name}', fontsize=16)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(self.output_dir / "train_loss_plot.png", dpi=300, bbox_inches="tight")
        
        plt.close()
        
        # Print summary statistics
        logger.info("=== Experiment Summary ===")
        logger.info(f"Runtime: {self.results['runtime']:.2f} seconds")
        logger.info(f"Final model accuracy: {final_acc:.4f}")
        logger.info(f"Number of rounds: {self.config.num_rounds}")
        logger.info(f"Local epochs per round: {self.config.local_epochs}")
        logger.info(f"Clients per round: {self.config.num_clients_per_round}/{self.config.num_total_clients}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Server learning rate: {self.config.server_learning_rate}")
        
        return {"final_accuracy": final_acc}


def run_parameter_sweep(base_config: FederatedExperimentConfig, parameter_grid: Dict[str, List[Any]]):
    """Run experiments with all combinations of parameters in the grid.
    
    Parameters:
        base_config (FederatedExperimentConfig): Base configuration for experiments
        parameter_grid (Dict[str, List[Any]]): Dictionary mapping parameter names to lists of values
        
    Returns:
        List[Dict]: List of experiment results
    """
    # Generate all combinations of parameters
    from itertools import product
    
    # Get keys and values from parameter grid
    keys = parameter_grid.keys()
    values = parameter_grid.values()
    
    results = []
    
    # Iterate through all combinations
    for combination in product(*values):
        # Create a copy of the base config
        config_dict = asdict(base_config)
        
        # Update with the current parameter combination
        for key, value in zip(keys, combination):
            config_dict[key] = value
        
        # Create a unique experiment name based on varied parameters
        param_str = "_".join(f"{key}={value}" for key, value in zip(keys, combination))
        config_dict["experiment_name"] = f"{base_config.experiment_name}_{param_str}"
        
        # Create config and runner
        config = FederatedExperimentConfig(**config_dict)
        runner = FederatedExperimentRunner(config)
        
        try:
            # Setup and run experiment
            runner.setup()
            result = runner.run()
            runner.visualize_results()
            results.append(result)
        except Exception as e:
            logger.error(f"Error running experiment {config_dict['experiment_name']}: {e}")
            continue
    
    return results


def run_learning_rate_experiment(base_config, learning_rates, output_dir="./results/federated_experiments"):
    """Run experiments with different learning rates for client training.
    
    Parameters:
        base_config (FederatedExperimentConfig): Base configuration for experiments
        learning_rates (List[float]): List of learning rate values to try
        output_dir (str): Directory to save results
        
    Returns:
        pd.DataFrame: DataFrame with experiment results
    """
    parameter_grid = {"learning_rate": learning_rates}
    results = run_parameter_sweep(base_config, parameter_grid)
    
    # Convert results to DataFrame for analysis
    df_data = []
    for result in results:
        config = result["config"]
        final_accuracy = result["final_model"]["metrics"]["accuracy"]
        df_data.append({
            "learning_rate": config["learning_rate"],
            "final_accuracy": final_accuracy,
            "runtime": result["runtime"]
        })
    
    df = pd.DataFrame(df_data)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(df["learning_rate"], df["final_accuracy"], marker='o', linestyle='-')
    plt.xscale('log')  # Learning rates are often compared on a log scale
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Client Learning Rate', fontsize=14)
    plt.ylabel('Final Accuracy', fontsize=14)
    plt.title('Effect of Client Learning Rate on Model Accuracy', fontsize=16)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save plot and data
    plt.tight_layout()
    plt.savefig(output_path / "learning_rate_comparison.png", dpi=300, bbox_inches="tight")
    df.to_csv(output_path / "learning_rate_comparison.csv", index=False)
    
    return df


def run_aggregation_frequency_experiment(base_config, frequencies, output_dir="./results/federated_experiments"):
    """Run experiments with different aggregation frequencies.
    
    Parameters:
        base_config (FederatedExperimentConfig): Base configuration for experiments
        frequencies (List[int]): List of aggregation frequency values to try
        output_dir (str): Directory to save results
        
    Returns:
        pd.DataFrame: DataFrame with experiment results
    """
    parameter_grid = {"aggregation_frequency": frequencies}
    results = run_parameter_sweep(base_config, parameter_grid)
    
    # Convert results to DataFrame for analysis
    df_data = []
    for result in results:
        config = result["config"]
        final_accuracy = result["final_model"]["metrics"]["accuracy"]
        df_data.append({
            "aggregation_frequency": config["aggregation_frequency"],
            "final_accuracy": final_accuracy,
            "runtime": result["runtime"]
        })
    
    df = pd.DataFrame(df_data)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(df["aggregation_frequency"], df["final_accuracy"], marker='o', linestyle='-')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Aggregation Frequency (rounds)', fontsize=14)
    plt.ylabel('Final Accuracy', fontsize=14)
    plt.title('Effect of Aggregation Frequency on Model Accuracy', fontsize=16)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save plot and data
    plt.tight_layout()
    plt.savefig(output_path / "aggregation_frequency_comparison.png", dpi=300, bbox_inches="tight")
    df.to_csv(output_path / "aggregation_frequency_comparison.csv", index=False)
    
    return df


def run_server_lr_experiment(base_config, server_learning_rates, output_dir="./results/federated_experiments"):
    """Run experiments with different server learning rates.
    
    Parameters:
        base_config (FederatedExperimentConfig): Base configuration for experiments
        server_learning_rates (List[float]): List of server learning rate values to try
        output_dir (str): Directory to save results
        
    Returns:
        pd.DataFrame: DataFrame with experiment results
    """
    parameter_grid = {"server_learning_rate": server_learning_rates}
    results = run_parameter_sweep(base_config, parameter_grid)
    
    # Convert results to DataFrame for analysis
    df_data = []
    for result in results:
        config = result["config"]
        final_accuracy = result["final_model"]["metrics"]["accuracy"]
        df_data.append({
            "server_learning_rate": config["server_learning_rate"],
            "final_accuracy": final_accuracy,
            "runtime": result["runtime"]
        })
    
    df = pd.DataFrame(df_data)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(df["server_learning_rate"], df["final_accuracy"], marker='o', linestyle='-')
    plt.xscale('log')  # Learning rates are often compared on a log scale
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Server Learning Rate', fontsize=14)
    plt.ylabel('Final Accuracy', fontsize=14)
    plt.title('Effect of Server Learning Rate on Model Accuracy', fontsize=16)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save plot and data
    plt.tight_layout()
    plt.savefig(output_path / "server_lr_comparison.png", dpi=300, bbox_inches="tight")
    df.to_csv(output_path / "server_lr_comparison.csv", index=False)
    
    return df


def run_local_epochs_experiment(base_config, epochs_values, output_dir="./results/federated_experiments"):
    """Run experiments with different numbers of local epochs.
    
    Parameters:
        base_config (FederatedExperimentConfig): Base configuration for experiments
        epochs_values (List[int]): List of local epoch values to try
        output_dir (str): Directory to save results
        
    Returns:
        pd.DataFrame: DataFrame with experiment results
    """
    parameter_grid = {"local_epochs": epochs_values}
    results = run_parameter_sweep(base_config, parameter_grid)
    
    # Convert results to DataFrame for analysis
    df_data = []
    for result in results:
        config = result["config"]
        final_accuracy = result["final_model"]["metrics"]["accuracy"]
        df_data.append({
            "local_epochs": config["local_epochs"],
            "final_accuracy": final_accuracy,
            "runtime": result["runtime"]
        })
    
    df = pd.DataFrame(df_data)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(df["local_epochs"], df["final_accuracy"], marker='o', linestyle='-')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Local Epochs', fontsize=14)
    plt.ylabel('Final Accuracy', fontsize=14)
    plt.title('Effect of Local Epochs on Model Accuracy', fontsize=16)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save plot and data
    plt.tight_layout()
    plt.savefig(output_path / "local_epochs_comparison.png", dpi=300, bbox_inches="tight")
    df.to_csv(output_path / "local_epochs_comparison.csv", index=False)
    
    return df


def run_multi_seed_experiment(base_config, seeds, output_dir="./results/federated_experiments"):
    """Run experiments with multiple random seeds to assess stability.
    
    Parameters:
        base_config (FederatedExperimentConfig): Base configuration for experiments
        seeds (List[int]): List of random seed values to try
        output_dir (str): Directory to save results
        
    Returns:
        pd.DataFrame: DataFrame with experiment results
    """
    parameter_grid = {"seed": seeds}
    results = run_parameter_sweep(base_config, parameter_grid)
    
    # Convert results to DataFrame for analysis
    df_data = []
    for result in results:
        config = result["config"]
        final_accuracy = result["final_model"]["metrics"]["accuracy"]
        df_data.append({
            "seed": config["seed"],
            "final_accuracy": final_accuracy,
            "runtime": result["runtime"]
        })
    
    df = pd.DataFrame(df_data)
    
    # Calculate statistics
    mean_accuracy = df["final_accuracy"].mean()
    std_accuracy = df["final_accuracy"].std()
    
    # Plot results as a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df["seed"].astype(str), df["final_accuracy"])
    plt.axhline(y=mean_accuracy, color='r', linestyle='--', alpha=0.7, label=f'Mean: {mean_accuracy:.4f} ± {std_accuracy:.4f}')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.xlabel('Random Seed', fontsize=14)
    plt.ylabel('Final Accuracy', fontsize=14)
    plt.title('Model Accuracy Across Different Random Seeds', fontsize=16)
    plt.legend()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Save plot and data
    plt.tight_layout()
    plt.savefig(output_path / "multi_seed_comparison.png", dpi=300, bbox_inches="tight")
    df.to_csv(output_path / "multi_seed_comparison.csv", index=False)
    
    # Print summary
    logger.info(f"Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    
    return df


# Example usage
if __name__ == "__main__":
    base_config = FederatedExperimentConfig(
        experiment_name="federated_experiment",
        dataset_name="femnist",
        model_name="cnn",
        num_rounds=10,
        local_epochs=1,
        learning_rate=0.01,
        server_learning_rate=1.0,
        data_distribution="natural"
    )
    
    # Run the experiment
    runner = FederatedExperimentRunner(base_config)
    runner.setup()
    results = runner.run()
    runner.visualize_results()
    
    # Example parameter sweep
    # run_learning_rate_experiment(base_config, [0.001, 0.01, 0.1]) 