from logging import DEBUG, WARNING
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.simulation.ray_transport.fn_client_proxy import FnClientProxy
from flwr.common import GetPropertiesIns

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy

from c2m3.common.client_utils import (
    Net, 
    get_network_generator_cnn, 
    get_model_parameters,
    set_model_parameters
)
from c2m3.modules.pl_module import MyLightningModule
from c2m3.match.frank_wolfe_sync_merging import frank_wolfe_synchronized_merging

import torch
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
from collections import OrderedDict

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

NUM_CLASSES_FEMNIST = 62

# Aggregation function using preloaded DataLoaders and datasets
def frank_wolfe_dataset_aggregate(weights_results, train_datasets, batch_size=64):
    """Aggregate client models using the Frank-Wolfe algorithm with direct dataset objects.
    
    Parameters
    ----------
        weights_results (List[Tuple[NDArrays, int]]): List of (parameters, num_examples) tuples
        train_datasets (List[Dataset]): List of training datasets corresponding to each client
        batch_size (int): Batch size for creating the DataLoaders
        
    Returns
    -------
        NDArrays: Aggregated model parameters
    """
    lightning_modules = []
    network_generator = get_network_generator_cnn()
    
    # Create DataLoaders from the datasets
    train_loaders = [
        DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        ) 
        for dataset in train_datasets
    ]

    for params, num_examples in weights_results:
        # We first put the NDArray in a CNN pytorch module
        net: Net = network_generator()
        set_model_parameters(net, params)
        # Then put the CNN module in pytorch MyLightningModule
        mlm = MyLightningModule(net, num_classes=NUM_CLASSES_FEMNIST)
        lightning_modules.append(mlm)
    
    merged_lightning_module = frank_wolfe_synchronized_merging(lightning_modules, train_loaders)

    return get_model_parameters(merged_lightning_module.model)


class DatasetFrankWolfeSync(Strategy):
    """C2M3 n-model merging, frank-wolfe implementation with dataset objects directly.

    This strategy is designed to work with the federated_experiment.py when using direct dataset objects
    rather than file paths. The key difference is that this strategy doesn't need to load datasets from
    files as they are provided directly by the client.

    Parameters
    ----------
    train_datasets : List[Dataset]
        List of training datasets for each client.
    fraction_fit : float, optional
        Fraction of clients used during training. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    batch_size : int, optional
        Batch size for data loading. Defaults to 64.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        train_datasets: List[Dataset],
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        batch_size: int = 64,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.train_datasets = train_datasets
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.batch_size = batch_size
        
        # Client ID to dataset mapping
        self.client_dataset_mapping = {}

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"DatasetFrankWolfeSync(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Store which clients were sampled for this round
        for client in clients:
            # Extract client ID from ClientProxy
            if isinstance(client, FnClientProxy):
                cid = client.cid
                # Map client ID directly to its corresponding dataset index without modulo
                # This assumes each client has its own unique dataset partition
                self.client_dataset_mapping[cid] = int(cid)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if failures:
            log(DEBUG, "Failures: %s", failures)
            if isinstance(failures[0], BaseException):
                raise failures[0]
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        # Convert results
        weights_results = []
        client_datasets = []
        
        for client_proxy, fit_res in results:
            weights_results.append(
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            )
            # Get the client ID to find its corresponding dataset
            if isinstance(client_proxy, FnClientProxy):
                cid = client_proxy.cid
                # Use the mapping we created during configure_fit
                dataset_idx = self.client_dataset_mapping.get(cid, int(cid))
                # Ensure index is within bounds
                if dataset_idx >= len(self.train_datasets):
                    log(WARNING, f"Client ID {cid} maps to dataset index {dataset_idx} which is out of bounds. Using the first dataset instead.")
                    dataset_idx = 0
                client_datasets.append(self.train_datasets[dataset_idx])
        
        # Aggregate parameters using our frank_wolfe algorithm
        parameters_aggregated = ndarrays_to_parameters(
            frank_wolfe_dataset_aggregate(
                weights_results, 
                client_datasets,
                batch_size=self.batch_size
            )
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated 