from logging import DEBUG, WARNING
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

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

from c2m3.models.tiny_resnet import TinyResNet, BasicBlock
from c2m3.modules.pl_module import MyLightningModule
from c2m3.match.permutation_spec import TinyResNetPermutationSpecBuilder
from c2m3.match.frank_wolfe_sync_merging import frank_wolfe_synchronized_merging
from c2m3.common.client_utils import get_model_parameters, set_model_parameters, CIFAR10FromSamples

from torch.utils.data import DataLoader, Dataset
import torch
import json
from PIL import Image
from torchvision import transforms

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""

NUM_CLASSES_CIFAR10 = 10

# weights_results: List(Tuple(NDArray parameters, number of examples))
# return NDArray
def frank_wolfe_aggregate_tiny_resnet(weights_results, train_loaders):
    """
    Aggregate CIFAR10 TinyResNet models using Frank-Wolfe synchronized merging.
    
    Args:
        weights_results: List of (parameters, num_examples) tuples
        train_loaders: List of training DataLoaders for each client
        
    Returns:
        NDArrays: Merged model parameters
    """
    lightning_modules = []
    
    for params, num_examples in weights_results:
        # Create a TinyResNet model
        net = TinyResNet(block=BasicBlock, num_classes=NUM_CLASSES_CIFAR10)
        
        # Set parameters from the client
        set_model_parameters(net, params)
        
        # Wrap in a Lightning module
        mlm = MyLightningModule(net, num_classes=NUM_CLASSES_CIFAR10)
        lightning_modules.append(mlm)
    
    # Create permutation spec for TinyResNet
    perm_spec_builder = TinyResNetPermutationSpecBuilder()
    perm_spec = perm_spec_builder.create_permutation_spec()
    
    # Perform synchronized merging with the permutation specification
    merged_lightning_module = frank_wolfe_synchronized_merging(
        lightning_modules, 
        train_loaders,
        permutation_spec=perm_spec
    )

    return get_model_parameters(merged_lightning_module.model)


class FrankWolfeSyncTinyResNet(Strategy):
    """C2M3 n-model merging for TinyResNet CIFAR10 models using Frank-Wolfe implementation.

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
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
    """

    def __init__(
        self,
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
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

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

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FrankWolfeSyncTinyResNet(accept_failures={self.accept_failures})"
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
        """Aggregate fit results using Frank-Wolfe synchronized merging."""
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
        train_loaders: List[DataLoader] = []
        
        # Get configuration for dataloader
        config = self.on_fit_config_fn(0) if self.on_fit_config_fn else {"batch_size": 32, "num_workers": 0}
        ins: GetPropertiesIns = GetPropertiesIns(config={})
        
        for client_proxy, fit_res in results:
            weights_results.append(
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            )
            
            # Get client properties
            props = client_proxy.get_properties(ins=ins, timeout=None).properties
            
            # Create dataloader for client data
            batch_size = int(config.get("batch_size", 32))
            num_workers = int(config.get("num_workers", 0))
            
            full_file: Path = Path(props["partition"]) / str(props["cid"])
            
            try:
                with open(full_file, "r") as f:
                    mapping = json.load(f)
                
                # Create a dataset from the mapping
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                
                # Create a dataset from the mapping
                dataset = CIFAR10FromSamples(
                    samples=mapping["train"],
                    data_dir=Path(props.get("data_dir", "./")),
                    transform=transform
                )
                
                train_loader = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    drop_last=True,
                )
                train_loaders.append(train_loader)
                
            except Exception as e:
                log(WARNING, f"Could not create dataloader for client {props['cid']}: {e}")
                # Create a dummy dataloader to maintain indexing
                dummy_dataset = torch.utils.data.TensorDataset(
                    torch.randn(1, 3, 32, 32), torch.zeros(1, dtype=torch.long)
                )
                train_loaders.append(DataLoader(dummy_dataset, batch_size=1))
        
        # Perform Frank-Wolfe aggregation with TinyResNet permutation specs
        parameters_aggregated = ndarrays_to_parameters(
            frank_wolfe_aggregate_tiny_resnet(weights_results, train_loaders)
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