from flwr.server.strategy import Strategy
import numpy as np
from typing import List, Tuple, Optional
from flwr.common import Parameters, Scalar, NDArray
from flwr.server.client_proxy import ClientProxy
from flwr.simulation.ray_transport.fn_client_proxy import FnClientProxy


def initialize_multiple_parameters(initial_parameters_list, client_manager):
    """
    Assign different initial parameters to different clients
    """
    clients: List[FnClientProxy] = client_manager.all()
    # print(f'{clients=}')
    
    if len(clients) > len(initial_parameters_list):
        # If more clients than parameter sets, cycle through the list
        param_sets = (initial_parameters_list * 
                        ((len(clients) // len(initial_parameters_list)) + 1))
    else:
        param_sets = initial_parameters_list

    # Map each client to a unique initial parameter set
    client_params = {}
    for i, client in enumerate(clients):
        # print(client)
        # param_sets[i] should already be a list of ndarrays representing one complete model parameter set
        client_params[client] = Parameters(
            tensors=param_sets[i],  # This is already the list of tensors for this client
            tensor_type="numpy.ndarray"
        )
    
    return client_params