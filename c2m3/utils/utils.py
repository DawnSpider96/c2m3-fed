import copy
import json
import logging
import time
from functools import wraps
from pathlib import Path
from pydoc import locate
from typing import Any, Dict, List, Tuple, Union

# import hydra
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
# import wandb
from omegaconf import ListConfig
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from scipy.interpolate import interp1d
from scipy.misc import derivative
from torch.utils.data import DataLoader

from nn_core.common import PROJECT_ROOT
from nn_core.serialization import load_model

from c2m3.modules.pl_module import MyLightningModule

ModelParams = Dict[str, torch.Tensor]


pylogger = logging.getLogger(__name__)

MODEL_SEED_TO_SYMBOL = {
    0: "o",
    1: "a",
    2: "b",
    3: "c",
    4: "d",
    5: "e",
    6: "f",
    7: "g",
    8: "h",
    9: "i",
    10: "j",
    11: "k",
    12: "l",
    13: "m",
    14: "n",
    15: "p",
    16: "q",
    17: "r",
    18: "s",
    19: "t",
    20: "u",
    "dummy_a": "x",
    "dummy_b": "y",
    "dummy_c": "z",
}


def project_onto(a, b):
    return torch.dot(a, b) / torch.dot(b, b) * b


def normalize_unit_norm(a):
    return a / torch.norm(a, p=2)


def to_np(tensor):
    if tensor.nelement() == 1:  # Check if the tensor is a scalar
        return tensor.item()  # Convert a scalar tensor to a Python number
    else:
        return tensor.cpu().detach().numpy()  # Convert a tensor to a numpy array


def map_model_seed_to_symbol(seed):
    return MODEL_SEED_TO_SYMBOL[seed]


def flatten_params(model):
    return model.state_dict()


def calculate_global_radius(dist_matrix, k=5, target_percentage=0.8):
    """
    Calculate a global radius such that a target percentage of neurons have at least K neighbors.
    """
    num_neurons = dist_matrix.shape[0]
    max_radius = torch.max(dist_matrix)
    step = max_radius / 100  # Increment step for radius
    radius = 0

    while radius <= max_radius:
        count = 0
        for i in range(num_neurons):
            # Calculate distances
            neuron_distances = dist_matrix[i]

            # Count how many neurons are within the current radius
            neighbors_within_radius = torch.sum(neuron_distances <= radius).item() - 1  # Exclude the neuron itself

            # Check if the current neuron has at least K neighbors
            if neighbors_within_radius >= k:
                count += 1

        # Check if the current radius satisfies the condition for the target percentage of neurons
        if (count / num_neurons) >= target_percentage:
            return radius
        radius += step

    return radius


def plot_cumulative_density(neuron_dists, min_value, num_radii):
    max_dist = torch.max(neuron_dists)
    radii = torch.linspace(min_value, max_dist, num_radii)

    cumulative_counts = [torch.sum(neuron_dists <= radius) for radius in radii]

    # Normalize cumulative counts to get the fraction (percentage) of neighbors
    cumulative_counts_normalized = torch.tensor(cumulative_counts) / len(neuron_dists)

    # Interpolate the cumulative counts to compute derivative
    interpolated_counts = interp1d(radii, cumulative_counts_normalized, kind="cubic")

    print(interpolated_counts)
    # Compute the derivative (approximate local density)
    # Compute the derivative within the bounds of the interpolated function
    # Using a slightly reduced range to avoid boundary issues
    radius_min = radii[1] + 1e-1  # start slightly above the minimum to avoid boundary issue
    radius_max = radii[-2] - 1e-1  # end slightly below the maximum to avoid boundary issue
    fine_radii = torch.linspace(radius_min, radius_max, num_radii)
    density = [derivative(interpolated_counts, r, dx=1e-2) for r in fine_radii]

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot cumulative counts
    plt.subplot(1, 2, 1)
    plt.plot(radii, cumulative_counts_normalized, label="Cumulative Counts")
    plt.xlabel("Radius")
    plt.ylabel("Fraction of Neighbors")
    plt.title("Cumulative Counts vs Radius")
    plt.grid(True)
    plt.legend()

    # Plot derivative (density)
    plt.subplot(1, 2, 2)
    plt.plot(fine_radii, density, label="Density (Derivative)", color="orange")
    plt.xlabel("Radius")
    plt.ylabel("Density")
    plt.title("Density vs Radius")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def linear_interpolation(model_a, model_b, lamb):
    return (1 - lamb) * model_a + lamb * model_b


def linear_interpolate(lambd: float, model_a: Union[MyLightningModule, Dict], model_b: Union[MyLightningModule, Dict]):
    """
    Linearly interpolate models given as LightningModules or as StateDicts.
    """
    pylogger.info(f"Evaluating interpolated model with lambda: {lambd}")

    if isinstance(model_a, torch.Tensor) and isinstance(model_b, torch.Tensor):
        # flat model parameters, interpolate them as vectors
        return (1 - lambd) * model_a + lambd * model_b

    if isinstance(model_a, MyLightningModule) and isinstance(model_b, MyLightningModule):
        model_a = model_a.state_dict()
        model_b = model_b.state_dict()

    interpolated_model = copy.deepcopy(model_a)

    for param_name in model_a:
        interpolated_model[param_name] = (1 - lambd) * model_a[param_name] + (lambd) * model_b[param_name]

    return interpolated_model


def l2_norm_models(state_dict1, state_dict2):
    """Calculate the L2 norm of the difference between two state dictionaries."""
    diff_squared_sum = sum(torch.sum((state_dict1[key] - state_dict2[key]) ** 2) for key in state_dict1)
    return torch.sqrt(diff_squared_sum)


def cosine_models(state_dict1, state_dict2):
    """Calculate the cosine similarity between two state dictionaries."""
    dot_product = sum(torch.dot(state_dict1[key].view(-1), state_dict2[key].view(-1)) for key in state_dict1)
    norm1 = torch.sqrt(sum(torch.sum(state_dict1[key] ** 2) for key in state_dict1))
    norm2 = torch.sqrt(sum(torch.sum(state_dict2[key] ** 2) for key in state_dict2))
    return dot_product / (norm1 * norm2)


def average_models(model_params, reduction="mean"):
    if not isinstance(model_params, List):
        model_params = list(model_params.values())

    if reduction == "mean":
        return {k: torch.mean(torch.stack([p[k].float() for p in model_params]), dim=0) for k in model_params[0].keys()}
    elif reduction == "median":
        return {k: torch.median(torch.stack([p[k] for p in model_params]), dim=0)[0] for k in model_params[0].keys()}
    elif reduction == "normal":
        return {
            k: torch.normal(
                torch.mean(torch.stack([p[k] for p in model_params]), dim=0),
                torch.std(torch.stack([p[k] for p in model_params]), dim=0),
            )
            for k in model_params[0].keys()
        }
    else:
        raise ValueError(f"Invalid reduction {reduction}")


def get_checkpoint_callback(callbacks):
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback


# def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
#     """Instantiate the callbacks given their configuration.

#     Args:
#         cfg: a list of callbacks instantiable configuration
#         *args: a list of extra callbacks already instantiated

#     Returns:
#         the complete list of callbacks to use
#     """
#     callbacks: List[Callback] = list(args)

#     for callback in cfg:
#         pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
#         callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

#     return callbacks


def block(i, j, n):
    return slice(i * n, (i + 1) * n), slice(j * n, (j + 1) * n)


def load_model_from_info(model_info_path, seed=None, zipped=True):
    suffix = "" if seed is None else f"_{seed}.json"
    model_info_path_seed = model_info_path + suffix

    model_info = json.load(open(model_info_path_seed))
    model_class = locate(model_info["class"])

    suffix = ".zip" if zipped else ""
    model = load_model(model_class, checkpoint_path=Path(str(PROJECT_ROOT) + "/" + model_info["path"] + suffix))
    model.eval()

    return model


def load_model_from_artifact(run, artifact_path):
    pylogger.info(f"Trying to load {artifact_path}")

    artifact = run.use_artifact(artifact_path)
    artifact.download()

    model = load_model(MyLightningModule, checkpoint_path=Path(artifact.file()))
    model.eval()

    return model


def save_permutations(permutations: Dict[str, Dict], path: str):
    """ """
    for source, targets in permutations.items():
        if targets is None:
            continue
        for target, source_target_perms in targets.items():
            if source_target_perms is None:
                continue
            for perm_name, perm in source_target_perms.items():
                if perm is None:
                    continue
                permutations[source][target][perm_name] = perm.tolist()

    with open(path, "w+") as f:
        json.dump(permutations, f)


def save_factored_permutations(permutations: Dict[str, Dict], path: str):

    for symbol, perms in permutations.items():
        for perm_name, perm in perms.items():
            if perm is None:
                continue
            permutations[symbol][perm_name] = perm.tolist()

    with open(path, "w+") as f:
        json.dump(permutations, f)


class OnSaveCheckpointCallback(Callback):
    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: Dict[str, Any]
    ) -> None:
        metadata = getattr(pl_module, "metadata", None)
        if metadata is not None:
            checkpoint["metadata"] = metadata


def unravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> Tuple[torch.LongTensor, ...]:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (N,).
        shape: The targeted shape, (D,).

    Returns:
        A tuple of unraveled coordinate tensors of shape (D,).
    """

    coord = unravel_indices(indices, shape)
    return tuple(coord)


def to_relative_path(path: Path):
    if not isinstance(path, Path):
        path = Path(path)
    return str(path.relative_to(PROJECT_ROOT))


def vector_to_state_dict(vec, model):
    """
    Convert a flattened parameter vector into a state_dict for the model.
    """
    state_dict = model.state_dict()

    pointer = 0
    for name, param in state_dict.items():
        num_param = param.numel()  # Number of elements in the parameter

        # Replace the original parameter with the corresponding part of the vector
        state_dict[name].copy_(vec[pointer : pointer + num_param].view_as(param))

        pointer += num_param

    return state_dict


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    return colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )


def fuse_batch_norm_into_conv(conv, bn):
    """
    Fuses a batchnorm layer into a convolutional layer by updating the weights and bias of the convolutional layer.
    Code from https://github.com/KellerJordan/REPAIR
    """

    fused = torch.nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=True,
    )

    # setting weights
    w_conv = conv.weight.clone()
    bn_std = (bn.eps + bn.running_var).sqrt()
    gamma = bn.weight / bn_std
    fused.weight.data = w_conv * gamma.reshape(-1, 1, 1, 1)

    # setting bias
    b_conv = conv.bias if conv.bias is not None else torch.zeros_like(bn.bias)
    beta = bn.bias + gamma * (-bn.running_mean + b_conv)
    fused.bias.data = beta

    return fused


class ConvertToRGB:
    def __call__(self, image):
        convert_to_rgb(image)


def convert_to_rgb(image):
    if image.mode != "RGB":
        return image.convert("RGB")

    # return np.array(image)
    return image


def get_interpolated_loss_acc_curves(
    model_a: MyLightningModule,
    model_b: MyLightningModule,
    lambdas: np.array,
    ref_model: MyLightningModule,
    loader: DataLoader,
    trainer: pl.Trainer,
):

    interp_losses = []
    interp_accs = []

    for lambd in lambdas:
        interp_results = evaluate_interpolated_model(
            model_a=model_a, model_b=model_b, lambd=lambd, ref_model=ref_model, loader=loader, trainer=trainer
        )

        interp_losses.append(interp_results["loss/test"])
        interp_accs.append(interp_results["acc/test"])

    return interp_losses, interp_accs


def evaluate_interpolated_model(
    model_a: MyLightningModule,
    model_b: MyLightningModule,
    lambd: float,
    ref_model: MyLightningModule,
    loader: DataLoader,
    trainer: pl.Trainer,
):

    interp_params = linear_interpolate(
        model_a=model_a.model.state_dict(), model_b=model_b.model.state_dict(), lambd=lambd
    )

    ref_model.model.load_state_dict(interp_params)

    test_results = trainer.test(ref_model, loader, verbose=False)[0]

    return test_results


def cumulative_sum(arr):
    cum_sum = []
    current_sum = 0

    for i in range(len(arr)):
        current_sum += arr[i]

        cum_sum.append(current_sum)

    return cum_sum


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):

        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")

        # wandb.log({"merging_time": total_time})
        return result

    return timeit_wrapper


def get_model(model):
    while hasattr(model, "model"):
        model = model.model

    return model


def set_seed(seed: int) -> None:
    """
    Set the random seed for all relevant random number generators to ensure reproducibility.
    
    Args:
        seed: The random seed to use
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    pylogger.info(f"Random seed set to {seed}")
