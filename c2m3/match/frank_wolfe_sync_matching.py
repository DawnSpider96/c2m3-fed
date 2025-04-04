import copy
import logging
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from scipy.optimize import fminbound
from tqdm import tqdm

from c2m3.match.frank_wolfe_matching import collect_perm_sizes, initialize_perm_matrices, project_gradients
from c2m3.match.permutation_spec import PermutationSpec
from c2m3.match.utils import PermutationIndices, PermutationMatrix, generalized_inner_product, perm_cols, perm_rows
from c2m3.match.weight_matching import solve_linear_assignment_problem
from c2m3.utils.utils import get_model

# from flwr.common.typing import NDArray

pylogger = logging.getLogger(__name__)


def frank_wolfe_synchronized_matching(
    params: Dict[str, Dict], # Symbol to State dict of model
    perm_spec: PermutationSpec,
    symbols: List[str],
    combinations: List[Tuple],
    max_iter: int,
    initialization_method: str,
    keep_soft_perms: bool = False,
    score_tolerance: float = 1e-6,
    device=None,
    verbose=False,
):
    """
    Frank-Wolfe based matching with factored permutations P_AB = P_A P_B^T, with P_B^T mapping from B to the universe
    and P_A mapping from the universe to A.
    """
    if device is None:
        # Check if CUDA is available and set device accordingly
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        pylogger.setLevel(logging.INFO)
    else:
        pylogger.setLevel(logging.WARNING)

    ref_params = params[symbols[0]]

    perm_sizes = collect_perm_sizes(perm_spec, ref_params)

    perm_names = list(perm_spec.perm_to_layers_and_axes.keys())

    perm_matrices = {
        symbol: initialize_perm_matrices(perm_sizes, initialization_method, device=device) for symbol in symbols
    }

    projected_gradients = {
        symb: {perm: torch.zeros((perm_sizes[perm], perm_sizes[perm])) for perm in perm_names} for symb in symbols
    }

    old_obj = 0.0
    patience_steps = 0
    obj_values, step_sizes, perm_history = [], [], []

    for iteration in tqdm(range(max_iter), desc="Weight matching"):
        pylogger.info(f"Iteration {iteration}")

        gradients = {
            symb: {perm: torch.zeros((perm_sizes[perm], perm_sizes[perm]), device=device) for perm in perm_names}
            for symb in symbols
        }

        for fixed_symbol, permutee_symbol in combinations:
            params_a, params_b = params[fixed_symbol], params[permutee_symbol]

            pylogger.debug(f"Collecting gradients for {fixed_symbol} and {permutee_symbol}")

            collect_gradients_frank_wolfe_model_pair(
                params_a,
                params_b,
                fixed_symbol,
                permutee_symbol,
                perm_spec.layer_and_axes_to_perm,
                perm_matrices,
                gradients,
                device,
            )

        # Line 11 pseudocode (LAP)
        # Check again why symb in symbols corresponds to P in Permutation Matrices
        for symb in symbols:
            projected_gradients[symb] = project_gradients(gradients[symb], device)

        line_search_step_func = partial(
            line_search_step_sync,
            proj_grads=projected_gradients,
            perm_matrices=perm_matrices,
            params=params,
            combinations=combinations,
            layers_and_axes_to_perms=perm_spec.layer_and_axes_to_perm,
            device=device,
        )

        # Line 14 pseudocode
        # line search to determine step_size (alpha)
        # Arguments are ALL permutation matrices, and projected gradients
        step_size, func_val, _, num_called_func = fminbound(
            line_search_step_func, 0, 1, xtol=1e-3, maxfun=20, full_output=1
        )

        pylogger.info(f"Step size: {step_size}, function value: {func_val}, num called functions: {num_called_func}")

        perm_matrices = update_perm_matrices_sync(symbols, perm_matrices, step_size, projected_gradients)

        new_obj = get_all_pairs_global_obj_sync(
            params, combinations, perm_matrices, perm_spec.layer_and_axes_to_perm, device
        )

        pylogger.info(f"Objective: {np.round(new_obj, 6)}")

        obj_values.append(new_obj)
        step_sizes.append(step_size)
        # perm_history.append(perm_matrices)

        if (new_obj - old_obj) < score_tolerance:
            patience_steps += 1
        else:
            patience_steps = 0
            old_obj = new_obj

        if patience_steps >= 5:
            break

    perm_indices = {
        symb: {
            p: perm if keep_soft_perms else solve_linear_assignment_problem(perm)
            for p, perm in perm_matrices[symb].items()
        }
        for symb in symbols
    }

    opt_infos = {"obj_values": obj_values, "step_sizes": step_sizes, "perm_history": perm_history}

    return perm_indices, opt_infos


# Changes gradients dictionary
# Gradients is of same type as perm_matrices
# Dict of layer name (eg "conv1.bias") to tuple of strings, each matching 
#   a matrix in perm_matrices[symbol], keyed by the same string

# params_a/b is dictionary of (layer name, weight matrix)
# symbol_a/b is simply string or character
def collect_gradients_frank_wolfe_model_pair(
    params_a, params_b, symbol_a, symbol_b, layers_and_axes_to_perms, perm_matrices, gradients, device="cpu"
):

    # collect the gradients for a single pair of models, e.g. a, b. These will be 4 different gradients
    # layer is a String eg "conv1.bias", "conv1.weight", "fc1.weight"
    # axes_and_perms is type Tuple. Content was filled in Permutation_Spec class
    for layer, axes_and_perms in layers_and_axes_to_perms.items():
        assert layer in params_a.keys() and layer in params_b.keys()

        # Every layer has a Weight Matrix
        # Wa, Wb are torch tensors. 
        # If 1D tensor, unsqueeze it into a 2D matrix (all elements become 1D tensors)
        Wa, Wb = params_a[layer], params_b[layer]
        if Wa.dim() == 1:
            Wa = Wa.unsqueeze(1)
            Wb = Wb.unsqueeze(1)
        if Wa.dim() == 0:
            continue

        # any permutation acting on the first axis is permuting rows
        # axes_and_perms is a tuple of length at least 2.
        # One for rows (dimension 0) and one for columns (dimension 1)
        # TYPE OF row_perm_id and col_perm_id STRING. Indicates which permutation matrix to use.
        row_perm_id = axes_and_perms[0]
        assert row_perm_id is None or row_perm_id in perm_matrices[symbol_a].keys()

        # perm_matrices[symbol_a] is model A's DICT of permutation (current holder) matrices.
        # Maps id of permutation matrix 
        # torch.eye returns identity matrix (of size specified by argument)
        # ie perm_matrices[symbol_a] is a list of matrices
        Pa_curr = perm_matrices[symbol_a][row_perm_id] if row_perm_id is not None else torch.eye(Wa.shape[0])
        Pb_curr = perm_matrices[symbol_b][row_perm_id] if row_perm_id is not None else torch.eye(Wa.shape[0])

        # any permutation acting on the second axis is permuting columns
        col_perm_id = axes_and_perms[1] if len(axes_and_perms) > 1 else None
        assert col_perm_id is None or col_perm_id in perm_matrices[symbol_a].keys()

        Pa_prev = perm_matrices[symbol_a][col_perm_id] if col_perm_id is not None else torch.eye(Wb.shape[1])
        Pb_prev = perm_matrices[symbol_b][col_perm_id] if col_perm_id is not None else torch.eye(Wb.shape[1])

        Pa_curr, Pb_curr = Pa_curr.to(device), Pb_curr.to(device)
        Pa_prev, Pb_prev = Pa_prev.to(device), Pb_prev.to(device)

        if row_perm_id:
            gradients[symbol_a][row_perm_id] += compute_grad_P_curr_sync(Wa, Wb, Pa_prev, Pb_prev, Pb_curr)
            gradients[symbol_b][row_perm_id] += compute_grad_P_curr_sync(Wb, Wa, Pb_prev, Pa_prev, Pa_curr)

        if col_perm_id:
            gradients[symbol_a][col_perm_id] += compute_grad_P_prev_sync(Wa, Wb, Pa_curr, Pb_curr, Pb_prev)
            gradients[symbol_b][col_perm_id] += compute_grad_P_prev_sync(Wb, Wa, Pb_curr, Pa_curr, Pa_prev)


def compute_grad_P_curr_sync(Wa, Wb, Pa_prev, Pb_prev, Pb_curr, debug=True):
    """
    Wa Pa_prev Pb_prev^T Wb^T Pb_curr
    """

    # Wb^T Pb_curr
    Wb_col_perm = perm_cols(x=Wb.transpose(1, 0), perm=Pb_curr)

    # Pb_prev^T (Wb^T Pb_curr)
    Wb_row_col_perm = perm_rows(x=Wb_col_perm, perm=Pb_prev.T)

    # Pa_prev (Pb_prev^T (Wb^T Pb_curr))
    Pa_prev_Pb_prevT_WbT_Pb_curr = perm_rows(x=Wb_row_col_perm, perm=Pa_prev)

    gradient = generalized_inner_product(Wa, Pa_prev_Pb_prevT_WbT_Pb_curr)

    if debug and len(Wa.shape) == 2:
        assert torch.allclose(gradient, Wa @ Pa_prev @ Pb_prev.T @ Wb.T @ Pb_curr, atol=1e-2)

    return gradient


def compute_grad_P_prev_sync(Wa, Wb, Pa_curr, Pb_curr, Pb_prev, debug=True):
    """
    Wa^T Pa_curr Pb_curr^T Wb Pb_prev
    """

    # Wb Pb_prev
    Wb_col_perm = perm_cols(x=Wb, perm=Pb_prev)

    # Pb_curr^T (Wb Pb_prev)
    Wb_row_col_perm = perm_rows(x=Wb_col_perm, perm=Pb_curr.T)

    # Pa_curr (Pb_curr^T (Wb Pb_prev))
    Pa_curr_Pb_currT_Wb_Pb_prev = perm_rows(x=Wb_row_col_perm, perm=Pa_curr)

    gradient = generalized_inner_product(Wa.transpose(1, 0), Pa_curr_Pb_currT_Wb_Pb_prev)

    if debug and len(Wa.shape) == 2:

        assert torch.allclose(gradient, Wa.T @ Pa_curr @ Pb_curr.T @ Wb @ Pb_prev, atol=1e-1)

    return gradient


def line_search_step_sync(
    t: float,
    params,
    combinations,
    proj_grads: Dict[str, Dict[str, torch.Tensor]],
    perm_matrices: Dict[str, PermutationIndices],
    layers_and_axes_to_perms,
    device="cpu",
):

    tot_obj = 0.0

    for symb_a, symb_b in combinations:
        params_a, params_b = params[symb_a], params[symb_b]
        interpolated_perms = {symb_a: {}, symb_b: {}}

        for symb in [symb_a, symb_b]:
            for perm_name, perm in perm_matrices[symb].items():

                proj_grad = proj_grads[symb][perm_name]

                interpolated_perms[symb][perm_name] = (1 - t) * perm + t * proj_grad

        tot_obj += get_global_obj_layerwise_sync(
            params_a, params_b, symb_a, symb_b, interpolated_perms, layers_and_axes_to_perms, device
        )

    return -tot_obj


def get_all_pairs_global_obj_sync(params, combinations, perm_matrices, layers_and_axes_to_perms, device):
    tot_obj = 0.0

    for symb_a, symb_b in combinations:
        params_a, params_b = params[symb_a], params[symb_b]
        tot_obj += get_global_obj_layerwise_sync(
            params_a, params_b, symb_a, symb_b, perm_matrices, layers_and_axes_to_perms, device
        )

    return tot_obj


def get_global_obj_layerwise_sync(
    params_a, params_b, symbol_a, symbol_b, perm_matrices, layers_and_axes_to_perms, device
):
    tot_obj = 0.0

    for layer, axes_and_perms in layers_and_axes_to_perms.items():
        assert layer in params_a.keys()
        assert layer in params_b.keys()

        Wa, Wb = params_a[layer], params_b[layer]

        if Wa.dim() == 1:
            Wa = Wa.unsqueeze(1)
            Wb = Wb.unsqueeze(1)
        if Wa.dim() == 0:
            continue

        # any permutation acting on the first axis is permuting rows
        row_perm_id = axes_and_perms[0]
        assert row_perm_id is None or row_perm_id in perm_matrices[symbol_a].keys()
        Pa_curr = (
            perm_matrices[symbol_a][row_perm_id] if row_perm_id is not None else torch.eye(Wa.shape[0], device=device)
        )
        Pb_curr = (
            perm_matrices[symbol_b][row_perm_id] if row_perm_id is not None else torch.eye(Wa.shape[0], device=device)
        )

        # any permutation acting on the second axis is permuting columns
        col_perm_id = axes_and_perms[1] if len(axes_and_perms) > 1 else None
        assert col_perm_id is None or col_perm_id in perm_matrices[symbol_a].keys()
        Pa_prev = (
            perm_matrices[symbol_a][col_perm_id] if col_perm_id is not None else torch.eye(Wa.shape[1], device=device)
        )
        Pb_prev = (
            perm_matrices[symbol_b][col_perm_id] if col_perm_id is not None else torch.eye(Wa.shape[1], device=device)
        )

        layer_similarity = compute_layer_similarity_sync(Wa, Wb, Pa_curr, Pb_curr, Pa_prev, Pb_prev)

        tot_obj += layer_similarity.cpu().numpy()

    return tot_obj


def compute_layer_similarity_sync(Wa, Wb, Pa_curr, Pb_curr, Pa_prev, Pb_prev):
    """
    tr(Wa.T Pa_curr Pb_curr^T Wb (Pa_prev Pb_prev^T)^T )
    """

    # (Pa_prev Pb_prev^T)^T

    Pa_prev_Pb_prevT = (Pa_prev @ Pb_prev.T).T

    # Wb (Pa_prev Pb_prev^T)^T
    Wb_Pa_prev_Pb_prevT = perm_cols(x=Wb, perm=Pa_prev_Pb_prevT)

    # Pb_curr^T (Wb (Pa_prev Pb_prev^T)^T)
    Wb_Pa_prev_Pb_prevT_Pb_curr = perm_rows(x=Wb_Pa_prev_Pb_prevT, perm=Pb_curr.T)

    # Pa_curr (Pb_curr^T (Wb (Pa_prev Pb_prev^T)^T))
    Wb_Pa_prev_Pb_prevT_Pb_curr_Pa_curr = perm_rows(x=Wb_Pa_prev_Pb_prevT_Pb_curr, perm=Pa_curr)

    inner_product = generalized_inner_product(Wa.transpose(1, 0), Wb_Pa_prev_Pb_prevT_Pb_curr_Pa_curr)
    layer_similarity = torch.trace(inner_product)

    return layer_similarity


def update_perm_matrices_sync(
    symbols: List[str],
    perm_matrices: Dict[str, Dict[str, PermutationMatrix]],
    step_size: float,
    proj_grads: Dict[str, Dict[str, torch.Tensor]],
):
    new_perm_matrices = {symb: {} for symb in symbols}

    for symb in symbols:

        for perm_name, perm in perm_matrices[symb].items():
            new_perm_matrices[symb][perm_name] = (1 - step_size) * perm + step_size * proj_grads[symb][perm_name].cpu()
            # new_perm_matrices[symb][perm_name] = (1 - step_size) * perm + step_size * proj_grads[symb][perm_name].cuda()

    return new_perm_matrices


def exact_gen_dot_product(x, y):

    result = torch.zeros((x.shape[0], y.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            dot = x[i, j].flatten() @ y[i, j].flatten()
            result[i, j] = dot

    return result
