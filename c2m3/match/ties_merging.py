# """
# Implementation of the TIES (Transplanting Instead of Ensembling in Subspaces) algorithm
# for model merging as described in the paper: https://arxiv.org/pdf/2306.01708
# """

# import logging
# import torch
# import torch.nn as nn
# import numpy as np
# from typing import Dict, List, Optional, Tuple
# from collections import OrderedDict

# logger = logging.getLogger(__name__)

# def compute_weight_matrix_svd(weights: torch.Tensor, threshold: float = 0.9) -> Tuple[torch.Tensor, int]:
#     """
#     Compute the subspace basis using SVD for a weight matrix and determine 
#     how many singular values to keep based on the threshold.
    
#     Args:
#         weights: Weight matrix tensor
#         threshold: Percentage of energy to preserve (default: 0.9)
        
#     Returns:
#         basis: Basis vectors (right singular vectors)
#         k: Number of basis vectors kept
#     """
#     # Reshape if needed (for conv layers)
#     original_shape = weights.shape
#     if len(original_shape) > 2:
#         # For convolutional layers, reshape to [out_channels, in_channels*kernel_size*kernel_size]
#         weights = weights.reshape(original_shape[0], -1)
    
#     # Compute SVD
#     U, S, V = torch.svd(weights)
    
#     # Determine number of singular values to keep based on energy threshold
#     total_energy = torch.sum(S**2)
#     energy_ratio = torch.cumsum(S**2, dim=0) / total_energy
#     k = torch.sum(energy_ratio <= threshold).item() + 1
#     k = min(k, len(S))  # Ensure k is valid
    
#     # Return the basis vectors (right singular vectors)
#     return V[:, :k], k

# def compute_layer_subspaces(state_dicts: List[Dict[str, torch.Tensor]]) -> Dict[str, List[Tuple[torch.Tensor, int]]]:
#     """
#     Compute subspaces for each layer across all models.
    
#     Args:
#         state_dicts: List of model state dictionaries
        
#     Returns:
#         subspaces: Dictionary mapping layer names to lists of (basis, k) tuples
#     """
#     subspaces = {}
    
#     # Get the first model to determine layers
#     first_dict = state_dicts[0]
    
#     for layer_name, param in first_dict.items():
#         # Skip non-weight parameters (like biases and batch norm)
#         if len(param.shape) <= 1 or 'bias' in layer_name or 'bn' in layer_name or 'norm' in layer_name:
#             continue
            
#         subspaces[layer_name] = []
        
#         # Compute subspace for this layer across all models
#         for state_dict in state_dicts:
#             weights = state_dict[layer_name]
#             basis, k = compute_weight_matrix_svd(weights)
#             subspaces[layer_name].append((basis, k))
            
#     return subspaces

# def merge_ties(state_dicts: List[Dict[str, torch.Tensor]], alpha: float = 0.5) -> Dict[str, torch.Tensor]:
#     """
#     Merge multiple models using the TIES algorithm.
    
#     Args:
#         state_dicts: List of model state dictionaries
#         alpha: Interpolation parameter between direct averaging and 
#                subspace-based transplantation (default: 0.5)
               
#     Returns:
#         merged_state_dict: Merged model state dictionary
#     """
#     logger.info(f"Merging {len(state_dicts)} models using TIES with alpha={alpha}")
    
#     # Initialize with a copy of the first state dict
#     merged_state_dict = OrderedDict()
#     first_dict = state_dicts[0]
    
#     # Compute subspaces for weight layers
#     subspaces = compute_layer_subspaces(state_dicts)
    
#     # Process each layer
#     for layer_name, param in first_dict.items():
#         # Direct averaging for non-weight parameters or small layers
#         if layer_name not in subspaces:
#             merged_param = torch.stack([sd[layer_name] for sd in state_dicts]).mean(dim=0)
#             merged_state_dict[layer_name] = merged_param
#             continue
            
#         # For weight layers, apply TIES algorithm
#         original_shape = param.shape
#         flattened = len(original_shape) > 2
        
#         # Get all parameters for this layer
#         layer_params = [sd[layer_name] for sd in state_dicts]
        
#         # Reshape if needed
#         if flattened:
#             layer_params = [p.reshape(original_shape[0], -1) for p in layer_params]
            
#         # Average the weights directly
#         avg_weight = torch.stack(layer_params).mean(dim=0)
        
#         # Get subspaces for this layer
#         layer_subspaces = subspaces[layer_name]
#         basis_list = [basis for basis, _ in layer_subspaces]
        
#         # Apply TIES merging - project onto subspaces and average
#         ties_component = torch.zeros_like(avg_weight)
#         for i, (model_weight, (basis, k)) in enumerate(zip(layer_params, layer_subspaces)):
#             # Project onto subspace
#             projection = model_weight @ basis @ basis.t()
#             ties_component += projection
            
#         ties_component = ties_component / len(layer_params)
        
#         # Interpolate between direct averaging and TIES
#         merged_weight = (1 - alpha) * avg_weight + alpha * ties_component
        
#         # Reshape back if needed
#         if flattened:
#             merged_weight = merged_weight.reshape(original_shape)
            
#         merged_state_dict[layer_name] = merged_weight
    
#     return merged_state_dict

# def merge_models_ties(models: List[nn.Module], alpha: float = 0.5) -> nn.Module:
#     """
#     High-level function to merge a list of PyTorch models using TIES.
    
#     Args:
#         models: List of PyTorch models with identical architecture
#         alpha: Interpolation parameter
        
#     Returns:
#         merged_model: A new model with merged weights
#     """
#     # Extract state dictionaries
#     state_dicts = [model.state_dict() for model in models]
    
#     # Apply TIES merging
#     merged_state_dict = merge_ties(state_dicts, alpha)
    
#     # Create a new model with the same architecture as the first model
#     merged_model = type(models[0])()
#     merged_model.load_state_dict(merged_state_dict)
    
#     return merged_model 