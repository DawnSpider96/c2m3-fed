import torch.nn as nn
from c2m3.models.cnn import CNN
from c2m3.models.mlp import MLP
# Import other model classes as needed


class LayerNorm2d(nn.Module):
    """
    Mimics JAX's LayerNorm. This is used in place of BatchNorm to mimic Git Re-basin's setting as close as possible.
    """

    def __init__(self, num_features):
        super(LayerNorm2d, self).__init__()
        self.layer_norm = nn.LayerNorm((num_features,))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)

        x = self.layer_norm(x)

        x = x.permute(0, 3, 1, 2)
        return x


class BatchNorm2d(nn.Module):
    """
    Just a quirky wrapper around BatchNorm to have the same PermutationSpec as LayerNorm2d. (Should be fixed in the future)
    """

    def __init__(self, num_features):
        super().__init__()
        self.layer_norm = nn.BatchNorm2d(num_features)

    def forward(self, x):
        return self.layer_norm(x)


def get_model_class(model_name):
    """
    Factory function to get model class based on model name.
    
    Args:
        model_name (str): Name of the model to get ('cnn', 'mlp', etc.)
        
    Returns:
        function: A constructor function for the requested model.
    """
    model_registry = {
        "cnn": CNN,
        "mlp": MLP,
        # Add more models here as they are implemented
    }
    
    model_name = model_name.lower()
    if model_name not in model_registry:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(model_registry.keys())}")
    
    return model_registry[model_name]
