from typing import List, Optional
import torch.nn as nn

class Adapters:
    def __init__(self, layer_type: List[str]):
        """
        Initialize an Adapters object with a list of supported layer types.

        Args:
            layer_type (List[str]): List of supported layer types.
        """
        self.layer_type = layer_type

    @staticmethod
    def layer_type_check(layer: str) -> bool:
        """
        Check if a given layer type is supported.

        Args:
            layer (str): The layer type to check.

        Returns:
            bool: True if the layer type is supported, False otherwise.
        """
        layers = ["nn.Linear", "nn.Embedding", "nn.Conv1d", "nn.Conv2d"]
        return layer in layers

    def __call__(self, fn):
        """
        Decorator to apply an adapter function to specified layers.

        Args:
            fn (Callable): The adapter function to be applied.

        Returns:
            Callable: Decorated function with adapter applied.
        """
        def decorated_fn():
            if all(self.layer_type_check(layer) for layer in self.layer_type):
                print(f"Layers to be adjusted using AdapterLoRa: {self.layer_type}")
                print("Adapter Applied:", fn.__name__)
                fn()
            else:
                print("Some layer types are not supported.")
        return decorated_fn


class Optimizer:
    def __init__(self, optimizer: nn.Module):
        pass  # You can add initialization logic here
