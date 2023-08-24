import torch.nn as nn

def make_lora_replace(model, quantized_fn, Rank, layers, depth=1, path="", verbose=True):
    """
    Replace specified linear layers in the model with quantized layers using LoRA.

    Args:
        model (nn.Module): The input model to be modified.
        quantized_fn (Callable): The function to quantize a linear layer.
        Rank (int): The rank parameter for LoRA adaptation.
        layers (list): List of layer names to be adapted.
        depth (int): Current depth in recursion (default is 1).
        path (str): Current path in model hierarchy (default is empty string).
        verbose (bool): Flag to print verbose messages (default is True).

    Returns:
        nn.Module: The modified model with specified layers quantized using LoRA.
    """
    if depth > 10:
        return model

    if isinstance(model, nn.Linear) and any(item in path for item in layers):
        if verbose:
            print(f"Found linear layer to quantize: {path}", type(model))
        return quantized_fn(model, Rank)

    for key, module in model.named_children():
        if isinstance(module, nn.Linear) and any(item in path for item in layers):
            layer = quantized_fn(module, Rank)
            setattr(model, key, layer)
            if verbose:
                print(f"Found linear layer to quantize: {path}:{key}", type(module))
        elif isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            for i, elem in enumerate(module):
                layer = make_lora_replace(
                    elem, quantized_fn, Rank, layers, depth + 1, f"{path}:{key}[{i}]", verbose=verbose
                )
                if layer is not None:
                    module[i] = layer
        else:
            layer = make_lora_replace(
                module, quantized_fn, Rank, layers, depth + 1, f"{path}:{key}", verbose=verbose
            )
            if layer is not None:
                setattr(model, key, layer)

    return model
