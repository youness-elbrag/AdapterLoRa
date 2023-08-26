import torch.nn as nn
from typing import Optional
from .Adapter import Adapters

def quantize_layer(**kwargs):
    """
    Apply the appropriate quantization function to the given layer.

    Args:
        **kwargs: Dictionary containing the quantization parameters.

    Returns:
        nn.Module: The quantized layer.
    """
    method = kwargs["method"]
    model = kwargs["model"]
    Rank = kwargs.get("Rank")
    lora_alpha = kwargs.get("lora_alpha")
    scale_grad_by_freq = kwargs.get("scale_grad_by_freq")
    padding_idx = kwargs.get("padding_idx")
    max_norm = kwargs.get("max_norm")
    threshold = kwargs.get("threshold")

    if isinstance(model, nn.Linear):
        quantize_fn = kwargs["quantize_fn"]
        if quantize_fn is not None:
            return quantize_fn(
                method=method,
                model=model,
                Rank=Rank,
                **kwargs
            )

    elif isinstance(model, nn.Embedding):
        quantize_fn_ = kwargs["quantize_fn_"]
        if quantize_fn_ is not None:
            return quantize_fn_(
                method=method,
                model=model,
                Rank=Rank,
                lora_alpha=lora_alpha,
                scale_grad_by_freq=scale_grad_by_freq,
                padding_idx=padding_idx,
                max_norm=max_norm,
                **kwargs
            )

    return model


def make_lora_replace(
    model, method: str, LayerType, quantize_fn=None, quantize_fn_=None, Rank=0, layers=None,
    depth=1, path="", verbose=True
):
    """
    Replace specified linear and embedding layers in the model with quantized layers using LoRA.

    Args:
        model (nn.Module): The input model to be modified.
        LayerType (str): Type of layers to quantize. "nn.Linear" for linear layers, "nn.Embedding" for embedding layers.
        quantize_fn (Callable, optional): The function to quantize a linear layer.
        quantize_fn_ (Callable, optional): The function to quantize an embedding layer.
        Rank (int, optional): The rank parameter for LoRA adaptation.
        layers (list, optional): List of layer names to be adapted.
        depth (int, optional): Current depth in recursion (default is 1).
        path (str, optional): Current path in model hierarchy (default is empty string).
        verbose (bool, optional): Flag to print verbose messages (default is True).

    Returns:
        nn.Module: The modified model with specified layers quantized using LoRA.
    """
    AdaptersLayer = ["nn.Linear", "nn.Embedding"]

    if depth > 10:
        return model

    if LayerType[0] in AdaptersLayer and isinstance(model, nn.Linear) and any(item in path for item in layers):
        if verbose:
            print(f"Found linear layer to quantize: {path}", type(model))
        if quantize_fn is not None:
            return quantize_fn(
                method=method,
                model=model,
                Rank=Rank
            )

    if LayerType[1] in AdaptersLayer and isinstance(model, nn.Embedding) and any(item in path for item in layers):
        if verbose:
            print(f"Found embedding layer to quantize: {path}", type(model))
        if quantize_fn_ is not None:
            return quantize_fn_(
                method=method,
                model=model,
                Rank=Rank,
                lora_alpha=lora_alpha,
                scale_grad_by_freq=scale_grad_by_freq,
                padding_idx=padding_idx,
                max_norm=max_norm
            )

    for key, module in model.named_children():
        if isinstance(module, (nn.Linear, nn.Embedding)) and any(item in path for item in layers):
            quantized_layer = quantize_layer(module, quantize_fn, quantize_fn_, Rank)
            setattr(model, key, quantized_layer)
            if verbose:
                print(f"Found linear or embedding layer to quantize: {path}:{key}", type(module))
        elif isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            for i, elem in enumerate(module):
                layer = make_lora_replace(
                    elem, method, LayerType, quantize_fn, quantize_fn_, Rank, layers,
                    depth + 1, f"{path}:{key}[{i}]", verbose=verbose
                )
                if layer is not None:
                    module[i] = layer
        else:
            layer = make_lora_replace(
                module, method, LayerType, quantize_fn, quantize_fn_, Rank, layers,
                depth + 1, f"{path}:{key}", verbose=verbose
            )
            if layer is not None:
                setattr(model, key, layer)

    return model
