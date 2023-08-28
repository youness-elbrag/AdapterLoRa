import torch.nn as nn
import bitsandbytes as bnb
import loralib as LoRa
import loratorch as LoRaT
from typing import Optional
from .Quantized import AdapterLoRa

LAYERS = AdapterLoRa.layertyep
print(LAYERS)

def Layer(model, new_layer):
    """
    Copy weights and biases from the original layer to the new layer.

    Args:
        model (nn.Module): The original layer.
        new_layer (nn.Module): The new layer.

    Returns:
        nn.Module: The new layer with copied weights and biases.
    """
    new_layer.weight = nn.Parameter(model.weight.detach().clone())

    if model.bias is not None:
        new_layer.bias = nn.Parameter(model.bias.detach().clone())

    return new_layer

@Adapters(LAYERS)
def LoRaLinear(method: str, model: nn.Module, Rank: Optional[int], threshold: Optional[int]):
    """
    Replace a linear layer with a quantized layer using specified method.

    Args:
        method (str): The quantization method ("LoRa", "SandBytes", "LoRaTorch").
        model (nn.Module): The input model containing the linear layer.
        Rank (Optional[int]): The rank parameter for LoRA adaptation.
        threshold (Optional[int]): The threshold parameter for SandBytes adaptation.

    Returns:
        nn.Module: The modified model with the quantized layer.
    """
    Adapters = ["LoRa", "SandBytes", "LoRaTorch"]

    if method in Adapters:
        if method == "LoRa":
            new_layer = LoRa.Linear(
                in_features=model.in_features,
                out_features=model.out_features,
                bias=model.bias is not None,
                r=Rank
            )
            return Layer(model, new_layer)

        if method == "SandBytes":
            new_layer = bnb.nn.Linear8bitLt(
                model.in_features,
                model.out_features,
                bias=model.bias is not None,
                has_fp16_weights=False,
                threshold=threshold
            )
            return Layer(model, new_layer)

        if method == "LoRaTorch":
            new_layer = LoRaT.Linear(
                in_features=model.in_features,
                out_features=model.out_features,
                bias=model.bias is not None,
                r=Rank
            )
            return Layer(model, new_layer)

    else:
        raise ValueError(f"Unsupported method or invalid method name: {method}")

@Adapters(LAYERS)
def LoRaEmbedding(
    method: str,
    model: nn.Module,
    Rank: Optional[int],
    lora_alpha: Optional[int],
    scale_grad_by_freq: Optional[int],
    padding_idx: Optional[int],
    max_norm: Optional[int]
):
    """
    Replace an embedding layer with a quantized layer using specified method.

    Args:
        method (str): The quantization method ("LoRa", "SandBytes", "LoRaTorch").
        model (nn.Module): The input model containing the embedding layer.
        Rank (Optional[int]): The rank parameter for LoRA adaptation.
        lora_alpha (Optional[int]): The alpha parameter for LoRA adaptation.
        scale_grad_by_freq (Optional[int]): The scale_grad_by_freq parameter for LoRA adaptation.
        padding_idx (Optional[int]): The padding_idx parameter for LoRA adaptation.
        max_norm (Optional[int]): The max_norm parameter for LoRA adaptation.

    Returns:
        nn.Module: The modified model with the quantized layer.
    """
    Adapters = ["LoRa", "SandBytes", "LoRaTorch"]

    if method in Adapters:
        if method == "LoRa":
            new_layer = LoRa.Embedding(
                model.num_embeddings,
                model.embedding_dim,
                r=Rank,
                lora_alpha=lora_alpha,
                max_norm=model.max_norm is not None,
                scale_grad_by_freq=model.scale_grad_by_freq is not None,
                padding_idx=model.padding_idx is not None
            )
            return new_layer

        if method == "SandBytes":
            new_layer = bnb.nn.StableEmbedding(
                model.num_embeddings,
                model.embedding_dim
            )
            return new_layer

        if method == "LoRaTorch":
            new_layer = LoRaT.Embedding(
                model.num_embeddings,
                model.embedding_dim,
                r=Rank,
                max_norm=model.max_norm is not None,
                scale_grad_by_freq=model.scale_grad_by_freq is not None,
                padding_idx=model.padding_idx is not None
            )
            return new_layer

    else:
        raise ValueError(f"Unsupported method or invalid method name: {method}")
