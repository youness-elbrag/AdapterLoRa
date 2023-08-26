import loralib as LoRa 
import loratorch as LoRaT
import torch.nn as nn
from typing import Optional
import bitsandbytes as nn 



def Layer(model, new_layer):
    new_layer.weight = nn.Parameter(model.weight.detach().clone())

    if model.bias is not None:
        new_layer.bias = nn.Parameter(model.bias.detach().clone())

    return new_layer

def LoRaLinear(method:str, model:nn.Module, Rank:Optional[int],threshold:Optional[int]):
    Adapters = ["LoRa","SandBytes","LoRaTorch"]
    if Adapters.__contains__(Adapters) == True:
        if method == "LoRa":
            new_layer = LoRa.Linear(
                                in_features=model.in_features,
                                out_features=model.out_features,
                                bias=model.bias is not None,
                                r=Rank
            )
            return Layer(model . new_layer)

        if method == "SandBytes":
            new_layer = bnb.nn.Linear8bitLt(
                model.in_features,
                model.out_featuresm2, 
                bias=model.bias is not None, 
                has_fp16_weights=False, 
                threshold=6.0
                )
            return Layer(model . new_layer)


        if method == "LoRaTorch": 
            new_layer = LoRaT.Linear(
                                in_features=model.in_features,
                                out_features=model.out_features,
                                bias=model.bias is not None,
                                r=Rank
                                        )
            return Layer(model . new_layer)

    else:
        raise ValueError(f"there's no method support yet or may you inster invalide name method {method}")


def LoRaEmbedding(method:str,
     model:nn.Module , 
     Rank:Optional[int], 
     lora_alpha:Optional[int],
     scale_grad_by_freq:Optional[int],
     padding_idx:Optional[int],
     max_norm:Optional[int]):

    Adapters = ["LoRa","SandBytes","LoRaTorch"]
    if Adapters.__contains__(Adapters) == True:
        if method == "LoRa":
            new_layer = LoRa.Embedding(model.num_embeddings, 
                        model.embedding_dim, 
                        r=Rank,
                        lora_alpha=lora_alpha,
                        max_norm=model.max_norm is not None,
                        scale_grad_by_freq=model.scale_grad_by_freq is not None,
                        padding_idx=model.padding_idx is not None
                )
            return new_layer

        if method == "SandBytes":
            new_layer=  bnb.nn.StableEmbedding(model.num_embeddings, 
                        model.embedding_dim ) 
            return new_layer

        if method == "LoRaTorch":
            new_layer = LoRaT.Embedding(model.num_embeddings, 
                        model.embedding_dim, 
                        r=Rank,
                        max_norm=model.max_norm is not None,
                        scale_grad_by_freq=model.scale_grad_by_freq is not None,
                        padding_idx=model.padding_idx is not None
                )
            return new_layer
    else:
        raise ValueError(f"there's no method support yet or may you inster invalide name method {method}")

