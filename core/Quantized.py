import torch.nn as nn
from .LayersAdaptes import *
from .utils import make_lora_replace
import loralib as lora
import loratorch as loraT

class CastOutputToFloat(nn.Module):
    def forward(self, x):
        return x.to(torch.float32)

class AdapterLoRa(nn.Module):

    class LoRaConfig(object):
        def __init__(self):
            self.method = None
            self.Adapters = ["LoRa", "SandBytes", "LoRaTorch"]
            self.Rank = None
            self.model = None
            self.Instance_Layer = []
            self.layertyep = []
            self.QMODEL = None
            self.LORA = None
            self.BITSAND = None

        def SetMethod(self, method):
             if method in self.Adapters:
                self.method = method
             else:
                raise ValueError("Invalid method provided")
        def



    def __init__(self, model: nn.Module, method: str, Rank: int, *args, **kwargs):
        super(AdapterLoRa, self).__init__()
        

        self.extra_args = args
        self.extra_kwargs = kwargs

    def add_layer_and_Instance_Layer(self, layertyep: str, layer: str):
        self.Instance_Layer.append(layer)
        self.layertyep.append(layertyep)
        return self.layertyep, self.Instance_Layer

    def freeze_weights(self, weight_freeze=False):
        for param in self.model.parameters():
            param.requires_grad = weight_freeze
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

        self.model.gradient_checkpointing_enable()
        self.model.encoder, self.model.decoder = CastOutputToFloat(), CastOutputToFloat()
        
    def reconstruct_model(self, verbose=False):
        if not isinstance(self.model, nn.Module):
            return "Please make sure the model is based on Torch nn.Module"

        self.QMODEL = make_lora_replace(
            model=self.model,
            method=self.method,
            LayerType=self.layertyep,
            quantize_fn=LoRaLinear,
            quantize_fn_=LoRaEmbedding,
            Rank=self.Rank,
            layers=self.Instance_Layer,
            *self.extra_args,
            **self.extra_kwargs
        )
        return "Model successfully reconstructed with LoRA-adapted layers"

    def implement_lora(self, verbose=False):
        total_trainable_params_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if verbose:
            print(f"Total trainable parameters before LoRA: {total_trainable_params_before}")

        if self.method == "LoRa":
             self.LORA.mark_only_lora_as_trainable(self.QMODEL)
        elif self.method == "LoRaTorch":
            loraT.mark_only_lora_as_trainable(self.QMODEL)
        elif self.method == "SandBytes":
             self.QMODEL
        
        total_trainable_params_after = sum(p.numel() for p in self.QMODEL.parameters() if p.requires_grad)
        
        if verbose:
            print(f"Total trainable parameters after AdapterLoRA: {total_trainable_params_after}")

        return self.QMODEL
