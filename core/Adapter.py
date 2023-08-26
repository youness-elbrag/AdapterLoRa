import bitsandbytes as bnb
import loratorch as LoraT
import loralib as lora
import torch.nn as nn 
from typing import dict , Optional , Union




class Adapters(object):
    
    def __init__(self, layerTyep:list, Method:func)-> nn.Module:
        self.layer = layerTyep

    @staticmethod
    def LayerType(self , layer):
        layers = ["nn.Linear" , "nn.Embedding", "nn.Conv1d","nn.Conv2d"]
        AdaptedLayer = []
        for i in layer:
            for j in layers:
                if layer[i] == layers[j]:
                    AdaptedLayer.append(layer[i])
            return f"{layers[i]} not support Please Visit \n Docs to list correct Layer support"
        return AdaptedLayer

    def __call__(self, fn):
        if self.LayerType(self.layer):
            def __fn():
                print(f"Layers to adjusted Used AdapterLoRa: {[layer for layer in self.layer]}")
                print("Adapter Applied:", fn.__name__)        
                fn()
            return __fn



class Optimzer:
    def __init__(self, Optimzer: nn.Module):
        pass