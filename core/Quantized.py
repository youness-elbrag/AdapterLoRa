import loratorch as LoraT 
import torch.nn as nn 
import numpy as np 
from utils import make_lora_replace



class Quantized(ABC):

    def __init__(self , method=None , Rank):
        super(Quantized, self).__init__()
        self.method = method
        self.Rank = Rank
        self.LORA = True

    @staticmethod
    def lora_layer(layer, method, Rank):
        new_layer = method.Linear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,  # Fixing the bias check
            r=Rank
        )
            # Cloning the tensor
        new_layer.weight = nn.Parameter(layer.weight.detach().clone())  
        
        if layer.bias is not None:
            new_layer.bias = nn.Parameter(layer.bias.detach().clone())  
        
        return new_layer

    def ReconstractedModel(self, model):
        IF isinstance(model , nn.Module) is not True:
            return f"Please make sure the model based on Torch nn.Module"
        
        if self.LORA == True:
            make_lora_replace(model, self.method , self.Rank)

        return NotImplemented



   

