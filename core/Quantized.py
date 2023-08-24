import loratorch as LoraT 
import torch.nn as nn 
import numpy as np 
from utils import make_lora_replace


class CastOutputToFloat(nn.Sequential):        
    def forward(self, model): 
        return super().forward(x).to(torch.float32)
        
class AdapterLoRa:

    def __init__(self , model , method=None , Rank):
        super(AdapterLoRa, self).__init__()
        self.method = method
        self.Rank = Rank
        self.LORA = True
        self.model = model 

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

    def FreezeWieght(self,model, weight_frezz=False):

        for param in model.parameters():
            param.requires_grad = weight_frezz  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        # reduce number of stored activations

        model.gradient_checkpointing_enable()  
    
        model.encoder , model.decoder= CastOutputToFloat(model.encoder) ,  CastOutputToFloat(model.decoder)

    def ReconstractedModel(self, model):
        
        if isinstance(model , nn.Module) is not True:
            return f"Please make sure the model based on Torch nn.Module"
        
        if self.LORA == True:
            make_lora_replace(self.model, self.method , self.Rank)

    def ImplemwntLoRa(self):

        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters before LoRA: {total_trainable_params}")

        ## Apply LoRA
        self.method.mark_only_lora_as_trainable(self.model)

        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters after LoRA: {total_trainable_params}")

        return self.model

     



   

