import loratorch as LoraT
import torch.nn as nn
import loralib as lora
from utils import make_lora_replace

class CastOutputToFloat(nn.Module):
    def forward(self, x):
        return x.to(torch.float32)

class AdapterLoRa(nn.Module):
    def __init__(self, model: nn.Module, method: str, Rank: int):
        """
        AdapterLoRa constructor.

        Args:
            model (nn.Module): The input model to which LoRA adaptation will be applied.
            method (str): The method to use for LoRA adaptation ("LoRa" or "LoRaTorch").
            Rank (int): The rank parameter for LoRA adaptation.
        """
        super(AdapterLoRa, self).__init__()

        self.methods = {"LoRa": lora, "LoRaTorch": LoraT}
        self.Rank = Rank
        self.LORA = True
        self.model = model
        self.layer = []

        if method in self.methods:
            self.LoRa = self.methods[method]
        else:
            raise ValueError("Invalid method provided")

    def add_layer(self, layer: str):
        """
        Add a layer to the list of layers to be adapted.

        Args:
            layer (str): The name of the layer to add.

        Returns:
            list: The updated list of layers.
        """
        self.layer.append(layer)
        return self.layer

    def lora_layer(self, layer, Rank):
        """
        Create a LoRA adapted layer.

        Args:
            layer (nn.Module): The layer to adapt.
            Rank (int): The rank parameter for LoRA adaptation.

        Returns:
            nn.Module: The adapted layer.
        """
        new_layer = self.LoRa.Linear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            r=Rank
        )

        new_layer.weight = nn.Parameter(layer.weight.detach().clone())

        if layer.bias is not None:
            new_layer.bias = nn.Parameter(layer.bias.detach().clone())

        return new_layer

    def freeze_weights(self, weight_freeze=False):
        """
        Freeze model weights.

        Args:
            weight_freeze (bool): Flag to freeze model weights.

        Returns:
            None
        """
        for param in self.model.parameters():
            param.requires_grad = weight_freeze
            if param.ndim == 1:
                param.data = param.data.to(torch.float32)

        self.model.gradient_checkpointing_enable()
        self.model.encoder, self.model.decoder = CastOutputToFloat(), CastOutputToFloat()

    def reconstruct_model(self):
        """
        Reconstruct the model using LoRA-adapted layers.

        Returns:
            str: A message indicating the success of the reconstruction or an error message.
        """
        if not isinstance(self.model, nn.Module):
            return "Please make sure the model is based on Torch nn.Module"

        if self.LORA:
            make_lora_replace(self.model, self.lora_layer, self.Rank, self.layer)
            return "Model successfully reconstructed with LoRA-adapted layers"

    def implement_lora(self):
        """
        Implement LoRA adaptation on the model.

        Returns:
            nn.Module: The model with LoRA adaptation applied.
        """
        total_trainable_params_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters before LoRA: {total_trainable_params_before}")

        self.LoRa.mark_only_lora_as_trainable(self.model)

        total_trainable_params_after = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total trainable parameters after LoRA: {total_trainable_params_after}")

        return self.model
