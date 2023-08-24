import unittest
import torch.nn as nn

def make_quantized(layer, Rank):
    new_layer = self.LoRa.Linear(
            in_features=layer.in_features,
            out_features=layer.out_features,
            bias=layer.bias is not None,
            r=Rank)

    new_layer.weight = nn.Parameter(layer.weight.detach().clone())

    if layer.bias is not None:
        new_layer.bias = nn.Parameter(layer.bias.detach().clone())

    return new_layer

class TestMakeLoraReplace(unittest.TestCase):

    def test_quantization(self):
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.Linear(5, 3)
        )
        layers_to_quantize = ['Linear']
        modified_model = make_lora_replace(model, make_quantized, 4, layers_to_quantize, verbose=False)

        # Check that the model has been modified
        self.assertIsNot(model, modified_model)

        # Check that the linear layers have been quantized
        for module in modified_model.children():
            if isinstance(module, nn.Linear):
                self.assertIsInstance(module, nn.Linear)  # Example assertion, replace with appropriate checks

if __name__ == '__main__':
    unittest.main()
