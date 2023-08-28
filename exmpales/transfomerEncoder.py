import torch.nn as nn
import torch
import os 
import sys

current_dir = os.path.dirname(__file__)
target_dir = os.path.abspath(os.path.join(current_dir, ".././"))
sys.path.insert(0, target_dir)

from core.Quantized import AdapterLoRa

model = nn.TransformerEncoderLayer(d_model=512, nhead=8)


Adpate_model = AdapterLoRa(model , method="LoRa", Rank=4)
Adpate_model.add_layer_and_Instance_Layer("nn.Linear","self_attn")
Adpate_model.add_layer_and_Instance_Layer("nn.Linear","linear1")
Adpate_model.add_add_layer_and_Instance_Layerlayer("nn.Linear","linear2")
Adpate_model.reconstruct_model()
model = Adpate_model.implement_lora(verbose=True)



