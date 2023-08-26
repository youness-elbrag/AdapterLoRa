# Adapter-LoRa for Quantization  

<div align="center">
  <img src="assets/LoRa.png" alt="LoRa-Logo" width="200">

[![Made With Love](https://img.shields.io/badge/Made%20With-Love-orange.svg)](https://github.com/youness-elbrag/AdapterLoRa/)
[![GitHub issues](https://img.shields.io/github/issues/kyegomez/Med-Palm)](https://github.com/youness-elbrag/AdapterLoRa/issues) 
[![GitHub forks](https://img.shields.io/github/forks/kyegomez/Med-Palm)](https://github.com/youness-elbrag/AdapterLoRa/network) 
[![GitHub stars](https://img.shields.io/github/stars/kyegomez/Med-Palm)](https://github.com/youness-elbrag/AdapterLoRa/stargazers) [![GitHub license](https://img.shields.io/github/license/youness-elbrag/AdapterLoRa)](https://github.com/youness-elbrag/AdapterLoRa/blob/master/LICENSE)
</div>


## Comparative Features of "loralib" and "loratorch" Implementations

**Distinguishing the "loralib" and "loratorch" Approaches for Implementation**

The implementations of "loralib" and "loratorch" exhibit distinct methodologies, particularly when using the example of `nn.Linear`. The underlying mathematical representations are as follows:

1. * **loralib** Approach

  The computation is defined as:

  \[
  h = x W_0^\top + \frac{\alpha}{r} x(BA)^\top,
  \]

  where:
  - `x` is an input matrix of dimensions \(k \times n\),
  - `W_0` is a pre-trained weight matrix of dimensions \(m \times n\),
  - `r` is a predefined LoRA rank,
  - `B` and `A` are LoRA matrices of dimensions \(m \times r\) and \(r \times n\) respectively,
  - `\alpha` is a hyper-parameter.


1. For ``loralib``,
   $h = x W_0^\top + \frac{\alpha}{r} x(BA)^\top,$

where $x\in\mathbb{R}^{k\times n}$ is the input matrix, $W_0\in\mathbb{R}^{m\times n}$ is the pre-trained weight matrix, $r$ is the predefined LoRA rank, $B\in\mathbb{R}^{m\times r}$ and $A\in \mathbb{R}^{r\times n}$ are the LoRA matrixes, and $\alpha$ is a hyper-parameter.

2. For ``loratorch``,
   $h = x (W_0 + \frac{\alpha}{r} BA)^\top.$
   
``loralib`` computes $xW_0^\top$ and $x(BA)^\top$ respectively and then merges the results. While ``loratorch`` merges pre-trained weight $W_0$ and its LoRA weight $BA$ and then computes the results by simply using ``nn.Linear.forward()``. There is no difference between ``loralib`` and ``loratorch`` in the linear layers. But in some no-linear or complex layers, we are no sure whether this layer satisfies $L(x, W_0)+L(x, BA) = L(x, W_0+BA)$. Hence, it is difficult to extend LoRA to some complex layers by using ``loralib``. On the contrary, the idea of merging weights first in ``loratorch`` is more general and extensible. You just call ``merge_lora_param()`` in ``loratorch`` to merge weights and then call ``forward()`` in the original layer to compute the results. With the help of ``loratorch``, you can easily implement LoRA to any type of layer of ``torch.nn``.

## Supported Layers

|                           | ``loralib``    | ``loratorch``  |                                                    |
| ------------------------- |:--------------:|:--------------:| -------------------------------------------------- |
| ``nn.Linear``             | ✓              | ✓              | [linear.ipynb](https://github.com/Baijiong-Lin/LoRA-Torch/blob/main/examples/linear.ipynb)            |
| ``nn.Embedding``          | ✓              | ✓              | [embedding.ipynb](https://github.com/Baijiong-Lin/LoRA-Torch/blob/main/examples/embedding.ipynb)      |
| ``nn.Conv1d``             | ✓              | ✓              |                                                    |
| ``nn.Conv2d``             | ✓              | ✓              |                                                    |
| ``nn.Conv3d``             | ✓              | ✓              |                                                    |
| ``nn.MultiheadAttention`` | ✘              | ✓              |                                                    |
| ``MergedLinear``          | ✓ (Error)      | ✓              | [mergedlinear.ipynb](https://github.com/Baijiong-Lin/LoRA-Torch/blob/main/examples/mergedlinear.ipynb) |
| $\cdots$                  | hard to extend | easy to extend |                                                    |

*We compare the results of ``loralib`` and ``loratorch``  in [examples](./examples) to demonstrate the correctness of the implementation in ``loratorch``.*

## Quick Start

**The usage of ``AdapterLoRa``**

1. Install ``AdapterLoRa``.
   
  ```bash
   pip install git+https://github.com/Baijiong-Lin/LoRA-Torch
  ```

  ```python
  pip install AdapterLoRa
  ```

### Usage Tool AdpaterLoRa

```python

import torch.nn as nn
import torch
from core.Quantized import AdapterLoRa

model = nn.TransformerEncoderLayer(d_model=512, nhead=8)

Adpate_model = AdapterLoRa(model , method="LoRa", Rank=4)

"""
adding Linear Layer built Self.attention 
Replace the layers where you would like to use AdapterLoRa by using  add_layer function.
"""

Adpate_model.add_layer("self_attn") 
Adpate_model.add_layer("linear1")
Adpate_model.add_layer("linear2")

# reconstruct model Quantized 
Adpate_model.reconstruct_model()

# Iplmented LoRa Method
model = Adpate_model.implement_lora(verbose=True)
# Total trainable parameters before LoRA: 3176960
# Total trainable parameters after LoRA: 24576

# This sets requires_grad to False for all parameters without the string "lora_" in their names

# Training loop
for batch in dataloader:
    model.train()
```
### Saving Wieghts model 

* Save LoRA model (only the LoRA matrixes will be saved).

```python
import loralib as lora 
# ===== Before =====
# torch.save(model.state_dict(), checkpoint_path)
# ===== After =====
torch.save(lora.lora_state_dict(model), checkpoint_path)
```

### Loading the Pre-Trained Model 

* Load LoRA model (need to load the pre-trained model first).

```python
import loralib as lora 
# Load the pre-trained checkpoint first
model.load_state_dict(torch.load('ckpt_pretrained.pt'), strict=False)
# Then load the LoRA checkpoint
model.load_state_dict(torch.load('ckpt_lora.pt'), strict=False)
```


- <img src="assets/rocket.gif" width="32" height="32"/> Quantized Model <img src="assets/rocket.gif" width="32" height="32"/>

- <img src="assets/time.gif" width="32" height="32"/> Time to Train <img src="assets/time.gif" width="32" height="32"/>

- <img src="assets/money.gif" width="32" height="32"/> Cost to Train <img src="assets/money.gif" width="32" height="32"/>


## What's in it for you?

For each of the above four pillars, we are sharing our codebase and insights to:
- Assist you to leverage Transfomer-Based Model for your machines needs and challenges

- Boost reproducibility efforts which are becoming increasingly difficult with Transfomers 

i am providing Tool that are ready-to-use for Quantize the model:

- Finetuning Transfomer-Based on your proprietary dataset via PeFT methodologies such as LoRA and QLoRa

- Performing hyperparameter optimization to get the maximum performance out of these models

## What's the best way to use this repository?

Go over to the Transfomer-Based-specific directory that you are interested in, and open the ```README.md```. We have included details about the LLMs, followed by performance results on open-source datasets!

## Roadmap

Our plan is to perform these experiments on all the Transformer-Based model below. To that end, this is a tentative roadmap of the LLMs that we aim to cover:

- [x] TransfomerEncoder
- [x] TransfomerDecoder
- [x] Vision-Transfomer
- [x] minGPT 
- [x] OpenAI GPT-2 
- [ ] Inflection Pi **Under Progress**

## Correspondence

## Contributor

``AdapterLoRa`` is developed and maintained by 
''Youness ELbrag'' ([Email](younsselbrag@gmail.com) | [LinkedIn](https://www.linkedin.com/in/youness-el-brag-b13628203/))


