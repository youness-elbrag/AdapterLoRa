# Adapter-LoRa for Quantization  

<div align="center">
  <img src="assets/LoRa.png" alt="Nano-AutoGrad Logo" width="200">
  
[![Made With Love](https://img.shields.io/badge/Made%20With-Love-orange.svg)](https://github.com/youness-elbrag/AdapterLoRa/)
[![GitHub issues](https://img.shields.io/github/issues/kyegomez/Med-Palm)](https://github.com/youness-elbrag/AdapterLoRa/issues) 
[![GitHub forks](https://img.shields.io/github/forks/kyegomez/Med-Palm)](https://github.com/youness-elbrag/AdapterLoRa/network) 
[![GitHub stars](https://img.shields.io/github/stars/kyegomez/Med-Palm)](https://github.com/youness-elbrag/AdapterLoRa/stargazers) [![GitHub license](https://img.shields.io/github/license/youness-elbrag/AdapterLoRa)](https://github.com/youness-elbrag/AdapterLoRa/blob/master/LICENSE)
[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/Med-Palm)](https://twitter.com/intent/tweet?text=Excited%20to%20introduce%20Med-Palm,%20the%20all-new%20robotics%20model%20with%20the%20potential%20to%20revolutionize%20automation.%20Join%20us%20on%20this%20journey%20towards%20a%20smarter%20future.%20%23RT1%20%23Robotics&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMed-Palm)
[![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMed-Palm)
[![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMed-Palm&title=Introducing%20Med-Palm%2C%20the%20All-New%20Robotics%20Model&summary=Med-Palm%20is%20the%20next-generation%20robotics%20model%20that%20promises%20to%20transform%20industries%20with%20its%20intelligence%20and%20efficiency.%20Join%20us%20to%20be%20a%20part%20of%20this%20revolutionary%20journey%20%23RT1%20%23Robotics&source=)
![Discord](https://img.shields.io/discord/999382051935506503)
[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMed-Palm&title=Exciting%20Times%20Ahead%20with%20Med-Palm%2C%20the%20All-New%20Robotics%20Model%20%23RT1%20%23Robotics) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMed-Palm&t=Exciting%20Times%20Ahead%20with%20Med-Palm%2C%20the%20All-New%20Robotics%20Model%20%23RT1%20%23Robotics)
[![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMed-Palm&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=Med-Palm%2C%20the%20Revolutionary%20Robotics%20Model%20that%20will%20Change%20the%20Way%20We%20Work%20%23RT1%20%23Robotics)
[![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=I%20just%20discovered%20Med-Palm,%20the%20all-new%20robotics%20model%20that%20promises%20to%20revolutionize%20automation.%20Join%20me%20on%20this%20exciting%20journey%20towards%20a%20smarter%20future.%20%23RT1%20%23Robotics%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2FMed-Palm)
</div>


If the last 6 months of AI research felt like a decade to you, you are not alone! With a new Large Language Model (LLM) released every other week, it has been challenging to keep up with the current pace of innovation in AI. While there many LLM model which not Non-Hungging Face model Hard to Quantize the model if realsed as Pre-trianed model , Adapter-LoRa is Tool help to Assign **nn.LInear-** to LoRa Linear Decompsition 


### Usage Tool AdpaterLoRa

```python

import torch.nn as nn
import torch
from core.Quantized import AdapterLoRa

model = nn.TransformerEncoderLayer(d_model=512, nhead=8)

Adpate_model = AdapterLoRa(model , method="LoRa", Rank=4)

# adding Linear Layer buitl Self.attention 
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

Our plan is to perform these experiments on all the LLMs below. To that end, this is a tentative roadmap of the LLMs that we aim to cover:

- [x] TransfomerEncoder
- [x] TransfomerDecoder
- [x] Vision-Transfomer
- [ ] BioMedGPT **Under Progress**
- [ ] SalesForce XGen **Under Progress**
- [ ] OpenAI GPT-2 **Under Progress**
- [ ] Inflection Pi **Under Progress**

## Correspondence

If you have any questions or issues, or would like to contribute to this repository, please reach out to:

- Youness ELbrag ([Email](younsselbrag@gmail.com) | [LinkedIn](https://www.linkedin.com/in/youness-el-brag-b13628203/))


