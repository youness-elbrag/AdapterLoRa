# Adapter-LoRa for Quantization  

If the last 6 months of AI research felt like a decade to you, you are not alone! With a new Large Language Model (LLM) released every other week, it has been challenging to keep up with the current pace of innovation in AI. While there many LLM model which not Non-Hungging Face model Hard to Quantize the model if realsed as Pre-trianed model , Adapter-LoRa is Tool help to Assign **nn.LInear-** to LoRa Linear Decompsition 

```python
class Linear(nn.Module):
    def __init__(self , in_features , out_features, bais=False):
        super(Linear , self).__init__()
        self.a = nn.Paramerts(torch.zero(-1 , 0))
        self.b = nn.Paramters(torch.onse(-1 , 0)))
    def forward(self input):
        if bais=True:
            return x*a + b 
        return x * b
```

- <img src="assets/rocket.gif" width="32" height="32"/> Performance and productivety <img src="assets/rocket.gif" width="32" height="32"/>
- <img src="assets/time.gif" width="32" height="32"/> Time to Train <img src="assets/time.gif" width="32" height="32"/>
- <img src="assets/money.gif" width="32" height="32"/> Cost to Train <img src="assets/money.gif" width="32" height="32"/>


## What's in it for you?

For each of the above four pillars, we are sharing our codebase and insights to:
- Assist you to leverage Transfomer-Based Model for your business needs and challenges

- Boost reproducibility efforts which are becoming increasingly difficult with LLMs

i am providing Tool that are ready-to-use for Quantize the model:

- Finetuning Transfomer-Based on your proprietary dataset via PeFT methodologies such as LoRA and Prefix Tuning

- Performing hyperparameter optimization to get the maximum performance out of these models

## What's the best way to use this repository?

Go over to the TRansfomer-Based-specific directory that you are interested in, and open the ```README.md```. We have included details about the LLM, followed by performance results on open-source datasets!

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


