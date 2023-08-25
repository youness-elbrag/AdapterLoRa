import torch.nn as nn 
import torch
import math 


class ScaleDotProductAttention(nn.Module):

    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeur    y(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self,config):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.attention_dropout = nn.Dropout(config["attention_droput"])

    def forward(self, q, k, v,output_attentions=False):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)
        score =  self.attention_dropout(score)

        # 4. multiply with Value
        v = score @ v
        if not output_attentions:
            return (v, None)

        return v, score


class MultiHeadAttention(nn.Module):

    def __init__(self,config):
        super(MultiHeadAttention, self).__init__()
        self.n_head = Config["num_heads"]
        self.attention = ScaleDotProductAttention(config)
        self.w_q = nn.Linear(config["embedding_size"], config["embedding_size"],bias = config["qkv_bias"])
        self.w_k = nn.Linear(config["embedding_size"], config["embedding_size"],bias = config["qkv_bias"])
        self.w_v = nn.Linear(config["embedding_size"], config["embedding_size"],bias = config["qkv_bias"])
        self.w_concat = nn.Linear(config["embedding_size"], config["embedding_size"])

    def forward(self, q, k, v,output_attentions=False):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v,output_attentions=output_attentions)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization
        if not output_attentions:
            return (out, None)

        return out , attention

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
