import torch
import torch.nn as nn
import math
import torch.functional as F
from model import clones

def attention(query, key, value, mask, dropout):
    '''
    1. 计算过程
        attention计算方式，对于输入的X矩阵，由$ \mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V $计算
        （Q * K^T）/sqrt(D_k)用于计算n个词之间的attention系数，结果为(n, n)矩阵
    2. 示例：
        query, 为(batch_size, head_num=8, word_num=10, 64)依次为batch大小，注意力头的大小, 词的数量，词Embeding的维度
        query, key， value维度分别(128, 8, 10, 64), (128, 8, 11, 64), (128, 8, 11, 64)
        query来自目标序列， key和value原序列
    '''
    d_k = query.size(-1) # -1维元素个数
    # 计算10个目标词之间的attention系数，前面两维度一样，后面两维进行相乘，10个词与11个词的attention_score, 
    # 维度分别(128, 8, 10, 64), (128, 8, 64, 11)
    # 由于是64维, 词向量维度的内积， 除以sqrt(d_k)防止scores值过大
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k) # 结果为（128， 8， 10， 11）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        # 依据mask矩阵为0， 对scores填充-1e9, 计算softmax时会忽视
    p_attn = F.softmax(scores, dim=-1) # 对scores最后一个维度做softmax，维度不变
    if dropout is not None:
        p_attn = dropout(p_attn) # 执行dropout
    # 计算 p_atten * V 的值， 由(128, 8, 10, 11) matmul (128, 8 , 11, 64)
    # 每个词向量64维度，计算词向量每个维度上与其他词的 “attention” 数值的内积
    return torch.matmul(p_attn, value), p_attn # 经过一次attetion后，维度依旧与query一样
    
    


class MultiHeadedAttention(nn.Module):
    '''
        MutiAttention模块
    '''
    def __init__(self, h, d_model, dropout=0.1):
        '''
            h为attention头的数量， d_model为维度
        '''
        # dropout的值怎么设置
        super(MultiHeadedAttention, self).__init__()





