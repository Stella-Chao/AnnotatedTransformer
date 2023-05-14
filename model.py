import torch.nn as nn
import copy
from torch.nn.functional import log_softmax


def clones(module, n):
    # 返回N层的moduleList
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

class Generator(nn.Module):
    # linear + softmax
    def __init__(self, d_model, vocab):
        # d_model至vocab维度 Linear
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
    def forword(self, x):
        return log_softmax(self.proj(x), dim=-1)

class EncoderDecoder(nn.Module):
    '''
        Encoder-Decoder Model
        src数据 -- 编码器 -- hidden -- 解码器 -- tgt结构
        encoder, decoder: 编码器 解码器
        src_embed, tgt_embed: 源数据embedding方法，目标数据embedding方法
        generator: linear + softmax generation step
    '''
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forword(self, src, tgt, src_mask, tgt_mask):
        # src数据 -- 编码器 -- hidden -- 解码器 -- tgt结构
        return self.decoder(self.encoder(src, src_mask),  # src embedding mask
                            src_mask,
                            tgt,
                            tgt_mask
                            )

    def encode(self, src, src_mask):
        # 对embed的数据做掩码处理
        return self.encoder(self.src_embed(src), src_mask)

    def decoder(self, memory, src_mask, tgt, tgt_mask):
        # memory即encoding部分
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)




