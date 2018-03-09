# encoding=utf-8

'''
word2vec skipgram negative sampling model

@author PLM
@date 2018-03-09
'''

import torch
import torch.nn as nn

class SkipgramNegSampling(nn.Module):
    '''skipgram模型，训练v和u'''

    def __init__(self, vocab_size, embed_size):
        super(SkipgramNegSampling, self).__init__()
        self.embedding_v = nn.Embedding(vocab_size, embed_size)
        self.embedding_u = nn.Embedding(vocab_size, embed_size)
        self.logsigmoid = nn.LogSigmoid()
        
        init_range = (2.0 / (vocab_size + embed_size)) ** 0.5
        self.embedding_v.weight.data.uniform_(-init_range, init_range)
        self.embedding_u.weight.data.uniform_(-0.0, 0.0)

    def forward(self, center_words, context_words, negative_words):
        '''
        Args:
            center_words -- 中心单词，[b, 1]
            context_words -- 目标（上下文）单词，[b, 1]
            negative_words -- 目标单词的负采样单词，[b, k]
        Returns:
            loss -- -(正分+负分)
        '''
        # [b, 1, d]
        center_embeds = self.embedding_v(center_words)
        context_embeds = self.embedding_u(context_words)
        # [b, k, d]
        neg_embeds = -self.embedding_u(negative_words)
        # 计算得分
        # [b,1,d] [b,d,1]=[b,1,1]=[b,1]
        positive_score = context_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        # [b,k,d] [b,1,d]=[b,k,1]=[b,k]=[b,1]
        negative_score = neg_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)
        negative_score = torch.sum(negative_score, 1).view(-1, 1)
        
        # log loss
        loss = self.logsigmoid(positive_score) + self.logsigmoid(negative_score)
        return -torch.mean(loss)

    def get_vector(self, inputs):
        ''' 求单词的向量
        Args:
            inputs -- 输入的单词id
        Returns:
            embeds -- 词向量
        '''
        embeds = self.embedding_v(inputs)
        return embeds
