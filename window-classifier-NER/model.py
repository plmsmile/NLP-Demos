#!/usr/bin/env python
# -*-coding: utf8-*-

'''
WindowClassifier模型

@author: PLM
@date: 2018-03-10
'''


import torch
import torch.nn as nn


class WindowClassifier(nn.Module):

    def __init__(self, vocab_size, tag_size, embed_size, hidden_size, 
                    window_size, dropout_p=0.3):
        super(WindowClassifier, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.window2hidden = nn.Linear(embed_size*(2*window_size+1), hidden_size)
        self.hidden2hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden2tag = nn.Linear(hidden_size, tag_size)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout

    def forward(self, inputs):
        '''
        Args:
            inputs -- [b, wsize]
        Returns:
            tag -- [b] 每个window中间词汇的类别
        '''
        # b*w*d
        embeds = self.embed(inputs)
        # 把window内的向量concat构成一个大向量。（累加会丢失距离信息）
        # b*(w*d)
        window = embeds.view(-1, embeds.size(1)*embeds.size(2))
        # b*h
        h0 = self.window2hidden(window)
        #h0 = self.dropout(h0)
        h1 = self.hidden2hidden(h0)
        h1 = self.relu(h1)
        #h1 = self.dropout(h1)

        # b*t
        tag = self.hidden2tag(h1)
        tag = self.softmax(tag)
        return tag
