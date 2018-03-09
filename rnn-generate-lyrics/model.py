#! /usr/bin/env python
# encoding=utf-8

'''
自动生成歌词的RNN模型

@author PLM
@date 2018-03-07
'''

import torch
import torch.nn as nn


class CharRNN(nn.Module):
    '''CharRNN embedding-gru-linear'''
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, dropout_p=0.1, bidir=False):
        '''init
        Args:
            vocab_size -- 词汇表大小
            embed_size -- 词编码维数，GRU的input_size
            hidden_size -- GRU的hidden_size
            n_layers -- 层数
            dropout_p -- 随机失活概率
            bidir -- 双向RNN
        '''
        super(CharRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.bidir = bidir
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=n_layers, batch_first=True, 
                          dropout=dropout_p, bidirectional=bidir)
        self.output2word = nn.Linear(hidden_size, vocab_size)

    def forward(self, sentences, hidden=None):
        '''输入一些句子和hidden_state
        Args:
            sentences -- [b, seq_len]
            hidden -- [n_l*n_dir, b, h]
        Returns:
            words -- [b, s, vocab_size]
            hidden -- [n_l*n_dir, b, h]
        '''
        self.gru.flatten_parameters()
        embeds = self.embedding(sentences)
        output, hidden = self.gru(embeds, hidden)
        if self.bidir:
            output = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]
        words = self.output2word(output).contiguous()
        return words, hidden
