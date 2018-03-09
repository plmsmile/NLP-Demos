#!/usr/bin/env python
# encoding=utf-8


'''
生成歌词的数据预处理

@author PLM
@date 2018-03-07
'''

import numpy as np
import torch


class TextConverter(object):
    def __init__(self, file_path, max_vocab_size=5000):
        ''' 读取文件初始化预料辅助类
        Args:
            file_path: 语料库文件
            max_vocab_size: 最大词汇表size
        '''
        with open(file_path, 'r') as f:
            corpus = f.read()
        # 去标点符号
        corpus = corpus.replace('\n', ' ').replace('\r', ' ').replace('，', ' ').replace('。', ' ')
        # 字典
        vocab = set(corpus)
        vocab_count = {}
        for w in corpus:
            vocab_count[w] = vocab_count.get(w, 0) + 1

        # 如果vocab_size太大，则去掉出现频率太低的字
        if (len(vocab) > max_vocab_size - 1):
            vocab_count = sorted(vocab_count.items(), key=lambda d:-d[1])[:max_vocab_size-1]
            vocab_count = dict(vocab_count)
            vocab = set(vocab_count.keys())

        # 未知词标记
        self.vocab = vocab
        self.unknown = '<UNK>'
        vocab.add(self.unknown)
        self.word2idx = {w : i for i, w in enumerate(vocab)}
        self.idx2word = {i : w for i, w in enumerate(vocab)}


    @property
    def vocab_size(self):
        '''字典大小，已包括<UNK>'''
        return len(self.vocab)

    def word2index(self, word):
        '''一个词汇的编号，从0开始编号'''
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx[self.unknown]

    def index2word(self, index):
        '''根据编号找到单词'''
        if index < self.vocab_size:
            return self.idx2word[index]
        return self.unknown

    def sentence2indices(self, sentence):
        '''句子编码成编号数组
        Args:
            setence -- 句子字符串
        Returns:
            indexes -- 句子编码数组 
        '''
        indexes = []
        for w in sentence:
            indexes.append(self.word2index(w))
        return indexes

    def indices2sentence(self, indexes):
        '''单词编码序列转成句子
        Args:
            indexes -- 单词编码数组
        Returns:
            sentence -- 句子
        '''
        words = []
        for idx in indexes:
            idx = int(idx)
            words.append(self.index2word(idx))
        return "".join(words)


class TextDataset(object):
    def __init__(self, file_path, text_helper, seq_len=20):
        ''' 初始化Dataset，把数据构建为LongTensor矩阵形式
        Args:
            file_path -- 数据文件
            text_helper -- 初始好的辅助类
            seq_len -- 句子的长度
        '''
        self.file_path = file_path
        with open(file_path, 'r') as f:
            corpus = f.read()
        corpus = corpus.replace('\n', ' ').replace('\r', ' ').replace('，', ' ').replace('。', ' ')
        # 划分为n_seq个长度为seq_len的句子，去掉剩余的
        n_seq = int(len(corpus) / seq_len)
        corpus = corpus[:n_seq * seq_len]
        corpus_indices = text_helper.sentence2indices(corpus)
        data = torch.LongTensor(corpus_indices)
        data = data.view(-1, seq_len)
        self.data = data

    def __getitem__(self, item):
        '''(我爱你, 爱你我) 这样构建数据对
        Returns:
            x -- 数据
            y -- 标签
        '''
        x = self.data[item, :]
        y = torch.zeros(x.shape)
        y[:-1], y[-1] = x[1:], x[0]
        return x, y

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    file_path = './dataset/jay.txt'
    th = TextConverter(file_path)
    data_set = TextDataset(file_path, th)
    x, y = data_set[0]
    print (x.shape, y.shape)
    print (th.indices2sentence(x.tolist()))
    print (th.indices2sentence(y.tolist()))

