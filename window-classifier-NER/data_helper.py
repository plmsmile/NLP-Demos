#!/usr/bin/env python
# -*-coding: utf8-*-

'''
Window classifier for NER 数据处理

@author: PLM
@date: 2018-03-10
'''

import nltk
import torch
import random
from torch.utils.data import DataLoader

from config import DefaultConfig


flatten = lambda l : [item for sublist in l for item in sublist]

class TextHelper(object):
    '''word2idx, tag2idx and so on.'''

    def __init__(self, data, dummy='<DUMMY>', unk='<UNK>'):
        sents, tags = list(zip(*data))
        sents = flatten(sents)
        tags = flatten(tags)
        # vocab
        vocab = list(set(sents))
        vocab.append(dummy)
        vocab.append(unk)
        # tags
        tagset = list(set(tags))
        tagset.append(unk)
        word2idx = {}
        for vo in vocab:
            word2idx[vo] = len(word2idx)
        idx2word =  {i:w for w, i in word2idx.items()}

        tag2idx = {}
        for tag in tagset:
            tag2idx[tag] = len(tag2idx)
        idx2tag = {i:t for t, i in tag2idx.items()}

        self.unk = unk
        self.dummy = dummy
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.tag2idx = tag2idx
        self.idx2tag = idx2tag

    @property
    def vocab_size(self):
        return len(self.word2idx) - 1

    @property
    def tag_size(self):
        return len(self.tag2idx)

    def word2index(self, word):
        return self.word2idx.get(word, self.word2idx[self.unk])

    def index2word(self, index):
        return self.idx2word.get(index, self.unk)

    def tag2index(self, tag):
        return self.tag2idx.get(tag, self.tag2idx[self.unk])

    def index2tag(self, index):
        return self.idx2tag.get(index, self.unk)


class TextDataset(object):
    '''构造数据集'''

    def __init__(self, data):
        self.data = data

    def __getitem__(self, i):
        ''' 获取第i条数据
        Args:
            i -- 第i条数据，即第i个window
        Returns:
            window_words: 2*w+1个的词语
            tag -- 中间词的tag
        '''
        words, tag = self.data[i]
        return torch.LongTensor(words), torch.LongTensor([tag])
    
    def __len__(self):
        return len(self.data)


def get_sentence_tags(corpus):
    '''获得语料库中的句子和对应的tags
    Args:
        corpus: 语料
    Returns:
        data: [(sent, tag)]，sent是分好词的句子，tag是句子中各个词对应的标记
    '''
    data = []
    for cor in corpus:
        sent, _, tag = list(zip(*cor))
        data.append((sent, tag))
    return data


def make_windows(source_data, th, dummy='<dummy>', window_size=2):
    ''' 构造(window, tag)数据
    Args:
        source_data -- [(sent, tag)]
        th -- TextHelper实例对象，用于word2id
        dummy -- 句子前后填充的字符
        window_size -- 中心词前后的单词数量，一共2*w+1
    Returns:
        windows -- [(2w+1个wordid, 中心词tag)]
    '''
    windows = []
    dummy = th.word2index(dummy)
    for sent, tags in source_data:
        sent = [th.word2index(w) for w in sent]
        tags = [th.tag2index(tag) for tag in tags]
        # 前后padding dummy数据
        padded = [dummy]*window_size + list(sent) + [dummy]*window_size
        # 多个长为2w+1的window，tag是中间词的标签
        min_windows = list(nltk.ngrams(padded, window_size * 2 + 1))
        pairs = []
        for i in range(len(sent)):
            pair = (min_windows[i], tags[i])
            pairs.append(pair)
        windows.extend(pairs)
    print ("all windows = ", len(windows))
    return windows


if __name__ == '__main__':
    opt = DefaultConfig()
    corpus = nltk.corpus.conll2002.iob_sents()[:10]
    raw_data = get_sentence_tags(corpus)
    th = TextHelper(raw_data, opt.dummy, opt.unk)
    windows = make_windows(raw_data, th, opt.dummy, opt.window_size)
    random.shuffle(windows)
    train_count = int (len(windows) * opt.train_ratio)
    train_windows = windows[:train_count]
    train_set = TextDataset(train_windows)
    train_data = DataLoader(train_set, opt.batch_size, 
                                shuffle=opt.shuffle, num_workers=opt.num_workers)
