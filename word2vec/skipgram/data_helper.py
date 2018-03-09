# encoding=utf-8

'''
Skip-gram's data helper

@author PLM
@date 2018-03-07
'''

import nltk
import torch
import random
from collections import Counter
from torch.utils.data import DataLoader

flatten = lambda l : [item for sublist in l for item in sublist]

class TextHelper(object):
    '''辅助类，构建词汇表，id-word互相转换'''
    def __init__(self, corpus, min_count=3, unktag='<UNK>'):
        '''初始化word2idex, index2word
        Args:
            corpus -- [[sentence], [sentence]]
            min_count -- 稀有词汇的词频标准
        '''
        corpus = [[word.lower() for word in sent]for sent in corpus]
        # 语料转成单词库
        corpus = flatten(corpus)
        word_count = Counter(corpus)
        # 稀有词汇 和 词汇表
        exclude = []
        for w, c in word_count.items():
            if c < min_count:
                exclude.append(w)
        vocab = list(set(corpus) - set(exclude))
        vocab.append(unktag)
        # 构建word2idx和idx2word
        word2idx = {}
        for vo in vocab:
            word2idx[vo] = len(word2idx)
        idx2word = {i:w for w, i in word2idx.items()}
        
        # unigram_table 用于负抽样，保存id
        num_total_words = sum([c for w, c in word_count.items() if w not in exclude])
        unigram_table = []
        Z = 0.001
        for vo in vocab:
            num = int(((word_count[vo] / num_total_words) ** 0.75) / Z)
            unigram_table.extend([word2idx[vo]] * num)
        
        print ("all=%d, vocab=%d, unigram=%d" % (len(word_count), len(vocab), len(unigram_table)))
        # 保存相应信息
        self.vocab = vocab
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.unk = unktag
        self.unigram_table = unigram_table
    
    def negative_sample(self, context_words, k=10):
        ''' 给每个上下文单词抽k个负例
        Args:
            context_words -- [b, 1], Tensor
            k -- 抽k个负例
        Returns:
            None
        '''
        batch_size = context_words.size(0)
        neg_samples = []
        for i in range(batch_size):
            sample = []
            wordid = int(context_words[i][0])
            while len(sample) < k:
                neg = random.choice(self.unigram_table)
                if neg == wordid:
                    continue
                sample.append(neg)
            neg_samples.append(sample)
        return neg_samples
    
    @property
    def vocab_size(self):
        return len(self.vocab)
    
    def word2index(self, word):
        return self.word2idx.get(word, self.word2idx[self.unk])
    
    def index2word(self, index):
        return self.idx2word.get(index, self.unk)
    
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
    '''构造数据集，'''
    def __init__(self, corpus, thelper, window_size=5, dummy='<DUMMY>'):
        '''
        Args:
            corpus -- 原始语料，[[sentence], [sentence]]
            thelper -- TextHelper辅助类，帮助word-id转换啊
            window_size -- 窗口大小
            dummy -- 空白填充token
        '''
        corpus = [[word.lower() for word in sent] for sent in corpus]
        windows = []
        for sentence in corpus:
            sentence = [dummy] * window_size + sentence + [dummy] * window_size
            # 从前向后依顺序构建长度为n的文法
            t = nltk.ngrams(sentence, 2 * window_size + 1)
            windows.append(list(t))
        windows = flatten(windows)
        
        data = []
        # 构建数据(center, context)
        for window in windows:
            for i in range(window_size * 2 + 1):
                # 上下文或中心词是稀有词汇
                if window[i] not in thelper.vocab or window[window_size] not in thelper.vocab:
                    continue
                # 中心词无意义
                if i == window_size or window[i] == dummy:
                    continue
                # (中心词，上下文)
                center_word = thelper.word2index(window[window_size])
                context_word = thelper.word2index(window[i])
                data.append((center_word, context_word))
        self.data = data
    
    def __getitem__(self, item):
        '''
        Returns:
            x -- 中心词id，LongTensor
            y -- 上下文id，LongTensor
        '''
        x, y = self.data[item]
        return torch.LongTensor([x]), torch.LongTensor([y])
    
    def __len__(self):
        return len(self.data)
    

def test_helper_dataset():
    corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))[:10]
    thelper = TextHelper(corpus)
    print (thelper.vocab_size)
    train_set = TextDataset(corpus, thelper)
    batch_size = 128
    train_data = DataLoader(train_set, batch_size, shuffle=True, num_workers=2)
    for x, y in train_data:
        print (x.shape, y.shape)
        neg_samples = thelper.negative_sample(y, 10)
        print (len(neg_samples))
        break
    print (len(train_set))


if __name__ == '__main__':
    test_helper_dataset()