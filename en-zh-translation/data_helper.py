#!/usr/bin/env python
# coding=utf-8

'''
英语-中文，数据处理
@author plm
@date 2017-10-15
'''

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import string
import random
import torch
from torch.autograd import Variable

PAD_token = 0
SOS_token = 1
EOS_token = 2
SHOW_LOG = True
USE_CUDA = True

def show_change():
    print("Hello")

class Lang(object):
    '''某一语言的辅助类，word2index, index2word, 词频等'''

    def __init__(self, name):
        self.name = name
        self.init_params()

    def init_params(self, trimmed = False):
        '''初始化参数'''
        # 修整标记
        self.trimmed = trimmed
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"PAD", 1:"SOS", 2:"EOS"}
        self.n_words = 3

    def index_word(self, word):
        '''添加一个词语'''
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def index_sentence(self, sentence, split_str=' '):
        '''添加一句话
        Args:
            sentence: 字符串，单词以空格分割
            split_str: 字符串单词分隔符，默认是空格
        '''
        for word in sentence.split(split_str):
            self.index_word(word)
    
    def index_words(self, words):
        '''添加词汇列表
        Args:
            words: 词汇列表
        '''
        for word in words:
            self.index_word(word)
    
    def trim(self, min_count, force=False):
        '''移除出现次数太少的单词
        Args:
            min_count: 最少出现次数
            force: 强制更新
        '''
        if self.trimmed and force is False:
            return
        keep_words = []
        
        for word, count in self.word2count.items():
            if count >= min_count:
                keep_words.append(word)
        ratio = round(len(keep_words) / self.n_words, 3)
        info = "keep words: {} / {} = {}".format(len(keep_words), self.n_words, ratio)
        showlog(info)
        
        # 重新更新参数，重新添加
        self.init_params(True)
        self.index_words(keep_words)


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_str(s):
    ''' 规整化字符串
    小写化，去掉特殊字符，给标点符号加上空格
    Args:
        s: 分词后的带空格的字符串。
    '''
    s = unicode_to_ascii(s.lower().strip())
    # 标点前+空格
    s = re.sub(r"([.!?！。，])", r" \1", s)
    # 去掉非汉字、非字母、非标点的字符
    s = re.sub(r'[^\u4e00-\u9fa5a-zA-Z.,!?。，！？ ]', r' ', s)
    # 多个空格用1个空格代替
    s = re.sub(r'\s+', r" ", s).strip()
    return s


def test_normalize_str():
    s = 'nihao你好，&&~~··你 在 干嘛呀！。干。。'
    print (s)
    sn = normalize_str(s)
    print (sn)


def showlog(info, force=SHOW_LOG):
    '''打日志'''
    if force:
        print (info)

    
def read_file(filename, max_read=20000):
    '''读取分词后的单个文件，返回List'''
    lines = []
    with open(filename, 'r') as f:
        idx = 0
        for line in f.readlines():
            line = normalize_str(line)
            lines.append(line)
            idx += 1
            if idx == max_read:
                break
        showlog('{}: read {}'.format(filename, idx))
    return lines


def read_data(en_file, zh_file, max_read=20000):
    ''' 读取英文中文文件。中文已经分好词。不通用的方法，后期再改
    Args:
        en_file, zh_file: 英文，中文文件的路径
        max_read: 单个文件最多读取多少条数据
    Returns:
        input_lang, target_lang: 两种语言的Lang对象
        pairs: 数据对，[[i1,o1], [i2,o2], ...]
    '''
    input_lang = Lang('English')
    target_lang = Lang('Chinese')
    input_lines = read_file(en_file, max_read)
    target_lines = read_file(zh_file, max_read)
    n_lines = len(input_lines)
    pairs = []
    print_every = 100000
    for i in range(n_lines):
        input_line = input_lines[i]
        target_line = target_lines[i]
        input_lang.index_sentence(input_line)
        target_lang.index_sentence(target_line)
        pairs.append([input_line, target_line])
        if i % print_every == 0:
            print ('%s/%s=%.3f' % (i, n_lines, i / n_lines))
    return pairs, input_lang, target_lang,


def legal_pair_bylen(pair, idx, max_len, min_len=1):
    ''' 根据pair中的某个句子的长度，判断是否留下
    Args:
        pair: [i, o]
        idx: 0或者1
        max_len: 最大长度
        min_len: 最小长度，默认是1
    '''
    words = pair[idx].split(' ')
    if len(words) >= min_len and len(words) <= max_len:
        return True
    return False
    

def legal_pair_bywords(pair, idx, lang):
    '''根据是否包含unknown词汇判断是否留下
    Args:
        pair: pair
        idx: 0或者1
        lang: idx语言对应的lang
    '''
    for w in pair[idx].split(' '):
        if w not in lang.word2index:
            return False
    return True
    

def remove_pairs(pairs, idx, lang=None, max_len=None, min_len=1):
    ''' 根据传入参数选择是否删掉pair
    lang:删掉unkonw词的pair。
    max_len, min_len: 选择不符合长度的pair
    Args:
        idx: 在pairs中的索引，0-input，1-target
        pairs:
        lang: 该语言的lang
        max_len: 最大长度
        min_len: 最小长度
    Return:
        pairs: 删除包含稀少单词的pair
    '''
    keep_pairs = []
    for p in pairs:
        keep = True
        if lang is not None:
            keep = legal_pair_bywords(p, idx, lang)
        if keep and max_len is not None:
            keep = legal_pair_bylen(p, idx, max_len, min_len)
        if keep:
            keep_pairs.append(p)
    
    ratio = round(len(keep_pairs) / len(pairs), 3)
    info = 'keep: {} / {} = {}'.format(len(keep_pairs), len(pairs), ratio)
    showlog(info)
    return keep_pairs


def indexes_from_sentence(lang, sentence):
    ''' 获得句子的词汇的id列表，加上结束标记'''
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def pad_seq(seq, max_length):
    ''' 为短句子填充到最大长度，填0
    Args:
        seq: 句子，以词汇id列表来表示
        max_length: 要填充到的长度
    Returns:
        seq: 填充好的句子
    '''
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def get_variable(tensor):
    ''' 直接获得variable，后面不用在判断，使用GPU或者不使用
    '''
    var = Variable(tensor)
    if USE_CUDA:
        var = var.cuda()
    return var


def random_batch(batch_size, pairs, input_lang, target_lang):
    ''' 随机选择一些样本
    Args:
        batch_size: 一批的大小
        pairs: 原数据
        input_lang, target_lang: 两种语言的工具类
    Returns:
        input_var: [s, b]，即[句子长度，s=句子个数]
        input_lengths: 真实长度 [b]，包含EOS
        target_var: [s, b]
        target_lengths: 真实长度 [b]，包含EOS
    '''
    input_seqs = []
    target_seqs = []
    
    # 随机选择pairs
    for i in range(batch_size):
        p = random.choice(pairs)
        input_seqs.append(indexes_from_sentence(input_lang, p[0]))
        target_seqs.append(indexes_from_sentence(target_lang, p[1]))
    
    # 选好之后按照大小输入句子长度排序
    seq_pairs = sorted(zip(input_seqs, target_seqs), key = lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    
    # 填充，真实长度，[b, maxlen]
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]
    
    # LongTensor (seq_len, batch_size)
    input_var = get_variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = get_variable(torch.LongTensor(target_padded)).transpose(0, 1)
    return input_var, input_lengths, target_var, target_lengths


def test_random_batch(pairs, input_lang, target_lang):
    input_var, in_lens, target_var, t_lens = random_batch(2, pairs, input_lang, target_lang)
    print ('input:', input_var.size(), in_lens)
    print ('target:', target_var.size(), t_lens)
    

if __name__ == '__main__':
    data_dir = './data'
    en_file = "{}/{}".format(data_dir, "seg_en")
    zh_file = "{}/{}".format(data_dir, "seg_zh")
    input_lang, target_lang, pairs = read_data(en_file, zh_file, 20000)
    pairs = remove_pairs(pairs, 0, max_len=25)
    test_random_batch(pairs, input_lang, target_lang)