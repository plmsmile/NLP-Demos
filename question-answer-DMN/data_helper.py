#!/usr/bin/env python
# -*-coding: utf8-*-

'''
数据处理

@author: PLM
@date: 2018-03-10
'''
import torch
import random
from copy import deepcopy
from torch.autograd import Variable
from config import DefaultConfig

flatten = lambda l : [item for sublist in l for item in sublist]


def get_variable(x, use_cuda=DefaultConfig.use_cuda, device_id=DefaultConfig.device_id):
    '''Tensor to Variable
    Args:
        x -- Tensor
        use_cuda -- 是否使用GPU，default=DefaultConfig.use_cuda
        device_id -- GPUID，device_id=DefaultConfig.device_id
    '''
    x = Variable(x)
    if use_cuda:
        x = x.cuda(device_id)
    return x


class TextHelper(object):
    '''构建word2idx，idx2word等。添加一些特殊符号'''

    def __init__(self, vocab, opt):
        # 额外的符号标记
        vocab.append(opt.seq_end)
        vocab.append(opt.seq_begin)
        vocab.append(opt.pad)
        vocab.append(opt.unk)
        # 构建dict
        word2idx = {}
        for vo in vocab:
            word2idx[vo] = len(word2idx)
        idx2word =  {i:w for w, i in word2idx.items()}
        self.pad = opt.pad
        self.seq_end = opt.seq_end
        self.seq_begin = opt.seq_begin
        self.unk = opt.unk
        self.word2idx = word2idx
        self.idx2word = idx2word
    
    @property
    def vocab_size(self):
        return len(self.word2idx)
    
    def word2index(self, word):
        return self.word2idx.get(word, self.word2idx[self.unk])
    
    def index2word(self, word):
        return self.word2idx.get(word, self.word2idx[self.unk])
    
    def sentence2indices(self, sentence):
        '''单词列表 -- 单词id列表'''
        indices = list(map(lambda w: self.word2index(w), sentence))
        return indices
    
    def indices2sentence(self, indices):
        '''词汇id列表 -- 单词列表'''
        sentence = list(map(lambda index: self.index2word(index), indices))
        return sentence


def load_raw_data(file_path, seq_end='</s>'):
    ''' 从文件中读取文本数据，并整合成[facts, question, answer]一条一条的可用数据，原始word形式
    Args:
        file_path -- 数据文件
        seq_end -- 句子结束标记
    Returns:
        data -- list，元素是[facts, question, answer]
    '''
    source_data = open(file_path).readlines()
    print (file_path, ":", len(source_data), "lines")
    # 去掉换行符号
    source_data = [line[:-1] for line in source_data]
    data = []
    for line in source_data:
        index = line.split(' ')[0]
        if index == '1':
            # 一个新的QA开始
            facts = []
            #qa = []
        if '?' in line:
            # 当前QA的一个问句
            # 问题 答案 答案所在句子的编号 \t分隔
            tmp = line.split('\t')
            question = tmp[0].strip().replace('?', '').split(' ')[1:] + ['?']
            answer = tmp[1].split() + [seq_end]
            facts_for_q = deepcopy(facts)
            data.append([facts_for_q, question, answer])
        else:
            # 普通的事件描述，简单句，只有.和空格
            sentence = line.replace('.', '').split(' ')[1:] + [seq_end]
            facts.append(sentence)
    return data 


def triple_word2id(triple_word_data, th):
    '''把文字转成id
    Args:
        triple_word_data -- [(facts, q, a)] word形式
        th -- textheler
    Returns:
        triple_id_data -- [(facts, q, a)]index形式
    '''
    # 把各个word转成数字id
    for t in triple_word_data:
        # 处理facts句子
        for i, fact in enumerate(t[0]):
            t[0][i] = th.sentence2indices(fact)
        # 问题与答案
        t[1] = th.sentence2indices(t[1])
        t[2] = th.sentence2indices(t[2])
    return triple_word_data


def get_data_loader(data, batch_size=1, shuffle=False):
    ''' 以batch的格式返回数据
    Args:
        data -- list格式的data
        batch_size -- 
        shuffle -- 每一个epoch开始的时候，对数据进行shuffle
    Returns:
        数据遍历的iterator
    '''
    if shuffle:
        random.shuffle(data)
    start = 0
    end = batch_size
    while (start < len(data)):
        batch = data[start:end]
        start, end = end, end + batch_size
        yield batch
    if end >= len(data) and start < len(data):
        batch = data[start:]
        yield batch 


def pad_batch_data(raw_batch_data, th):
    ''' 对数据进行padding，问题、答案、fact长度分别一致，同时每条数据的fact的数量一致。输入到网络的时候要用
    Args:
        raw_batch_data -- [[facts, q, a]]，都是以list wordid表示
        th -- TextHelper
    Returns:
        all_facts -- [b, nfact, flen]，pad后的facts，Variable
        all_facts_mask -- [b, nfact, flen]，facts的mask，Variable
        questions -- [b, qlen]，pad后的questions，Variable
        questions_mask -- [b, qlen]，questions的mask，Variable
        answers -- [b, alen]，pad后的answers，Variable
    '''
    all_facts, questions, answers = [list(i) for i in zip(*raw_batch_data)]
    batch_size = len(raw_batch_data)

    # 1. 计算各种长度。一个QA的facts数量，fact、Q、A句子的最大长度
    n_fact = max([len(facts) for facts in all_facts])
    flen = max([len(f) for f in flatten(all_facts)])
    qlen = max([len(q) for q in questions])
    alen = max([len(a) for a in answers])
    padid = th.word2index(th.pad)

    # 2. 对数据进行padding
    all_facts_mask = []
    for i in range(batch_size):
        # 2.1 pad fact
        facts = all_facts[i]
        for j in range(len(facts)):
            t = flen - len(facts[j])
            if t > 0:
                all_facts[i][j] = facts[j] + [padid] * t
        # fact数量pad
        while (len(facts) < n_fact):
            all_facts[i].append([padid] * flen)
        
        # 计算facts内容是否是填充给的，填充为1，不填充为0
        mask = [tuple(map(lambda v: v == padid, fact)) for fact in all_facts[i]]
        all_facts_mask.append(mask)
        
        # 2.2 pad question
        q = questions[i]
        if len(q) < qlen:
            questions[i] = q + [padid] * (qlen - len(q))
        # 2.3 pad answer
        a = answers[i]
        if len(a) < alen:
            answers[i] = a + [padid] * (alen - len(a))
    
    # 3. 把list数据转成Variable
    all_facts = get_variable(torch.LongTensor(all_facts))
    all_facts_mask = get_variable(torch.ByteTensor(all_facts_mask))
    answers = get_variable(torch.LongTensor(answers))
    questions = torch.LongTensor(questions)
    questions_mask = [(tuple(map(lambda v: v == padid, q))) for q in questions]
    questions_mask = torch.ByteTensor(questions_mask)
    questions, questions_mask = get_variable(questions), get_variable(questions_mask)
    return all_facts, all_facts_mask, questions, questions_mask, answers


def get_data_from_file(file_path, opt):
    ''' 直接从文件里面读取数据转成id形式，并且构建th对象。
    Args:
        file_path -- 文件路径，仅限于bAbI类型的数据
        opt -- 配置文件
    Returns:
        triple_id_data -- [[facts, question, answer]]，id形式
        th -- TextHelper实例
    '''
     # 构建数据
    triple_word_data = load_raw_data(file_path, opt.seq_end)
    facts, questions, answers = list(zip(*triple_word_data))
    # 构建词汇表及其辅助类
    vocab = list(set(flatten(flatten(facts)) + flatten(questions) + flatten(answers)))
    th = TextHelper(vocab, opt)
    # 数据转编码数据
    triple_id_data = triple_word2id(triple_word_data, th)
    return triple_id_data, th