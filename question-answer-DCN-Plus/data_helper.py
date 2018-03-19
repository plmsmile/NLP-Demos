#!/usr/bin/env python
# -*-coding: utf8-*-

'''
数据处理文件

@author: PLM
@date: 2018-03-17
'''
import torch
import json
import nltk
import re
from torch.autograd import Variable
from config import DefaultConfig

flatten = lambda l : [item for sublist in l for item in sublist]

is_logging = True


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
    
    @property
    def padid(self):
        return self.word2idx[self.pad]
    
    def word2index(self, word):
        return self.word2idx.get(word, self.word2idx[self.unk])
    
    def index2word(self, index):
        return self.idx2word.get(index, self.unk)
    
    def sentence2indices(self, sentence):
        '''单词列表 -- 单词id列表'''
        indices = list(map(lambda w: self.word2index(w), sentence))
        return indices
    
    def indices2sentence(self, indices):
        '''词汇id列表 -- 单词列表'''
        sentence = list(map(lambda index: self.index2word(index), indices))
        return sentence
    
    def add_vocabs(self, vocabs):
        ''' 给词汇表添加新的词汇
        Args:
            vocab -- 新的词汇列表
        '''
        for v in vocabs:
            if v not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[v] = idx
                self.idx2word[idx] = v


def log(info, forcing=None):
    ''' 记录一些基本信息
    Args:
        forcing -- 强制打印
    Returns:
        None
    '''
    if forcing is False:
        return
    if forcing is True or is_logging is True:
        print (info)
    return


def get_words(sentence):
    '''使用tokenizer把英语原始句子分开
    Args:
        sentence -- 原始英文句子，str
    Returns:
        words -- 分割后的单词，list
    '''
    return [w.replace("``", '"').replace("''", '"') for w in nltk.word_tokenize(sentence)]


def full2half(s):
    '''全角转半角'''
    d = {'\u3000': ' ',
         '，':',', 
         '。': '.',
         ',':',',
         '、':',',
         '【':'[',
         '】':'[',
         '（':'(',
         '）':')',
         '“': '"',
         '”': '"',
         '‘': '\'',
         '’': '\''
        }
    for ch, en in d.items():
        s = re.sub(ch, en, s)
    return s


def process_rawstr(rawstr):
    ''' 对原始的句子进行一些处理，\" 处理转义字符
    Args:
        rawstr -- 文件中读取处理的字符串
    Returns:
        处理后的字符串
    '''
    rawstr = full2half(rawstr)
    rawstr = rawstr.replace("''", '" ').replace("``", '" ')
    return rawstr


def get_word_index(word, context, chstart):
    ''' 找到sentence中以chstart开始的单词在context的words中的位置
    Args:
        sent -- 完整的sentence句子. str
        word -- 目标word
        context -- 段落字符串. str
        start -- sentence中目标单词w在context中的起始位置，要确保输入正确. int
    Returns:
        word_start -- w在context的words中的位置, 合法时. 不合法返回-1. int
    '''
    if chstart == 0:
        return 0
    words = get_words(context)
    
    r = chstart + len(word) - 1
    l = -1
    # 依次从右向左给w加一个字符，直到两者能够分割开来。分割点为pos
    for i in range(len(context) - chstart):
        l = chstart - i
        if (len(get_words(context[l:r+1])) == 2):
            break
    if l == -1:
        return 0
    word_index = len(get_words(context[0:l+1]))
    return word_index


def check(answerwords, contextwords):
    '''检查answer中的词汇与利用下标start从context中解析得到的词汇是否相等
    Args:
        answerwords -- 从answer中get_words得到的词汇
        contextwords -- 利用word_start, word_end从context中得到的词汇
    Returns:
        True -- match or False -- not match
    '''
    res = True
    for sw, cw in zip(answerwords, contextwords):
        if cw.find(sw) == -1:
            info = "ERROR. NotEqual. sw=[{}], cw=[{}]".format(sw, cw)
            log(info, False)
            res = False
    return res


def load_squad_data(file_path, max_len=600):
    ''' 从文件中读取数据
    Args:
        file_path -- 文件路径
        max_len -- 段落的最大长度
    Returns:
        data -- [[context, question, astart, aend]]
    '''
    documents = json.load(open(file_path, 'r'))['data']
    log("raw data: {}".format(len(documents)), True)
    datas = []
    skipped = 0
    nequal = 0
    # 多篇文章
    for i, d in enumerate(documents):
        title = d['title']
        # 多个段落
        for p in d['paragraphs']:
            context_str = process_rawstr(p['context'])
            context = get_words(context_str)
            if len(context) > max_len:
                skipped += 1
                continue
            # 多个问答对
            for qa in p['qas']:
                question_str = process_rawstr(qa['question'])
                question = get_words(question_str)
                # 目前的数据集实际上只有一个answer
                for a in qa['answers']:
                    answer_str = process_rawstr(a['text'])
                    answer = get_words(answer_str)
                    # 计算answer在context中的word级别的开始和结束地址
                    chstart = a['answer_start']
                    chend = chstart + len(answer_str) - len(answer[-1])
                    word_start = get_word_index(answer[0], context_str, chstart)
                    word_end = get_word_index(answer[-1], context_str, chend)
                    if check(answer, context[word_start:word_end+1]) is True:
                        item = [context, question, word_start, word_end]
                        datas.append(item)
                    else:
                        info = "s={}={}, e={}={}".format(word_start, context[word_start],
                                                         word_end, context[word_end])
                        nequal += 1
        if i % 20 == 0:
            log("{}/{}".format(i, len(documents)), True)
    log("{}/{}".format(len(documents), len(documents)), True)
    log("datas={}, skipped={}, nequal={}".format(len(datas), skipped, nequal), True)
    return datas


def make_digit_data(word_data, opt, th=None):
    '''构建词汇表，并把word形式的数据转换成id形式
    Args:
        worddata -- 单词形式的数据. [[document, question, start, end]]. word格式
        opt -- 配置文件
    Returns:
        data -- [[document, question, start, end]]. 数字id格式
        th -- 由数据集word_data构建的TextHelper实例
    '''
    documents, questions, starts, ends = list(zip(*word_data))
    vocabs = list(set(flatten(documents)).union(set(flatten(questions))))
    if th is None:
        th = TextHelper(vocabs, opt)
    else:
        th.add_vocabs(vocabs)
    
    data = []
    for d in word_data:
        item = []
        item.append(th.sentence2indices(d[0]))
        item.append(th.sentence2indices(d[1]))
        item.append(d[2])
        item.append(d[3])
        data.append(item)
    log("Build digit data success! vocab={}".format(th.vocab_size))
    return data, th


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


def pad_batch_data(batch_data, padid):
    '''把数据pad成同样长度的data
    Args:
        batch_data -- [[doc, q, start, end]]. id
        padid -- pad标记的wordid
    Returns:
        documents -- 文档. Variable. [b,dlen]
        questions -- 问题. Variable. [b,qlen]
        starts -- question在doc中的答案的起始值. [b]. Variable
        ends -- question在doc中的答案的结尾值. [b]. Vaariable
        dlens -- doc的真实长度. Variable. [b]
        qlens -- question的真实长度. Variable. [b]
    '''
    documents, questions, starts, ends = [list(l) for l in (zip(*batch_data))]
    bsize = len(documents)
    # 最大长度
    dlen = max([len(doc) for doc in documents])
    qlen = max([len(q) for q in questions])
    # 真实长度
    qlens = []
    dlens = []

    # pad，并记录真实长度
    for i in range(bsize):
        doc = documents[i]
        dlens.append(len(doc))
        if len(doc) < dlen:
            documents[i] = doc + [padid] * (dlen - len(doc))
        q = questions[i]
        if len(q) < qlen:
            questions[i] = q + [padid] * (qlen - len(q))
        qlens.append(len(q))
    
    documents = get_variable(torch.LongTensor(documents))
    questions = get_variable(torch.LongTensor(questions))
    dlens = get_variable(torch.LongTensor(dlens))
    qlens = get_variable(torch.LongTensor(qlens))
    starts = get_variable(torch.LongTensor(starts))
    ends = get_variable(torch.LongTensor(ends))
    return documents, questions, starts, ends, dlens, qlens


def get_data_from_file(file_path, opt):
    ''' 直接从文件里面读取数据转成id形式，并且构建th对象。
    Args:
        file_path -- 文件路径，仅限于SQuAD类型的数据
        opt -- 配置文件
    Returns:
        dataset -- [[document, question, start, end]]，id形式
        th -- TextHelper实例
    '''
    word_data = load_squad_data(file_path, opt.context_maxlen)
    dataset, th = make_digit_data(word_data, opt)
    return dataset, th


def test_get_word_index():
    context = "All of Notre Dame's undergraduate students are a part of one of the five undergraduate colleges at the school or are in the First Year of Studies program. The First Year of Studies program was established in 1962 to guide incoming freshmen in their first year at the school before they have declared a major. Each student is given an academic advisor from the program who helps them to choose classes that give them exposure to any major in which they are interested. The program also includes a Learning Resource Center which provides time management, collaborative learning, and subject tutoring. This program has been recognized previously, by U.S. News & World Report, as outstanding."
    sent = "U.S. News & World Report"
    context, sent = process_rawstr(context), process_rawstr(sent)
    sentwords = get_words(sent)
    contextwords = get_words(context)
    chstart = context.find(sent)
    chend = chstart + len(sent) - len(sentwords[-1])
    print ("context:\n{}\n".format(context))
    print (contextwords, "\n")
    print ("sent:\n{}\n".format(sent))
    print (sentwords, "\n")
    word_start = get_word_index(sentwords[0], context, chstart)
    word_end = get_word_index(sentwords[-1], context, chend)
    info = "s={}={}, e={}={}".format(word_start, contextwords[word_start],
                                     word_end, contextwords[word_end])
    print (info)
    check(sentwords, contextwords[word_start:word_end + 1])