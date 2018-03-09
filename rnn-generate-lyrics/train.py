#!/usr/bin/env python
# encoding=utf-8

'''
训练生成诗歌模型

@author PLM
@date 2018-03-07
'''


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np


from model import CharRNN
from helper import TextConverter
from helper import TextDataset


USE_CUDA = True
DEVICE_ID = 0


def get_variable(x):
    x = Variable(x)
    if USE_CUDA:
        x = x.cuda(DEVICE_ID)
    return x


class DefaultConfig(object):
    # dataset
    train_data_path = './dataset/poetry.txt'
    num_workers = 4
    seq_len = 20
    max_vocab = 5000
    predict_len = 50

    # model parameters
    embed_size = 128
    hidden_size = 128
    n_layers = 1
    dropout_p = 0.1
    bidir = False

    # train
    batch_size = 128
    learning_rate = 1e-1
    max_epochs = 200
    min_perplexity = 2.72

    # model save
    model_path = "./models/PoetryCharRNN.pkl"


def train(opt, th):
    ''' 训练模型
    Args:
        opt -- 参数
        th -- TextConverter对象
    Returns:
        None
    '''
    # 1. 训练数据
    data_set = TextDataset(opt.train_data_path, th)
    train_data = DataLoader(data_set, opt.batch_size, 
                                shuffle=True, num_workers=opt.num_workers)
    # 2. 初始化模型
    model = CharRNN(th.vocab_size, opt.embed_size, opt.hidden_size, opt.n_layers,
                    opt.dropout_p, opt.bidir)
    if USE_CUDA:
        model = model.cuda(DEVICE_ID)

    # 3. 优化配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)


    # 4. 训练
    for e in range(opt.max_epochs):
        epoch_loss = 0
        hidden = None
        for input_seqs, labels in train_data:
            # 都是[b, seq_len]，最后一个不足b
            # 准备input和hidden
            b = input_seqs.shape[0]
            if hidden is not None:
                hidden = hidden[:, :b, :]
            labels = labels.long().view(-1)
            input_seqs, labels = get_variable(input_seqs), get_variable(labels)

            # 前向计算
            probs, hidden = model(input_seqs, hidden)
            probs = probs.view(-1, th.vocab_size)

            # loss和反向
            loss = criterion(probs, labels)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            # 优化
            nn.utils.clip_grad_norm(model.parameters(), 5)
            optimizer.step()

            epoch_loss += loss.data[0]
        # 交叉熵
        entropy_loss = epoch_loss / len(train_data)
        perplexity = np.exp(entropy_loss)
        info = "epoch: {}, perp: {:.3f}" .format(e+1, perplexity)
        print (info)
        if perplexity <= opt.min_perplexity or e == opt.max_epochs - 1:
            print ("best model")
            torch.save(model, opt.model_path)
            break


def get_model(opt):
    '''从磁盘中加载模型'''
    return torch.load(opt.model_path)


def pick_wordid(preds, topn=5):
    '''从topn中每次依它们的概率随机选择一个
    Args:
        preds -- CharRNN预测结果，[1, vocab_size]
        topn -- 概率最大的前n个
    Returns:
        wordid -- 最终选择的wordid
    '''
    values, indices= preds.topk(topn, dim=1)
    probs = values / torch.sum(values)
    probs = probs.squeeze(0).data.cpu().numpy()
    labels = indices.squeeze(0).data.cpu().numpy()
    wordid = np.random.choice(labels, size=1, p = probs)
    return wordid.tolist()[0]


def predict(model, th, opt, begin_sentence="天青色等烟雨", predict_len=None):
    '''给一个开始，写剩余的诗歌
    Args:
        model -- CharRNN模型
        th -- TextConverter
        opt -- 配置
        begin_sentence -- 开始的句子
        predict_len -- 剩余句子的长度
    Return:
        res -- 最终完整的句子
    '''
    if predict_len is None:
        predict_len = opt.predict_len
    model = model.eval()
    # 把begin_sentence放入，预热模型
    begin_indices = th.sentence2indices(begin_sentence)
    begin = get_variable(torch.LongTensor(begin_indices))
    # batch=1
    begin = begin.unsqueeze(0)
    _, hidden = model(begin)
    # 最后一个词作为新的输入
    input_seq = begin[0,-1].unsqueeze(0)

    res = begin_indices

    for i in range(predict_len):
        output, hidden =  model(input_seq)
        output = output.squeeze(1)
        wordid = pick_wordid(output)
        res.append(wordid)
        input_seq = get_variable(torch.LongTensor([wordid])).view(1, -1)
    return th.indices2sentence(res)


if __name__ == '__main__':
    opt = DefaultConfig()
    th = TextConverter(opt.train_data_path, opt.max_vocab)
    train(opt, th)
    model = get_model(opt)
    res = predict(model, th, opt, '床前明月光')
    print (res)
