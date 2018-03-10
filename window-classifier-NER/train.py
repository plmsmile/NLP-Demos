#!/usr/bin/env python
# -*-coding: utf8-*-

'''
训练

@author: PLM
@date: 2018-03-10
'''



import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


from model import WindowClassifier
from config import DefaultConfig
from data_helper import *


def train(th, opt, train_data):
    ''' 训练模型
    '''

    model = WindowClassifier(th.vocab_size, th.tag_size, opt.embed_size,
                                opt.hidden_size, opt.window_size, opt.dropout_p)
    if opt.use_cuda:
        model = model.cuda(opt.gpu_id)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)

    for e in range(opt.max_epochs):
        losses = []

        for window, tag in train_data:
            #  把数据转成Variable
            inputs = opt.get_variable(window)
            y = opt.get_variable(tag).squeeze(1)
            # 前向
            yhat = model(inputs)
            # loss和反向
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.data.tolist()[0])

        avg_loss = np.mean(losses)
        if avg_loss <= opt.min_loss:
            info = "Best!! Epoch : %d, mean_loss : %.02f" % (e, avg_loss)
            print (info)
            losses = []
            break
        if e % opt.print_every_epoch == 0:
            info = "Epoch : %d, mean_loss : %.02f" % (e, avg_loss)
            print(info)
            losses = []
    torch.save(model, opt.model_path)


def get_model(model_path):
    return torch.load(model_path)


def get_data_loader(data, batch_size=1, shuffle=False, num_workers=1):
    ''' 从原始数据-Dataset-DataLoader'''
    data = TextDataset(data)
    return DataLoader(data, batch_size, shuffle=shuffle, num_workers=num_workers)


def go_test(model, test_data, opt, th):
    accuracy = 0
    ypred = []
    yreal = []
    for window, tag in test_data:
        inputs = opt.get_variable(window)
        y = tag.squeeze(1)
        yhat = model(inputs)
        values, yhat = torch.max(yhat, 1)

        yhat = yhat.data.cpu()
        eqres = torch.eq(yhat, y)
        accuracy += torch.sum(eqres)
        ypred.extend(yhat.tolist())
        yreal.extend(y.tolist())
    print ("accuracy = ", accuracy / len(test_data))


if __name__ == '__main__':
    # 配置和原始语料分析
    opt = DefaultConfig()
    corpus = nltk.corpus.conll2002.iob_sents()
    raw_data = get_sentence_tags(corpus)
    # 辅助类
    th = TextHelper(raw_data, opt.dummy, opt.unk)
    # 构建windows
    windows = make_windows(raw_data, th, opt.dummy, opt.window_size)
    random.shuffle(windows)
    # 训练数据
    train_data_num  = int (len(windows) * opt.train_ratio)
    train_data = windows[:train_data_num]
    train_data = get_data_loader(train_data, opt.batch_size, opt.shuffle, opt.num_workers)
    train(th, opt, train_data)

    # 加载模型
    model = get_model(opt.model_path)

    # 测试数据
    test_data = windows[train_data_num:]
    test_data = get_data_loader(test_data, 5)
    go_test(model, test_data, opt, th)
