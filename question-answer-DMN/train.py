#!/usr/bin/env python
# -*-coding: utf8-*-

'''
训练模型

@author: PLM
@date: 2018-03-10
'''
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from data_helper import get_variable, get_data_loader, get_data_from_file, pad_batch_data
from config import DefaultConfig
from model import DMN

def train(opt, th, train_data):
    ''' 训练
    Args:
        opt -- 配置信息
        th -- TextHelper实例
        train_data -- 训练数据，[[facts, question, answer]]
    '''
    # 加载原始数据
    seqbegin_id = th.word2index(th.seq_begin)
    
    model = DMN(th.vocab_size, opt.embed_size, opt.hidden_size, seqbegin_id, th.word2index(th.pad))
    if opt.use_cuda:
        model = model.cuda(opt.device_id)
    
    optimizer = optim.Adam(model.parameters(), lr = opt.learning_rate)
    loss_func = nn.CrossEntropyLoss(ignore_index=th.word2index(th.pad))
    
    for e in range(opt.max_epoch):
        losses = []
        for batch_data in get_data_loader(train_data, opt.batch_size, opt.shuffle):
            # batch内的数据进行pad，转成Variable
            allfacts, allfacts_mask, questions, questions_mask, answers = \
                    pad_batch_data(batch_data, th)
            
            # 前向
            preds = model(allfacts, allfacts_mask, questions, questions_mask, 
                          answers.size(1), opt.n_episode)
            # loss
            optimizer.zero_grad()
            loss = loss_func(preds, answers.view(-1))
            losses.append(loss.data.tolist()[0])
            # 反向
            loss.backward()
            optimizer.step()

        avg_loss = np.mean(losses)
        
        if avg_loss <= opt.min_loss or e % opt.print_every_epoch == 0 or e == opt.max_epoch - 1:    
            info = "e={}, loss={}".format(e, avg_loss)
            print (info)
            if e == opt.max_epoch - 1 and avg_loss > opt.min_loss:
                print ("epoch finish, loss > min_loss")
                torch.save(model, opt.model_path)
                break
            elif avg_loss <= opt.min_loss:
                print ("Early stop")
                torch.save(model, opt.model_path)
                break

                
def get_model(model_path):
    return torch.load(model_path)


def cal_test_accuracy(model, test_data, th, n_episode=DefaultConfig.n_episode):
    '''测试，测试数据'''
    batch_size = 1
    model.eval()
    correct = 0
    for item in get_data_loader(test_data, batch_size, False):
        facts, facts_mask, question, question_mask, answer = pad_batch_data(item, th)
        preds = model(facts, facts_mask, question, question_mask, answer.size(1), n_episode)
        #print (answer.data.shape, preds.data.shape)
        preds = preds.max(1)[1].data.tolist()
        answer = answer.view(-1).data.tolist()
        if preds == answer:
            correct += 1
    print ("acccuracy = ", correct / len(test_data)) 


def test_one_data(model, item, th, n_episode=DefaultConfig.n_episode):
    ''' 测试一条数据
    Args:
        model -- DMN模型
        item -- [facts, question, answer]
        th -- TextHelper
    Returns:
        None
    '''
    # batch_size = 1
    model.eval()
    item = [item]
    facts, facts_mask, question, question_mask, answer = pad_batch_data(item, th)
    preds = model(facts, facts_mask, question, question_mask, answer.size(1), n_episode)
    
    item = item[0]
    preds = preds.max(1)[1].data.tolist()
    fact = item[0][0]
    facts = [th.indices2sentence(fact) for fact in item[0]]
    facts = [" ".join(fact) for fact in facts]
    q = " ".join(th.indices2sentence(item[1]))
    a = " ".join(th.indices2sentence(item[2]))
    preds = " ".join(th.indices2sentence(preds))
    
    print ("Facts:")
    print ("\n".join(facts))
    print ("Question:", q)
    print ("Answer:", a)
    print ("Predict:", preds)
    print ()
    

if __name__ == '__main__':
    opt = DefaultConfig()
    train_data, th = get_data_from_file(opt.train_file, opt)
    print ("train_data:", len(train_data))
    train(opt, th, train_data)
    model = get_model(opt.model_path)
    test_data, test_th = get_data_from_file(opt.test_file, opt)
    print ("test_data:", len(test_data))
    cal_test_accuracy(model, test_data, th)
    for i in range(10):
        item = random.choice(test_data)
        #print (item)
        test_one_data(model, item, th)
