#!/usr/bin/env python
# -*-coding: utf8-*-

'''
DMN-QA模型配置文件

@author: PLM
@date: 2018-03-10
'''

class DefaultConfig(object):
    '''配置文件'''
    # 数据信息
    train_file = "./datasets/tasks_1-20_v1-2/en-10k/qa5_three-arg-relations_train.txt"
    test_file = "./datasets/tasks_1-20_v1-2/en-10k/qa5_three-arg-relations_test.txt"
    
    # 一些特殊符号
    seq_end = '</s>'
    seq_begin = '<s>'
    pad = '<pad>'
    unk = '<unk>'
    
    # DataLoader信息
    batch_size = 128
    shuffle = False
    # TODO
    num_workers = 1
    
    # model
    embed_size = 64
    hidden_size = 64
    # 对inputs推理的轮数
    n_episode = 3
    dropout_p = 0.1
    
    # train
    max_epoch = 500
    learning_rate = 0.001
    min_loss = 0.01
    print_every_epoch = 5
    
    # cuda信息
    use_cuda = True
    device_id = 0
    
    # model_path
    model_path = "./models/DMN.pkl"