#!/usr/bin/env python
# -*-coding: utf8-*-

'''
配置文件

@author: PLM
@date: 2018-03-17
'''

class DefaultConfig(object):
    # dataset
    train_file = "./datasets/train-v1.1.json"
    test_file = "./datasets/dev-v1.1.json"
    context_maxlen = 600
    
    # 一些特殊符号
    seq_end = '</s>'
    seq_begin = '<s>'
    pad = '<pad>'
    unk = '<unk>'
    
    # cuda信息
    use_cuda = True
    device_id = 0
