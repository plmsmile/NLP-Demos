#!/usr/bin/env python
# -*-coding: utf8-*-

'''
NER配置文件

@author: PLM
@date: 2018-03-10
'''

from torch.autograd import Variable


class DefaultConfig(object):
    # dataset
    window_size = 2
    batch_size = 128
    num_workers = 2
    shuffle = True
    train_ratio = 0.9

    # model
    embed_size = 128
    hidden_size = 128
    dropout_p = 0.3

    # train
    learning_rate = 0.001
    max_epochs = 100
    print_every_epoch = 10
    min_loss = 0.001

    # cuda
    use_cuda = True
    gpu_id = 0

    def get_variable(self, x):
        x = Variable(x)
        if self.use_cuda:
            x = x.cuda(self.gpu_id)
        return x

    # others 
    unk = '<UNK>'
    dummy = '<DUMMY>'

    # model path
    model_path = './models/WindowClassifierNER.pkl'
