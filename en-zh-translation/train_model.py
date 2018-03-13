#!/usr/bin/env
# encoding=utf-8


import time

import torch
import torch.optim as optim

import data_helper as dh
from data_helper import get_variable
from model import *
from masked_cross_entropy import *
import show as sh
import train_helper as th


if __name__ == '__main__':
    data_dir = './data'
    en_file = "{}/{}".format(data_dir, "seg_en")
    zh_file = "{}/{}".format(data_dir, "seg_zh")
    TARGET_MAX_LEN = 25
    USE_CUDA = False
    pairs, input_lang, target_lang = dh.read_data(en_file, zh_file, 20000)
    
    # 模型配置
    encoder_bidir = False
    score_method = 'general'
    hidden_size = 500
    n_layers = 2
    dropout_p = 0.1
    batch_size = 50

    # 训练和优化配置
    clip = 50.0
    teacher_forcing_ratio = 0.5
    learning_rate = 0.0001
    decoder_learning_ratio = 5.0
    n_epochs = 20000
    epoch = 0
    plot_every = 20
    print_every = 100
    evaluate_every = 1000
    # n_epochs = 10
    # epoch = 0
    # plot_every = 2
    # print_every = 1
    # evaluate_every = 10
    save_every = 2000
    model_dir = './models/1017'

    train_conf = {'clip': clip, 'teacher_forcing_ratio': teacher_forcing_ratio}

    # 初始化模型
    encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers=n_layers, 
                         dropout_p=dropout_p, bidir=encoder_bidir)
    decoder = AttnDecoderRNN(hidden_size, target_lang.n_words, score_method=score_method, 
                             n_layers=n_layers, dropout_p=dropout_p)

    # 优化器和loss
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    
    #print (encoder)
    #print (decoder)

    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    
    ecs = []
    dcs = []
    eca = 0
    dca = 0

    while epoch < n_epochs:
        epoch += 1
        input_batches, input_lengths, target_batches, target_lengths = dh.random_batch(
            batch_size, pairs, input_lang, target_lang)
        loss, ec, dc = th.train(input_batches, input_lengths,
                             target_batches, target_lengths, encoder, decoder,
                             encoder_optimizer, decoder_optimizer, train_conf)
        print_loss_total += loss
        plot_loss_total += loss
        eca += ec
        dca += dc

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % (sh.time_since(start, float(epoch) / n_epochs),
                                                   epoch, epoch / n_epochs * 100, print_loss_avg)
            print (print_summary)

        if epoch % evaluate_every == 0:
            th.evaluate_randomly(pairs, input_lang, target_lang, encoder, decoder, False, False, False)

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
        
        if epoch % save_every == 0:
            no = epoch / save_every
            to = n_epochs / save_every
            s = '{}_{}'.format(no, to)
            torch.save(encoder, model_dir + '/' + s + 'encoder.pkl')
            torch.save(encoder, model_dir + '/' + s + 'decoder.pkl')
            print ('epoch=%d saved model' % epoch)