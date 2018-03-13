#!/usr/bin/env python
# encoding=utf-8
'''
模型训练方法
@author PLM
@date 2017-10-16
'''
from __future__ import print_function

import random
import torch

import data_helper as dh
from data_helper import get_variable
from masked_cross_entropy import masked_cross_entropy
import show
import time


def show_decoder_outputs(decoder_outputs, target_lang):
    maxv, maxi = decoder_outputs.max(-1)
    n = 10 if maxi[0].size(0) > 10 else maxi[0].size(0)
    mv, mi = maxv.cpu().data[0][:n].numpy().tolist(), maxi.cpu().data[0][:n].numpy().tolist()
    words = [target_lang.index2word.get(i, "UNK") for i in mi]
    print ("decoder_max:", words)


def get_sentence(wordids, lang):
    words = [lang.index2word.get(wid, 'UNK') for wid in wordids]
    return ' '.join(words)
    
def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder,
         encoder_optimizer, decoder_optimizer, loss_func, train_conf, input_lang, target_lang):
    '''训练一批数据
    Args:
        input_batches， input_lengths: [is, b] [b]，长度包含EOS，不包含SOS
        target_batches, target_lengths: [ts, b], [b]
        encoder, decoder, optimizer: 
        train_conf: 训练时的配置文件
    '''
    batch_size = len(input_lengths)
    ts = target_batches.size(0)
    # 1. zero grad
    zerograd_start = time.time()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    zerograd_end = time.time()
   
    # 2. 输入encoder
    encoder_start = time.time()
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths)
    encoder_end = time.time()
   
    # 3. decoder 默认输入
    decoder_start = time.time()
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    
    # 3.1 先过SOS
    sos = [dh.SOS_token]* batch_size
    sos = [sos for i in range(ts)]
    sos = get_variable(torch.LongTensor(sos))
    decoder_outputs, decoder_hidden, attn_weights = decoder(sos, decoder_hidden, encoder_outputs)
    
    # 4. 输入到decoder
    max_target_len = max(target_lengths)
    # (ts,b,o)
    decoder_outputs, decoder_hidden, attn_weights = decoder(target_batches, decoder_hidden, encoder_outputs)
    decoder_end = time.time()
    
    #show_decoder_outputs(decoder_outputs, target_lang)
    #maxv, maxi = decoder_outputs.max(-1)
    # (b,ts,o) (b,ts)
    decoder_outputs = decoder_outputs.transpose(0, 1)
    target_batches = target_batches.transpose(0, 1)
    
    loss = 0
    for i in range(batch_size):
        tlen = target_lengths[i]
        # print (tlen, decoder_outputs[i].size())
        input = decoder_outputs[i][:tlen]
        target = target_batches[i][:tlen]
        #print (input.size(), target.size())
        loss += loss_func(input, target)
    
    contig_start = time.time()
    # logits = maxi.transpose(0, 1).contiguous()
    # target = target_batches.transpose(0, 1).contiguous()
    contig_end = time.time()
    # print (type(logits.data), type(target.data))
    # print (logits.size(), target.size())
    loss_start = time.time()
    # loss = masked_cross_entropy(logits, target, target_lengths)
    
    #loss = loss_func(logits, target)
    loss.backward()
    loss_end = time.time()
   

    optim_start = time.time()
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), train_conf['clip'])
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), train_conf['clip'])
    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()
    optim_end = time.time()
    
    zerograd_use = zerograd_end - zerograd_start
    encoder_use = encoder_end - encoder_start
    decoder_use = decoder_end - decoder_start
    contig_use = contig_end - contig_start
    loss_use = loss_end - loss_start
    optim_use = optim_end - optim_start
    
    #info = "%.3f, %.3f, %.3f, %.3f, %.3f, %.3f " % (zerograd_use, encoder_use, decoder_use,
    #                                          contig_use, loss_use, optim_use)
    #print (info)
    
    input_wordids = input_batches.transpose(0, 1)[0].cpu().data.tolist()[:input_lengths[0]-1]
    input_sentence = get_sentence(input_wordids, input_lang)
    target_wordids = target_batches.transpose(0, 1)[0].cpu().data.tolist()[:input_lengths[0]-1]
    target_sentence = get_sentence(target_wordids, target_lang)
    # print (sentence)
    #evaluate_sentence(input_sentence, input_lang, target_lang, encoder, decoder, print_res=True,
    #                  target_sentence=target_sentence, show_attention=False, show_in_visdom=False)
    #evaluate(sentence, input_lang, target_lang, encoder, decoder, target_maxlen=target_lengths[0] + 2)
    
    return loss.data[0], ec, dc


def evaluate(input_sentence, input_lang, target_lang, encoder, decoder, target_maxlen=25):
    ''' 验证一条句子
    Args:
        input_sentence: 输入的一个句子，原字符句子，不包含EOS
        target_maxlen: 翻译目标句子的最大长度，不包括EOS_token的长度
    Returns:
        decoded_words: 翻译后的词语
        decoder_attentions: Attention [目标句子长度，原句子长度]
    '''
    batch_size = 1
    # [s,1] 包含EOS 
    input_batches = [dh.indexes_from_sentence(input_lang, input_sentence)]
    # [1, s]
    input_batches = get_variable(torch.LongTensor(input_batches)).transpose(0, 1)
    input_lengths = [len(input_batches)]    
    
    # 非训练模式，避免dropout
    encoder.train(False)
    decoder.train(False)
    
    # [s,b,h],[nl,b,h]过encoder，准备decoder数据
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    #print (encoder_outputs.data[0][0][:10])
    # (ts,b)
    decoder_input = decoder.create_input_seqs(1, batch_size)
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    # 最终结果
    decoded_words = []
    decoder_attentions = torch.zeros(target_maxlen + 1, input_lengths[0])
    
    # 过decoder
    for di in range(target_maxlen):
        # 这里ts=b=1，即[1,1,o],[nl,1,h],[1,1,is]，原本[ts,b,o], [nl,b,h], [b,ts,is]
        # print ("input:", decoder_input.data.tolist())
        decoder_output, decoder_hidden, attn_weights = \
            decoder(decoder_input, decoder_hidden, encoder_outputs)
        # print ("attn:", attn_weights.data.tolist())
        # word信息
        word_id = parse_output(decoder_output).squeeze().cpu().data.numpy().tolist()[0]
        #show_decoder_outputs(decoder_output, target_lang)
        #maxv, maxi = decoder_output.squeeze().max(-1)
        #print ("evaluate:", word_id, maxv.data[0], maxi.data[0])
        word = target_lang.index2word[word_id]
        decoded_words.append(word)
        # attention
        decoder_attentions[di] += attn_weights.cpu().data.squeeze()
        
        if word_id == dh.EOS_token:
            break
        # 当前单词作为下一个的输入(ts,b)=(1,1)
        decoder_input = get_variable(torch.LongTensor([word_id])).view(1, -1)
    
    # 改变encoder的模式
    encoder.train(True)
    decoder.train(True)
    res = decoder_attentions[:di+1,:]
    #print ('input_length:{}, di={}, size={}'.format(input_lengths[0], di, res.size()))
    return decoded_words, res


def evaluate_randomly(pairs, input_lang, target_lang, encoder, decoder,
                      print_res=False, show_attention=False, show_in_visdom=False):
    ''' 随机翻译一条句子，并且打印结果 '''
    [input_sentence, target_sentence] = random.choice(pairs)

    evaluate_sentence(input_sentence, input_lang, target_lang,
                      encoder, decoder, target_sentence=target_sentence, print_res=print_res,
                      show_attention=show_attention, show_in_visdom=show_in_visdom)


def parse_output(output):
    ''' 解析得到output中的words信息，id表示。输入不受维数限制
    Args:
        output: [target_seqlen, batch_size, output]
    Returns:
        word_ids: [target_seqlen, batch_size] 翻译出来的word_id信息
    '''
    # -1最后一维，输入2也可以。保留前面的维数
    maxv, maxi = output.max(-1)
    return maxi


def evaluate_sentence(input_sentence, input_lang, target_lang, encoder, decoder, print_res=False,
                      target_sentence=None, show_attention=False, show_in_visdom=False):
    '''翻译并评估一条句子'''
    target_maxlen = 25
    if target_sentence is not None:
        target_maxlen = len(target_sentence.split()) + 2
    output_words, attentions = evaluate(input_sentence, input_lang, target_lang, 
                                       encoder, decoder, target_maxlen=target_maxlen)
    output_sentence = ' '.join(output_words)
    if print_res:
        print('>', input_sentence)
        if target_sentence is not None:
            print('=', target_sentence)
        print('< ', output_sentence)
    
    if show_attention:
        show.show_attention(input_sentence, output_words, attentions, 
                            target_sentence=target_sentence, show_in_visdom=show_in_visdom)
    
