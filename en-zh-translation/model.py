#!/usr/bin/env python
# encoding=utf-8

'''
翻译模型

@author PLM
@date 2017-10-16
'''
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from masked_cross_entropy import *
import data_helper as helper
from data_helper import get_variable

def my_log_softmax(x):
    '''只能处理3维的'''
    size = x.size()
    res = F.log_softmax(x.squeeze())
    res = res.view(size[0], size[1], -1)
    return res

class EncoderRNN(nn.Module):
    ''' 对句子进行编码 input-embeded-gru-output 
    [s, batch_size] -- [s, b, h]，即[句子长度，句子个数] -- [句子长度，句子个数，编码维数]
    '''
    def __init__(self, vocab_size, hidden_size, n_layers=1, dropout_p=0.1, bidir=False):
        super(EncoderRNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.bidir = bidir
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, 
                          dropout=dropout_p, bidirectional=bidir)
    
    def forward(self, input_seqs, input_lengths, hidden=None):
        ''' 对输入的多个句子经过GRU计算出语义信息
        1. input_seqs > embeded
        2. embeded - packed > GRU > outputs - pad -output
        Args:
            input_seqs: [s, b]
            input_lengths: list[int]，每个batch句子的真实长度
        Returns:
            outputs: [s, b, h]
            hidden: [n_layer, b, h]
        '''
        # 一次运行，多个batch，多个序列
        embedded = self.embedding(input_seqs)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_length = nn.utils.rnn.pad_packed_sequence(outputs)  
        
        # 双向，两个outputs求和
        if self.bidir is True:
            outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden


class Attn(nn.Module):
    '''计算对齐向量，只有general可以使用'''
    def __init__(self, score_type, hidden_size):
        '''
        Args:
            score_type: 计算score的方法，'dot', 'general', 'concat'
            hidden_size: Encoder和Decoder的hidden_size
        '''
        super(Attn, self).__init__()
        self.score_type = score_type
        self.hidden_size = hidden_size
        if score_type == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif score_type == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
    
    def score(self, decoder_rnn_output, encoder_output):
        ''' 计算Decoder中yt与Encoder中hs的打分。算出所有得分，再softmax就可以算出对齐向量。
        下面均是单个batch
        Args:
            decoder_rnn_output: [1, h]，Decoder中顶层RNN的输出[1,h] < [1,b,h]
            encoder_output: [1, h]，Encoder最后的输出[1,h] < [s,b,h]>
        Returns:
            energy: 即Yt与Xs的得分
        '''
        # dot 需要两个1维的向量
        if self.score_type == 'dot':
            energy = decoder_rnn_output.squeeze(0).dot(encoder_output.squeeze(0))
        elif self.score_type == 'general':
            energy = self.attn(encoder_output)
            energy = decoder_rnn_output.squeeze(0).dot(energy.squeeze(0))
        elif self.score_type == 'concat':
            h_o = torch.cat((decoder_rnn_output, encoder_output), 1)
            energy = self.attn(h_o)
            energy = self.v.squeeze(0).dot(energy.squeeze(0))
        return energy
    
    def forward(self, rnn_outputs, encoder_outputs):
        '''ts个时刻，计算ts个与is的对齐向量，也是注意力权值
        Args:
            rnn_outputs: Decoder中GRU的输出[ts, b, h]
            encoder_outputs: Encoder的最后的输出, [is, b, h]
        Returns:
            attn_weights: Yt与所有Xs的注意力权值，[b, ts, is]
        '''
        target_seqlen = rnn_outputs.size()[0]
        input_seqlen = encoder_outputs.size()[0]
        batch_size = encoder_outputs.size()[1]
        
        # (b, ts, h) (b, is, h)
        rnn_outputs = rnn_outputs.transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        
        if self.score_type == 'general':
            # (b, h, is)
            encoder_outputs = self.attn(encoder_outputs).transpose(1, 2)
            # [b,ts,is] <[b,ts,h]*[b,h,is]
            attn_energies = rnn_outputs.bmm(encoder_outputs)
            res = my_log_softmax(attn_energies)
            return res
        
        # attn_energies (b, s)
        attn_energies = get_variable(torch.zeros(batch_size, target_seqlen, input_seqlen))
        for b in range(batch_size):
            # (1, h) 当前一个GRU的输出
            decoder_rnn_output = rnn_outputs[b]
            for i in range(seq_len): 
                # (1, h) < (s, 1, h) 
                encoder_output = encoder_outputs[i, b, :].squeeze(0)
                attn_energies[b, i] = self.score(decoder_rnn_output, encoder_output)
        
        attn_weights = get_variable(torch.zeros(this_batch_size, seq_len))
        for b in range(this_batch_size):
            attn_weights[b] = F.softmax(attn_energies[b])
        return attn_weights
 

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, score_method='general', n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.score_method = score_method
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        
        # 选择attention
        if score_method != 'none':
            self.attn = Attn(score_method, hidden_size)
    
    def forward(self, input_seqs, last_hidden, encoder_outputs):
        '''
        一次输入(ts, b)，b个句子, ts=target_seq_len
        1. input > embedded 
        2. embedded, last_hidden --GRU-- rnn_output, hidden
        3. rnn_output, encoder_outpus --Attn-- attn_weights
        4. attn_weights, encoder_outputs --相乘-- context
        5. rnn_output, context --变换,tanh,变换-- output 
        Args:
            input_seqs: [ts, b] batch个上一时刻的输出的单词，id表示。每个batch1个单词
            last_hidden: [n_layers, b, h]
            encoder_outputs: [is, b, h]
        Returns:
            output: 最终的输出，[ts, b, o]
            hidden: GRU的隐状态，[nl, b, h]
            attn_weights: 对齐向量，[b, ts, is]
        '''
        batch_size = input_seqs.size()[1]
        ts = input_seqs.size()[0]
        ins = encoder_outputs.size()[0]
        
        embedded_start = time.time()
        
        embedded = self.embedding(input_seqs)
        embedded = embedded.view(ts, batch_size, self.hidden_size)
        
        # (ts, b, h), (nl, b, h)
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # [ts, b, is]
        
        attn_start = time.time()
        # 对齐向量 [b,ts,is]
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # print (rnn_output.data.squeeze().tolist()[:10])
        #print ("attn_weights:", attn_weights.data.tolist()[:10])
        # show_tensor("attn_weights", attn_weights)
        attn_end = time.time()
        # 新的语义 [b,ts,h] < [b,ts,is] * [b,is,h].
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # [ts,b,h] < 
        context = context.transpose(0, 1)
        # show_tensor("context", context)
        
        # 语义和输出 [ts, b, 2h] < [ts, b, h], [ts, b, h]
        output_context = torch.cat((rnn_output, context), 2)
        # [ts, b, h]
        output_context = self.concat(output_context)
        concat_output = F.tanh(output_context)
        
        # [ts, b, o]
        output = self.out(concat_output)
        # show_tensor("output", output)
        output = my_log_softmax(output)
        #print (output.max(-1)[0].data[0][0])
        output_end = time.time()
        
        rnn_use = attn_start - embedded_start
        attn_use = attn_end - attn_start
        remain_use = output_end - attn_end
        #print ('%.3f, %.3f, %.3f' % (rnn_use, attn_use, remain_use))
        # show_tensor("output", output)
        return output, hidden, attn_weights

    def init_outputs(self, seq_len, batch_size):
        outputs = torch.zeros(seq_len, batch_size, self.output_size)
        return get_varaible(outputs)
    
    def create_input_seqs(self, seq_len, batch_size):
        sos = [helper.SOS_token] * batch_size
        sos = [sos] * seq_len
        return get_variable(torch.LongTensor(sos))

def show_tensor(name, tensor, num=10):
    num = 10 if tensor.size(-1) < 10 else tensor.size(-1)
    print (name, tensor.squeeze().data.tolist()[:10])

def test_encoder(pairs, input_lang, target_lang):
    small_batch_size = 2
    input_batches, input_lengths, target_batches, target_lengths \
        = helper.random_batch(small_batch_size, pairs, input_lang, target_lang)
    print ('input:', input_batches.size())
    print ('target:', target_batches.size())

    small_hidden_size = 8
    small_n_layers = 2
    encoder_test = EncoderRNN(input_lang.n_words, small_hidden_size, small_n_layers, bidir=False)
    print (encoder_test)
    encoder_outputs, encoder_hidden = encoder_test(input_batches, input_lengths)
    print ('outputs:', encoder_outputs.size(), 'hidden:', encoder_hidden.size())
    

def test_model(pairs, input_lang, target_lang):
    batch_size = 2
    input_batches, input_lengths, target_batches, target_lengths \
        = helper.random_batch(batch_size, pairs, input_lang, target_lang)
    
    print ('input:', input_batches.size(), input_lengths)
    print ('target:', target_batches.size(), target_lengths)

    hidden_size = 8
    n_layers = 2
    encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers=n_layers, bidir=False)  
    decoder = AttnDecoderRNN(hidden_size, target_lang.n_words, n_layers=n_layers)
    
    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    
    print (decoder)
    print (encoder)
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths)
    print ('outputs:', encoder_outputs.size(), 'hidden:', encoder_hidden.size())
    max_target_len = max(target_lengths)
    decoder_input = decoder.create_input_seq(batch_size)
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    # (s, b, o)
    all_decoder_outputs = get_variable(torch.zeros(max_target_len, batch_size, decoder.output_size))
    
    use_teacher_forcing = random.random() < 1
    for t in range(max_target_len):
        #(b,o)
        output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
        all_decoder_outputs[t] = output
         # 喂真实lable，应该喂output的结果
        if use_teacher_forcing:
            decoder_input = target_batches[t]
        else:
            # 从output中找到两个最符合的单词
            words = []
            for b in range(batch_size):
                topv, topi = output[b].data.topk(1)
                words.append(topi)
            decoder_input = get_variable(torch.LongTensor(words))
    
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),
        target_batches.transpose(0, 1).contiguous(),
        target_lengths
    )
    print (loss)
    

if __name__ == '__main__':
    data_dir = './data'
    en_file = "{}/{}".format(data_dir, "seg_en_30000.txt")
    zh_file = "{}/{}".format(data_dir, "seg_zh_30000.txt")
    pairs, input_lang, target_lang = helper.read_data(en_file, zh_file, 100)
    test_model(pairs, input_lang, target_lang)
