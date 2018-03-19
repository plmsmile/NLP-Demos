#!/usr/bin/env python
# -*-coding: utf8-*-

'''
模型

@author: PLM
@date: 2018-03-18
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from data_helper import log, get_variable


class SequenceEncoder(nn.Module):
    '''对一个序列使用RNN的outputs进行编码'''
    def __init__(self, input_size, hidden_size, nlayer=1, bidir=True, sum_outputs=True):
        ''' 初始化GRU
        Args:
            input_size -- 输入维数
            hidden_size -- 要编码的维数
            nlayer -- GRU的层数
            bidir -- 双向
            sum_outputs -- 双向时，对RNN的outputs结果是否做相加
        '''
        super(SequenceEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nlayer = nlayer
        self.bidir = bidir
        self.sum_outputs = sum_outputs
        self.gru = nn.GRU(input_size, hidden_size, num_layers=nlayer, batch_first=True, bidirectional=bidir)
    
    def forward(self, seqs, real_lens, hidden=None):
        ''' 对序列进行编码
        Args:
            seqs -- [batch_size, seq_len, input_size]
            real_lens -- [batch_size] 各个序列的真实长度
            hidden -- [nlayer*ndir, batch_size, encode_size]
        Returns:
            encoded_seqs -- [batch_size, seq_len, encode_size]
        '''
         # 从大到小排序
        lens, indices = torch.sort(real_lens, 0, True) 
        # pack
        packed_seqs = pack(seqs[indices], lens.data.tolist(), batch_first=True)
        # 使用GRU的outpus作为编码结果
        packed_seqs, hidden = self.gru(packed_seqs, hidden)
        # 把顺序换回来
        encoded_seqs, _ = unpack(packed_seqs, batch_first=True)
        _, indices = torch.sort(indices, 0, False)
        encoded_seqs = encoded_seqs[indices]
        # bidir 简单两个output相加
        if self.bidir and self.sum_outputs:
            encoded_seqs = encoded_seqs[:, :, :self.hidden_size] + encoded_seqs[:, :, self.hidden_size:]
        return encoded_seqs
    
    
class CoattentionEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, padid, nlayer=1, dorpout_p=0.3, bidir=True):
        super(CoattentionEncoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.nlayer = nlayer
        self.bidir = bidir
        
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=padid)
        # 对问题和句子的embed进行编码，得到各自的编码矩阵E
        self.seq_encoder1 = SequenceEncoder(embed_size, hidden_size, nlayer, bidir)
        # 对coattention的中间结果进行编码
        self.seq_encoder2 = SequenceEncoder(hidden_size, hidden_size, nlayer, bidir)
        # 对最终所有的doc的各种编码concat结果进行编码，作为Encoder的最终输出
        self.seq_encoder3 = SequenceEncoder(6 * hidden_size, hidden_size, nlayer, bidir, False)
        # question的tanh线性层
        self.question_linear = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh()) 
        # 对问题和句子的summary再编码，得到新的embed
        self.summary_gru = nn.GRU(hidden_size, hidden_size, num_layers=nlayer,
                                 batch_first=True, bidirectional=bidir)
        self.dropout = nn.Dropout(dorpout_p)
    
    def init_hidden(self, bsize, bidir=None):
        '''init一个GRU的hidden state. 实际上bidir和nlayer都在前面配置过了. 
        所有GRU的层数、方向都、hidden_size都是一样的
        Args:
            bsize -- batch_size
            bidir -- 是否是双向GRU]
        Returns:
            hidden -- 初始化为0的
        '''
        bidir = self.bidir if bidir is None else bidir
        ndir = 1 if bidir is False else 2
        hidden = torch.zeros(ndir*self.nlayer, bsize, self.hidden_size)
        hidden = get_variable(hidden)
        return hidden
    
    def forward(self, documents, questions, dlens, qlens):
        ''' 对文档和问题进行编码
        Args:
            documents -- [b, dlen]
            questions -- [b, qlen]
            dlens -- [b], 每个document的真实长度
            qlens -- [b], 每个question的真实长度
        Returns:
            U -- [b, m, 2h]
        '''
        # 0. 基本信息
        bsize = dlens.size(0)
        dsize = documents.size(1)
        qsize = questions.size(1)
        #log("b={}, d={}, q={}".format(bsize, dsize, qsize))  
        
        # 1. 对document和question使用GRU进行编码
        # document encoding. [b,d,h]
        documents = self.dropout(self.embed(documents))
        dembed1 = self.seq_encoder1(documents, dlens)
        # question encoding. [b,q,h]
        questions = self.dropout(self.embed(questions))
        qembed1 = self.seq_encoder1(questions, qlens)
        qembed1 = self.question_linear(qembed1)
        
        # 2. 两层的Coattention
        dsummary1, qsummary1, dco_context1 = self.coattention(dembed1, qembed1)
        dembed2 = self.seq_encoder2(dsummary1, dlens)
        qembed2 = self.seq_encoder2(qsummary1, qlens)
        dsummary2, qummary2, dco_context2 = self.coattention(dembed2, qembed2)
        
        # 3. concat所有的d
        alld = torch.cat([dembed1, dembed2, dsummary1, dsummary2, dco_context1, dco_context2], -1)
        U = self.seq_encoder3(alld, dlens)
        return U
     
    def coattention(self, dembed, qembed):
        ''' 计算coattention，参见博客
        Args:
            dembed -- [b, m, h]
            qembed -- [b, n, h]
        Rerturns:
            dsummary -- [b, m, h]
            qsummary -- [b, n, h]
            dco_context -- [b, m, h]
        '''
        # 关联矩阵[b,m,n]
        L = torch.bmm(dembed, qembed.transpose(1, 2))
        # 权值向量 [b,m,n], [b,n,m]
        que_atten_weights = F.softmax(L, -1)
        doc_atten_weights = F.softmax(L.transpose(1, 2), -1)
        # Summary(普通的Context)
        dsummary = que_atten_weights.bmm(qembed)
        qsummary = doc_atten_weights.bmm(dembed)
        # Doc Coattention Context 
        dco_context = que_atten_weights.bmm(qsummary)
        return dsummary, qsummary, dco_context
    

class Maxout(nn.Module):
    '''Maxout激活层，类似训练多个，然后选择一个'''
    def __init__(self, input_size, output_size, pool_size):
        ''' 初始化线性层
        Args:
            input_size -- 输入维数
            output_size -- 输出维数
            pool_size -- 多个output，从中选择一个最好的
        '''
        super(Maxout, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.pool_size = pool_size
        self.trans = nn.Linear(input_size, output_size * pool_size)
    
    def forward(self, inputs):
        ''' 全连接转换，从pool个outputs中选择最大的作为output. 实际上前面输入多维也可以。[m, b, input_size]
        Args:
            inputs -- [b, input_size]
        Returns:
            value -- [b, output_size]
        '''
        # 全连接层转换计算 [b, o*p]
        out = self.trans(inputs)
        # 计算新的shape [b, o, p]
        shape = list(inputs.size())
        shape[-1] = self.output_size
        shape.append(self.pool_size)
        # view为新的[b,o,p]，再选择最大的pool作为返回值
        value, indices = out.view(shape).max(-1)
        return value
        

class HMN(nn.Module):
    '''Highway Maxout Network'''
    def __init__(self, hidden_size, pool_size=8):
        super(HMN, self).__init__()
        self.hidden_size = hidden_size
        
        # 计算r = mlp(ue,us,hi)
        self.mlp = nn.Linear(5 * hidden_size, hidden_size)
        # 合并u_r
        self.calm1 = Maxout(3 * hidden_size, hidden_size, pool_size)
        self.calm2 = Maxout(hidden_size, hidden_size, pool_size)
        # 把语义映射到1维，得到所有ut的得分. 残差连接m1和m2
        self.calm3 = Maxout(hidden_size*2, 1, pool_size)
    
    def forward(self, u, hi, preus, preue):
        ''' 计算所有文档单词在第i轮，与hi,preus,preue的得分
        Args:
            u --- [b, m, 2h]. U中m个单词的语义表示
            hi -- [b, h]. 第i次迭代时刻需要的hidden state
            preus -- [b, 2h]. pre U start，上一时刻预测的start，在U中的语义向量
            preue -- [b, 2h]. pre U end，上一时刻预测的end，在U中的语义向量
        Returns:
            scores -- [b, m]. U中m个单词在当前情况下的得分。start_hmn就是start得分，end_hmn就是end得分
        '''
        doclen = u.size(1)
        # [b, 5h] us,ue,hi -- r=[b,h]
        rinput = torch.cat([preus, preue, hi], -1)
        r = self.mlp(rinput).unsqueeze(1)
        # [b,1,h] -- [b, m, h]
        r = torch.cat([r] * doclen, 1)
        # [b, m, 3h]
        r_u = torch.cat([u, r], -1)
        m1 = self.calm1(r_u)
        m2 = self.calm2(m1)
        # [b, m, 1]
        m3 = self.calm3(torch.cat([m1, m2], -1))
        # [b, m]
        scores = m3.squeeze(-1)
        return scores


class DynamicDecoder(nn.Module):
    def __init__(self, hidden_size, pool_size=5, dropout_p =0.3, max_iter=4):
        super(DynamicDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.max_iter = max_iter
        self.dropout = nn.Dropout(dropout_p)
        # 输入ut hi ue us. 计算ut的得分
        self.hmn_start = HMN(hidden_size, pool_size)
        self.hmn_end = HMN(hidden_size, pool_size)
        self.grucell = nn.GRUCell(4 * hidden_size, hidden_size)
    
    def init_hidden(self, batch_size):
        hidden = torch.zeros(batch_size, self.hidden_size)
        return get_variable(hidden)
       
    def get_ustart_uend(self, info, startidx, endidx):
        '''循环去寻找U中对应startidx和endidx的向量
        Args:
            info -- [b, m, 2h], CoattentionEncoder对D的编码矩阵U
            startidx -- [b]，答案的起始id
            endidx -- [b]，答案的结束id
        Returns:
            ustart -- [b, 2h]，开始单词的语义向量
            uend -- [b, 2h]，结束单词的语义向量
        '''
        bsize = info.size(0)
        us = []
        ue = []
        # 循环去找
        for j in range(bsize):
            doc = info[j,:,:]
            us_j = torch.index_select(doc, 0, startidx[j])
            ue_j = torch.index_select(doc, 0, startidx[j])
            us.append(us_j)
            ue.append(ue_j)
        # [b,2h]
        us = torch.cat(us, 0)
        ue = torch.cat(ue, 0)
        return us, ue
    
    def forward(self, info):
        ''' 给文档和问题的综合信息info，预测答案
        Args:
            info -- [b, m, 2h]. CoattentionEncoder对D的编码矩阵U
        Returns:
            start -- [b]答案在D中的起始位置 
            end -- [b]答案在D中的结束位置
            all_scores -- [[start_scores, end_scores]].每一次迭代所有单词的得分. 
                          scores -- [b,m]. 因为是累积交叉熵，每一次迭代结果都要加入计算
        '''
        bsize = info.size(0)
        doclen = info.size(1)
        hidden = self.init_hidden(bsize)
        # 默认起始地址初始化为 [0,1]
        start = get_variable(torch.LongTensor([0]*bsize))
        end = get_variable(torch.LongTensor([1]*bsize))
        # 每一轮的start_scores, end_scores
        all_scores = []
        for i in range(self.max_iter): 
            # 1. [b,2h] 根据s和e的索引，从u中选择对应的start和end向量
            ustart, uend = self.get_ustart_uend(info, start, end)
            # 2. [b, m] 在当前条件下，计算u中每个词作为start、end的得分
            start_scores = self.hmn_start(info, hidden, ustart, uend)
            end_scores = self.hmn_end(info, hidden, ustart, uend)
            all_scores.append([start_scores, end_scores])
            # 3. [b] 选择得分最大的作为新的start和end 
            _, new_start = start_scores.max(-1)
            _, new_end = end_scores.max(-1)
            # 4. 比较和上一轮的结果，完全相同则停止
            eq_start = torch.eq(start, new_start)
            eq_end = torch.eq(end, new_end)
            eq_start = torch.sum(eq_start).data.tolist()[0]
            eq_end = torch.sum(eq_end).data.tolist()[0]
            if (eq_start == bsize and eq_end == bsize):
                #log("new_start--start, new_end--end, equal, break")
                break
            # 5. 更新hidden和start和end
            ustart, uend = self.get_ustart_uend(info, new_start, new_end)
            hidden = self.grucell(torch.cat([ustart, uend], 1), hidden)
            start, end = new_start, new_end
        return start, end, all_scores