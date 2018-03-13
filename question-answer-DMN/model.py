#!/usr/bin/env python
# -*-coding: utf8-*-

'''
Dynamice Memory Network for Question-Answer

@author: PLM
@date: 2018-03-10
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_helper import get_variable

class DMN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, padding_idx, seqbegin_id, dropout_p=0.1):
        '''
        Args:
            vocab_size -- 词汇表大小
            embed_size -- 词嵌入维数
            hidden_size -- GRU的输出维数
            padding_idx -- pad标记的wordid
            seqbegin_id -- 句子起始的wordid
            dropout_p -- dropout比率
        '''
        super(DMN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seqbegin_id = seqbegin_id
        
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.input_gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.question_gru = nn.GRU(embed_size, hidden_size, batch_first=True)    
        self.gate = nn.Sequential(
                        nn.Linear(hidden_size * 4, hidden_size),
                        nn.Tanh(),
                        nn.Linear(hidden_size, 1),
                        nn.Sigmoid()
                    )
        self.attention_grucell = nn.GRUCell(hidden_size, hidden_size)
        self.memory_grucell = nn.GRUCell(hidden_size, hidden_size)
        self.answer_grucell = nn.GRUCell(hidden_size * 2, hidden_size)
        self.answer_fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout_p)
        
        self.init_weight()
    
    def init_hidden(self, batch_size):
        '''GRU的初始hidden。单层单向'''
        hidden = torch.zeros(1, batch_size, self.hidden_size)
        hidden = get_variable(hidden)
        return hidden
    
    def init_weight(self):
        nn.init.xavier_uniform(self.embed.state_dict()['weight'])
        components = [self.input_gru, self.question_gru, self.gate, self.attention_grucell,
                     self.memory_grucell, self.answer_grucell]
        for component in components:
            for name, param in component.state_dict().items():
                if 'weight' in name:
                    nn.init.xavier_normal(param)
        nn.init.xavier_uniform(self.answer_fc.state_dict()['weight'])
        self.answer_fc.bias.data.fill_(0)
        
    def forward(self, allfacts, allfacts_mask, questions, questions_mask, alen, n_episode=3):
        '''
        Args:
            allfacts -- [b, n_fact, flen]，输入的多个句子
            allfacts_mask -- [b, n_fact, flen]，mask=1表示是pad的，否则不是
            questions -- [b, qlen]，问题
            questions_mask -- [b, qlen]，mask=1：pad
            alen -- Answer len
            seqbegin_id -- 句子开始标记的wordid
            n_episodes -- 
        Returns:
            preds -- [b * alen,  vocab_size]，预测的句子。b*alen合在一起方便后面算交叉熵
        '''
        # 0. 计算常用的信息，batch_size，一条数据nfact条句子，每个fact长度为flen，每个问题长度为qlen
        bsize = allfacts.size(0)
        nfact = allfacts.size(1)
        flen = allfacts.size(2)
        qlen = questions.size(1)
        
        # 1. 输入模块，用RNN编码输入的句子
        # TODO 两层循环，待优化
        encoded_facts = []
        # 对每一条数据，计算facts编码
        for facts, facts_mask in zip(allfacts, allfacts_mask):
            facts_embeds = self.embed(facts)
            facts.embeds = self.dropout(facts_embeds)
            hidden = self.init_hidden(nfact)
            # 1.1 把输入(多条句子)给到GRU
            # b=nf, [nf, flen, h], [1, nf, h]
            outputs, hidden = self.input_gru(facts_embeds, hidden)
            # 1.2 每条句子真正结束时(real_len)对应的输出，作为该句子的hidden。GRU：ouput=hidden
            real_hiddens = []

            for i, o in enumerate(outputs):
                real_len = facts_mask[i].data.tolist().count(0)
                real_hiddens.append(o[real_len - 1])
            # 1.3 把所有单个fact连接起来，unsqueeze(0)是为了后面的所有batch的cat
            hiddens = torch.cat(real_hiddens).view(nfact, -1).unsqueeze(0)
            encoded_facts.append(hiddens)
        # [b, nfact, h]
        encoded_facts = torch.cat(encoded_facts)

        # 2. 问题模块，对问题使用RNN编码
        questions_embeds = self.embed(questions)
        questions_embeds = self.dropout(questions_embeds)
        hidden = self.init_hidden(bsize)
        # [b, qlen, h], [1, b, h]
        outputs, hidden = self.question_gru(questions_embeds, hidden)
        real_questions = []
        for i, o in enumerate(outputs):
            real_len = questions_mask[i].data.tolist().count(0)
            real_questions.append(o[real_len - 1])
        encoded_questions = torch.cat(real_questions).view(bsize, -1)
        
        # 3. Memory模块
        memory = encoded_questions
        for i in range(n_episode):
            # e
            e = self.init_hidden(bsize).squeeze(0)
            # [nfact, b, h]
            encoded_facts_t = encoded_facts.transpose(0, 1)
            # 根据memory, episode，计算每一时刻的e。最终的e和memory来计算新的memory
            for t in range(nfact):
                # [b, h]
                bfact = encoded_facts_t[t]
                # TODO 计算4个特征，论文是9个
                f1 = bfact * encoded_questions
                f2 = bfact * memory
                f3 = torch.abs(bfact - encoded_questions)
                f4 = torch.abs(bfact - memory)
                z = torch.cat([f1, f2, f3, f4], dim=1)
                # [b, 1] 对每个fact的注意力
                gt = self.gate(z)
                e = gt * self.attention_grucell(bfact, e) + (1 - gt) * e
            # 每一轮的e和旧memory计算新的memory
            memory = self.memory_grucell(e, memory)
        
        # 4. Answer模块
        # [b, h]
        answer_hidden = memory
        begin_tokens = get_variable(torch.LongTensor([self.seqbegin_id]*bsize))
        # [b, h]
        last_word = self.embed(begin_tokens)
        preds = []
        for i in range(alen):
            inputs = torch.cat([last_word, encoded_questions], dim=1)
            answer_hidden = self.answer_grucell(inputs, answer_hidden)
            # to vocab_size
            probs = self.answer_fc(answer_hidden)
            # [b, v]
            probs = F.log_softmax(probs.float())
            _, indics = torch.max(probs, 1)
            last_word = self.embed(indics)
            # for cross entropy
            preds.append(probs.view(bsize, 1, -1))
            #preds.append(indics.view(bsize, -1))
        #print (preds[0].data.shape)
        preds = torch.cat(preds, dim=1)
        #print (preds.data.shape)
        return preds.view(bsize * alen, -1)