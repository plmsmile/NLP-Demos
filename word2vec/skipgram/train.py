# encoding=utf-8


'''
训练模型

@author PLM
@date 2018-03-09
'''
import nltk
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

from torch.utils.data import DataLoader

from data_helper import TextHelper, TextDataset
from model import SkipgramNegSampling
from config import DefaultConfig

def train(opt, th):
    '''训练
    Args:
        opt -- 配置，DefaultConfig实例
        th -- TextHelper实例
    '''
    corpus = list(nltk.corpus.gutenberg.sents(opt.book_name))[:10]
    train_set = TextDataset(corpus, th)
    train_data = DataLoader(train_set, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    
    model = SkipgramNegSampling(th.vocab_size, opt.embed_size)
    if opt.use_cuda:
        model = model.cuda(opt.device_id)
    
    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    
    losses = []
    for e in range(opt.max_epochs):
        for context_words, target_words in train_data:
            neg_words = th.negative_sample(target_words, opt.neg_nums)
            #print (type(context_words), context_words.shape)
            context_words = opt.get_variable(context_words)
            target_words = opt.get_variable(target_words)
            neg_words = opt.get_variable(torch.LongTensor(neg_words))
            
            loss = model(context_words, target_words, neg_words)
            
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


def cal_similar_words(model, word, th, opt, k=10):
    '''
    Args:
        model -- 模型
        word -- 原单词
        th -- TextHelper
        opt -- DefultConfig
        k -- 相似的k个单词
    Returns:
        前十个相似单词和相似度
    '''
    wordid = th.word2index(word)
    wordid = opt.get_variable(torch.LongTensor([wordid]))
    source_vector = model.get_vector(wordid)
    #print (wordid, source_vector)
    similarities = []
    for i, vo in enumerate(th.vocab):
        if vo == word:
            continue
        targetid = th.word2index(vo)
        targetid = opt.get_variable(torch.LongTensor([targetid]))
        target_vector = model.get_vector(targetid)
        #print (target_vector)
        cosine_sim = F.cosine_similarity(source_vector, target_vector).data.tolist()[0]
        similarities.append([vo, cosine_sim])
    return sorted(similarities, key = lambda x: -x[1])[:k]


def test_similarity(word, th, opt, model):
    similarities = cal_similar_words(model, word, th, opt)
    print (word)
    for sim in similarities[:10]:
        print (sim)


if __name__ == '__main__':   
    opt = DefaultConfig()
    corpus = list(nltk.corpus.gutenberg.sents('melville-moby_dick.txt'))
    th = TextHelper(corpus, opt.min_count, opt.unktag)
    del corpus
    # train(opt, th)
    model = get_model(opt.model_path)
    word = random.choice(th.vocab)
    test_similarity(word, th, opt, model)
