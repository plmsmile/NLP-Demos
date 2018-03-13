# encoding=utf-8

'''
Skipgram模型的配置

@author PLM
@date 2018-03-09
'''
from torch.autograd import Variable

class DefaultConfig(object):
    # dataset
    book_name = 'melville-moby_dick.txt'
    num_workers = 2
    
    # model parameters
    embed_size = 128
    
    # train
    batch_size = 128
    max_epochs = 1000
    neg_nums = 10
    print_every_epoch = 30
    
    learning_rate = 0.001
    min_loss = 0.01
    
    # model save
    model_path = "./models/SkipgramNegSampling.pkl"
    
    # others
    min_count = 3
    unktag = '<UNK>'
    
    use_cuda = True
    device_id = 0
    
    def get_variable(self, x):
        x = Variable(x)
        if self.use_cuda:
            x = x.cuda(self.device_id)
        return x
