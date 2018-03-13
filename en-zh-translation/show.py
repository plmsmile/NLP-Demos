#!/usr/bin/env python
# encoding=utf-8

'''
工具类
@author PLM
@date 2017-10-16
'''

from __future__ import print_function

import time
import math
import io
import socket

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torchvision
import visdom
from PIL import Image

vis = visdom.Visdom()

#import sys  

#reload(sys)  
#sys.setdefaultencoding('utf8')

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

HOSTIP = get_host_ip()


def as_minutes(seconds):
    '''把s秒转换成x分x秒形式'''
    m = math.floor(seconds / 60)
    seconds -= m * 60
    return '%dm %ds' %(m, seconds)


def time_since(since, percent):
    '''计算还剩多少时间吧'''
    now = time.time()
    seconds = now - since
    es = seconds / (percent)
    rs = es - seconds
    return '%s (%s)' % (as_minutes(seconds), as_minutes(rs))


def show_plot_visdom(hostname=HOSTIP):
    buf = io.BytesIO()
    plt.savefig(buf)
    buf.seek(0)
    attn_win = 'attention (%s)' % hostname
    vis.image(torchvision.transforms.ToTensor()(Image.open(buf)), 
              win=attn_win, opts={'title': attn_win})


def show_attention(input_sentence, output_words, attentions, target_sentence=None, show_in_visdom=False):
    ''' 展示翻译结果与原句子的注意力分配情况
    x轴-横着-原词汇，y轴-竖着-翻译结果词汇
    Args:
        input_sentence: 输入的原句子，字符串
        target_sentence: 原翻译的句子
        output_words: 翻译的单词，列表，包含EOS
        attentions: [target_len, input_length]，[结果词汇，原词汇]
        showin_visdom: 展示在visdom中
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #print ()
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    # 把plot展示到visdom中
    if show_in_visdom:
        output_sentence = ' '.join(output_words)
        win = 'evaluted (%s)' % hostname
        text = '<p>&gt; %s</p><p>= %s</p><p>&lt; %s</p>' % (input_sentence, target_sentence, output_sentence)
        vis.text(text, win=win, opts={'title': win})
        show_plot_visdom()
    
    plt.show()
    plt.close()
