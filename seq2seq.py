import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
       self.name = name
       self.word2index = {}
       self.word2count = {}
       self.index2word = {0: "SOS", 1: "EOS"}
       self.n_words = 2

    def index_words(self, sentence):
       for word in sentence.split(' '):
       self.index_word(word)

    def index_word(self, word):
       if word not in self.word2index:
          self.word2index[word] = self.n_words
          self.word2count[word] = 1
          self.index2word[self.n_words] = word
          self.n_words += 1
      else:
          self.word2count[word] += 1

   def unicode_to_ascii(s):
      return ''.join(
         c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
      )

   # 大小转小写，trim，去掉非字母。
   def normalize_string(s):
      s = unicode_to_ascii(s.lower().strip())
      s = re.sub(r"([.!?])", r" \1", s)
      s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
      return s

def read_clangs(lang1,lang2,reverse=False):
    print("read lines")

    lines = open('./%s-%s.txt'%(lang1,lang2)).read().strip(.split('\n'))
    paris = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    if reverse:
        paris = [list(reversed(p)) for p in paris]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang,output_lang,pairs

#过滤句子
MAX_LENGTH = 10
good_prefixes=(
    " i am "," i m"
)
def filter_pari(p):
    return len(p[0].split(' '))<MAX_LENGTH and len(p[1].split(" ")) < MAX_LENGTH and p[1].startwith(good_prefixes)
def filter_pairs(pairs):
    return [pair for pair in pairs if filter(pair)]
#将句子变成向量形式，文本归一化，通过长度和内容过滤
def prepare_data(lang1_name,lang2_name,reverse=False):
    input_lang,output_lang,pairs = read_langs(lang1_name,lang2_name,reverse)
    print("read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("indexing word..")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang,output_lang,pairs
input_lang,output_lang,pairs = prepare_data('eng','fra',True)
print(random.choice(pairs))

#将每个词替换成对应的index
def indexes_from_sentence(lang,sentence):
    return [lang.word2index[word] for word in sentence.split(' ') ]

def variable_from_sentence(lang,sentence):
    indexes = indexes_from_sentence(lang,sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1,1))
    if USE_CUDA:var = var.cuda()
    return var
def variables_from_pair(pair):
    input_variable = variable_from_sentence(input_lang,pair[0])
    target_variable = variable_from)sentence(output_lang,pair[1])
    return (input_variable,target_variable)
#定义模型
class EncoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size,n_layers=1):
        super (EncoderRNN,self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embeding = nn.embedding(inut_size,hidden_size)
        self.gru = nn.GRU(hidden_size,hidden_size,n_layers)

    def forward(self,word_inputs,hidden):
        #一次处理完一个输入的所有词
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len,1,-1)
        output,hidden =  self.gru(embedded,hidden)
        return output,hidden
    def  init_hidden(self):
        hidden  = Variable(torch.zeros(self.n_layers,1,self.hidden_size))
        if USE_CUDA:hidden = hidden.cuda()
        return hidden

class attentionDecoderRNN(nn.Module):
    def __init__(self,hidden_size,output_size,n_layers =1,dropout_p=0.1):
        super(attentionDecoderRNN,self).__init__()
        #定义参数
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        #定义网络
        self.embedding = nn.Embedding(output_size,hidden_size)
        self.dropout_p = nn.Dropout(dropout_p)
        self.attn = GeneralAttn(hidden_size)
        self.gru = nn.GRU(hidden_size*2,hidden_size,n_layers,dropout_p=dropout_p)
        self.out = nn.Linear(hidden_size,output_size)
    def forward(self, word_input,last_hidden,encoder_outputs):
        #每次运行一个时间步
        word_embedded  = self.embedding(word_input).view(1,1,-1)
        word_embedded = self.dropout(word_embedded)
        #计算attention weights
        attn_embedded = self.attn(last_hidden[-1],encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))

        rnn_input = torch.cat((word_embedded,context),2)
        output,hidden = self.gru(rnn_input,last_hidden)
        output = output.squeeze(0)
        output = F.log_softmax(self.out(torch.cat((output,context),1)),1)

        return output,hidden,attn_weights

    '''首先用encoder对输入句子进行编码，得到每个时刻的输出和最后一个时刻的隐状态，最后一个
    隐状态会作为decoder隐状态的初始值，用特殊的标记最为decoder的第一个输入'''

teacher_forcing_ratio = 0.5
clip = 5.0
def train(input_variable,target_variable,encoder,decoder,encoder_optimizer,decoder_optimizer)
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss  = 0
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_hidden  = encoder.init_hidden()
    encoder_outputs,encoder_hidden = encoder(input_variable,encoder_hidden)

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_context = Variable(torch.zeros(1,decoder.hidden_size))
    decoder_hidden = encoder_hidden
    # 随机选择是否Teacher Forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:

        # Teacher forcing：使用真实的输出作为下一个时刻的输入
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # 下一个时刻的输入来自target

    else:
        # 不使用 teacher forcing：使用decoder的预测作为下一个时刻的输入
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context,
                                                                                         decoder_hidden,                                                                               encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])

            # 选择最可能的词
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))  # 下个时刻的输入
            if USE_CUDA: decoder_input = decoder_input.cuda()

            # 如果decoder输出EOS_token，那么提前结束
            if ni == EOS_token: break
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(),clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(),clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout_p = 0.05 #初始化模型
encoder = EncoderRNN(input_lang.n_words,hidden_size,n_layers)
decoder = AttnDecoderRNN(attn_model,hidden_size,output_lang.n_words,n_layers,dropout_p=dropout_p)
if USE_CUDA():
    encoder.cuda()
    decoder.cuda()

learning_rate = 0.0001
encoder_optimizer = optim.Adam(encoder.parameters(),lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(),lr=learning_rate)
criterion = nn.NLLLoss

n_epochs = 5000
plot_every = 200
print_every = 1000

start = time.time()
plot_losses = []
print_loss_total = 0
plot_loss_total = 0

for epoch in range(1,n_epochs + 1):
    #得到一个训练句子对
    training_pari = variables_from_pair(random.choice(pair))
    input_variable = training_pair[0]
    target_variable = training_pair[1]
    loss = train(input_variable,target_variable,encoder,decoder,
                 encoder_optimizer,decoder_optimizer,criterion)
    print_loss_total+= loss
    plot_loss_total +=loss

    if epoch == 0:
        continue
        if epoch % print_every ==0
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = '%s(%d %d%%) % .4f' %(time_since(start,epoch/n_epochs),epoch,epoch / n_epochs*100,print_loss_avg)
            print(print_summary)
        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot.losses.appen(plot_loss_avg)
            plot_loss_total = 0

import matplotlib.pyplot as plt
import matplotlib.tocker as ticker
import numpy as np
%matplotlib inline
def show_plot(points):
    plt.figure()
    fig,ax = plt.subplots()
    loc = ticker.MutipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
show_plot(plot_losses)

def evaluate(sentence,max_length=MAX_LENGTH):
    input_variable = variable_from_sentence(input_lang,sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.init_hidden()
    encoder_outputs,encoder_hidden = encoder(input_variable,encoder_hidden)
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_context = Variable(torch.zeros(1,decoder.hidden_size))
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()
    decoder_hidden = encoder_hidden
    decoded_words = []
    decoder_attentions = torch.zeros(max_length,max_length)
    for di in range(max_length):
        decoder_output,decoder_context,decoder_hidden,decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        decoder_attentions[di,:decoder_attention.size(2)]+=decoder_attention.squeeze(0).squeeze(0).cpu().data
        topv,topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
            decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()
        return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]

    def evaluate_randomly():
        pair = random.choice(pairs)
        output_words,decoder_attn = evaluate(pair[0])
        output_sentence = ' '.join(output_words)

        print('>',pair[0])
        print('=',pair[1])
        print('<',output_sentence)
        print('')
    evaluate_randomly()

    #注意力的可视化
    def show_attention(input_sentence,output_words,attentions):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(),cmap='bone')
        fig.colorbar(cax)
        ax.xaxis.set_major_locator(ticker.MutipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MutipleLocator(1))
        plt.show()
        plt.close()

    def evaluate_and_show_attention(input_sentence):
        output_words,attentions = evaluate(input_sentence)
        print('input',input_sentence)
        print('output',' '.join(output_words))
        show_attention(input_sentence,output_words,attentions)

#数据处理
PAD_token = 0
SOS_token =1
EOS_token = 2
class Lang:

    def __init__(self,name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"PAD",1:"SOS",2:"EOS"}
        self.n_words = 3
    def seg_words(self,sentence):
        return sentence.split(' ')
    def index_words(self,sentence):
        words=self.seg_words(sentence)
        for word in words:
            self.index_word(word)


def index_words(self,word):
    if word not in self.word2index:
        self.word2index[word] = self.n_words
        self.word2count[word] = 1
        self.index2word[self.n_words] = word
        self.n_words += 1
    else:
        self.word2count[word] += 1

def trim(self,min_count):
    if self.trimmed:
        return
    self.trimmed = True

    keep_words = []
    for k,v in self.word2count.items():
        if v >=min_count:
            keep_words.append(k)
    print('keep_words %s /%s = %.4f'%(len(keep_words),len(self.word2index),len(keep_words) / len(self.word2index)))
    self.word2index = {}
    self.word2count = {}
    self.index2word = {0:"PAD",1:"SOS",2:"EOS"}
    self.n_words = 3
    for word in keep_words:
        self.index2word(word)

MIN_LENGTH = 3
MAX_LENGTH = 25
def filter_pairs(pairs):
    filter_pairs = []
    for pair in pairs:
        if len(pair[0]) >= MIN_LENGTH and len(pair[0]) <= MAX_LENGTH and len(pair[1]) >=MIN_LENGTH  and len(pair[1]) <= MAX_LENGTH:
            filter_pairs.append(pair)
    return filter_pairs
def pad_seq(seq,max_length):
    seq +=[PAD_token for i in range(max_length-len(seq))]
    return seq


