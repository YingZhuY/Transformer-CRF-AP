import argparse
import pickle
import collections
import logging
import math,copy
import os,sys,time
import errno
import random
from sys import maxsize
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
import fastNLP
from fastNLP.modules.encoder.transformer import TransformerEncoder
from fastNLP.modules.decoder.crf import ConditionalRandomField
from fastNLP import Const
from fastNLP import DataSetIter

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        #print(scores.size(),mask.size())
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)
        
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
        
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
        
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        #print(x.size(),mask.size())
        "Pass the input (and mask) through each layer in turn."
        mask=mask.byte().unsqueeze(-2)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
        
def make_encoder(N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    return Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        lr = self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        #if step>self.warmup: lr = max(1e-4,lr)
        return lr

def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0, betas=(0.9, 0.98), eps=1e-9))


def make_sure_path_exists(path):
    if len(path)==0: return 
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def is_dataset_tag(word):
    return len(word) > 2 and word[0] == '<' and word[-1] == '>'


def to_tag_strings(i2ts, tag_mapping, pos_separate_col=True):
    senlen = len(tag_mapping)
    key_value_strs = []

    for j in range(senlen):
        val = i2ts[tag_mapping[j]]
        pos_str = val
        key_value_strs.append(pos_str)
    return key_value_strs


def bmes_to_words(chars, tags):
    result = []
    if len(chars) == 0:
        return result
    word = chars[0]

    for c, t in zip(chars[1:], tags[1:]):
        if t.upper() == 'B' or t.upper() == 'S':
            result.append(word)
            word = ''
        word += c
    if len(word) != 0:
        result.append(word)

    return result


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class Embedding(nn.Module):
    def __init__(self,task_size, d_model, word_embedding=None, bi_embedding=None, word_size=None, freeze=True):
        super(Embedding, self).__init__()
        self.task_size=task_size        
        self.embed_dim = 0        
        
        self.task_embed = nn.Embedding(task_size,d_model)
        """
        if freeze:
            self.task_embed.weight.requires_grad = False
        """
        if word_embedding is not None:
            self.uni_embed = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding), freeze=freeze)
            self.embed_dim+=word_embedding.shape[1]
        else:
            if bigram_embedding is not None:
                self.embed_dim+=bi_embedding.shape[1]
            else: self.embed_dim=d_model
            assert word_size is not None
            self.uni_embed = nn.Embedding(word_size,self.embed_dim)
            
        if bi_embedding is not None:    
            self.bi_embed = nn.Embedding.from_pretrained(torch.FloatTensor(bi_embedding), freeze=freeze)
            self.embed_dim += bi_embedding.shape[1]*2
            
        print("Trans Freeze",freeze,self.embed_dim)
        
        if d_model!=self.embed_dim:
            self.F=nn.Linear(self.embed_dim,d_model)
        else :
            self.F=None
            
        self.d_model = d_model

    def forward(self, task, uni, bi1=None, bi2=None):
        #print(task,uni.size(),bi1.size(),bi2.size())
        #print(bi1,bi2)
        #assert False
        y_task=self.task_embed(task[:,0:1])
        y=self.uni_embed(uni[:,1:])
        if bi1 is not None:
            assert self.bi_embed is not None
            
            y=torch.cat([y,self.bi_embed(bi1),self.bi_embed(bi2)],dim=-1)
            #y2=self.bi_embed(bi)
            #y=torch.cat([y,y2[:,:-1,:],y2[:,1:,:]],dim=-1)
            
        #y=torch.cat([y_task,y],dim=1)
        if self.F is not None:
            y=self.F(y)
        y=torch.cat([y_task,y],dim=1)
        return y * math.sqrt(self.d_model)

def seq_len_to_mask(seq_len,max_len=None):
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        if max_len is None:
            max_len = int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)
    
    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        if max_len is None:
            max_len = seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")
    
    return mask        
    
class CWSModel(nn.Module):
    def __init__(self, encoder, src_embed, position, d_model, tag_size, crf=None):
        super(CWSModel, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.pos=copy.deepcopy(position)
        self.proj = nn.Linear(d_model, tag_size)
        self.tag_size=tag_size
        if crf is None:
            self.crf=None
            self.loss_f=nn.CrossEntropyLoss(size_average=False)
        else:
            print("crf")
            trans=fastNLP.modules.decoder.crf.allowed_transitions(crf,encoding_type='bmes')
            self.crf=ConditionalRandomField(tag_size,allowed_transitions=trans)
        #self.norm=nn.LayerNorm(d_model)

    def forward(self, task, uni, seq_len, bi1=None, bi2=None, tags=None):        
        mask=seq_len_to_mask(seq_len,uni.size(1))
        out=self.src_embed(task,uni,bi1,bi2)
        out=self.pos(out)
        #out=self.norm(out)
        #print(uni.size(),out.size(),mask.size(),seq_len)
        out=self.proj(self.encoder(out, mask.float()))
        
        if self.crf is not None:
            if tags is not None:
                out=self.crf(out, tags, mask)
                return {"loss":out}
            else:
                out,_ =self.crf.viterbi_decode(out, mask)
                return {"pred":out}
        else:
            if tags is not None:
                num=out.size(0)
                loss = self.loss_f(torch.masked_select(out,mask.unsqueeze(-1).expand_as(out)).contiguous().view(-1,self.tag_size), torch.masked_select(tags,mask))
                return {"loss":loss/num}
            else:
                out=torch.argmax(out,dim=-1)  
                return {"pred":out}


def make_CWS(N=6, d_model=256, d_ff=1024, h=4, dropout=0.2, tag_size=4, task_size=8, bigram_embedding=None, word_embedding=None, word_size=None, crf=None,freeze=True):
    c = copy.deepcopy
    #encoder=TransformerEncoder(num_layers=N,model_size=d_model,inner_size=d_ff,key_size=d_model//h,value_size=d_model//h,num_head=h,dropout=dropout)
    encoder=make_encoder(N=N,d_model=d_model,h=h,dropout=dropout,d_ff=d_ff)
    
    position = PositionalEncoding(d_model, dropout)
    embed=Embedding(task_size, d_model, word_embedding, bigram_embedding,word_size,freeze)
    model=CWSModel(encoder, embed, position, d_model, tag_size, crf=crf)
    
    for name,p in model.named_parameters():
        if p.dim() > 1 and p.requires_grad==True:
            nn.init.xavier_uniform(p)
    return model


NONE_TAG = "<NONE>"
START_TAG = "<sos>"
END_TAG = "<eos>"

DEFAULT_WORD_EMBEDDING_SIZE = 100

# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--word-embeddings", dest="word_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--bigram-embeddings", dest="bigram_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--crf", dest="crf", action="store_true", help="whether to use CRF")                    
parser.add_argument("--devi", default="0", dest="devi", help="gpu to use")
parser.add_argument("--step", default=0, dest="step", type=int,help="step")
parser.add_argument("--num-epochs", default=80, dest="num_epochs", type=int,
                    help="Number of epochs through training set")
parser.add_argument("--flex", default=-1, dest="flex", type=int,
                    help="Number of epochs through training set after freezing the pretrained embeddings")
parser.add_argument("--batch-size", default=256, dest="batch_size", type=int,
                    help="Minibatch size of training set")
parser.add_argument("--d_model", default=256, dest="d_model", type=int, help="d_model of transformer encoder")
parser.add_argument("--d_ff", default=1024, dest="d_ff", type=int, help="d_ff for FFN")
parser.add_argument("--N", default=6, dest="N", type=int, help="Number of layers")
parser.add_argument("--h", default=4, dest="h", type=int, help="Number of head")
parser.add_argument("--factor", default=2, dest="factor", type=float, help="factor for learning rate")
parser.add_argument("--dropout", default=0.2, dest="dropout", type=float,
                    help="Amount of dropout(not keep rate, but drop rate) to apply to embeddings part of graph")
parser.add_argument("--log-dir", default="result", dest="log_dir",
                    help="Directory where to write logs / saved models")
parser.add_argument("--task-name", default=time.strftime("%Y-%m-%d-%H-%M-%S"), dest="task_name",
                    help="Name for this task, use a comprehensive one")
parser.add_argument("--no-model", dest="no_model", action="store_true", help="Don't save model")
parser.add_argument("--always-model", dest="always_model", action="store_true",
                    help="Always save the model after every epoch")
parser.add_argument("--old-model", dest="old_model", help="Path to old model for incremental training")
parser.add_argument("--freeze", dest="freeze", action="store_true", help="freeze pretrained embeddings")

parser.add_argument("--python-seed", dest="python_seed", type=int, default=random.randrange(maxsize),
                    help="Random seed of Python and NumPy")
parser.add_argument("--test", dest="test", action="store_true", help="Test mode")

options = parser.parse_args()
task_name = options.task_name
root_dir = "{}/{}".format(options.log_dir, task_name)
make_sure_path_exists(root_dir)

devices=[int(x) for x in options.devi]
device = torch.device("cuda:{}".format(devices[0]))  

def init_logger():
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    log_formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    file_handler = logging.FileHandler("{0}/info.log".format(root_dir), mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
logger = init_logger()
# Log some stuff about this run
logger.info(' '.join(sys.argv))
logger.info('')
logger.info(options)

random.seed(options.python_seed)
np.random.seed(options.python_seed % (2 ** 32 - 1))
logger.info('Python random seed: {}'.format(options.python_seed))

# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = pickle.load(open(options.dataset, "rb"))
test_set=dataset["test_set"]
uni_vocab=dataset["uni_vocab"]
bi_vocab=dataset["bi_vocab"]

print(len(test_set))

# ===-----------------------------------------------------------------------===
# Build model and trainer
# ===-----------------------------------------------------------------------===

if options.word_embeddings is None:
    init_embedding=None
else:
    print("Load:",options.word_embeddings)
    init_embedding=fastNLP.io.embed_loader.EmbedLoader.load_with_vocab(options.word_embeddings, uni_vocab, normalize=False)
    
bigram_embedding = None
if options.bigram_embeddings:
    if options.bigram_embeddings == 'merged':
        logging.info('calculate bigram embeddings from unigram embeddings')
        bigram_embedding=np.random.randn(len(bi_vocab), init_embedding.shape[-1]).astype('float32')      
        for token, i in bi_vocab:
            if token.startswith('<') and token.endswith('>'): continue
            if token.endswith('>'):
                x,y=uni_vocab[token[0]], uni_vocab[token[1:]]
            else: 
                x,y=uni_vocab[token[:-1]], uni_vocab[token[-1]]
            if x==uni_vocab['<unk>']:
                x=uni_vocab['<pad>']
            if y==uni_vocab['<unk>']:
                y=uni_vocab['<pad>']
            bigram_embedding[i]=(init_embedding[x]+init_embedding[y])/2
    else:    
        print("Load:",options.bigram_embeddings)
        bigram_embedding=fastNLP.io.embed_loader.EmbedLoader.load_with_vocab(options.bigram_embeddings, bi_vocab, normalize=False)
        
# build model and optimizer    
i2t={0: 's', 1: 'b', 2: 'e', 3: 'm'}
if options.crf:
    print("use crf:",i2t)

freeze=True if options.freeze else False
model = make_CWS(d_model=options.d_model, N=options.N, h=options.h, d_ff=options.d_ff,dropout=options.dropout,word_embedding=init_embedding,bigram_embedding=bigram_embedding,tag_size=4,task_size=8,crf=i2t,freeze=freeze)

if True:  
    print("multi:",devices)
    model=nn.DataParallel(model,device_ids=devices)    

model=model.to(device)
    
optimizer = NoamOpt(options.d_model, options.factor, 4000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
optimizer._step=options.step

i2t=['s', 'b', 'e', 'm']
i2task=['<as>', '<msr>', '<pku>', '<ncc>', '<cityu>', '<ckip>', '<ctb>', '<sxu>']
test_set.set_input("ori_words")

word_dic = pickle.load(open("dict.pkl","rb"))
    
def tester(model,test_batch,write_out=True):
    res=[]
    split_or_not=[]
    context_list=[]
    model.eval()
    for batch_x in test_batch:
        batch_x=batch_x[0]
        with torch.no_grad():
            if bigram_embedding is not None:
                out=model(batch_x["task"],batch_x["uni"],batch_x["seq_len"],batch_x["bi1"],batch_x["bi2"])
            else: out = model(batch_x["task"],batch_x["uni"],batch_x["seq_len"])
        out=out["pred"]
        #print(out)
        num=out.size(0)
        out=out.detach().cpu().numpy()
        for i in range(num):
            length=int(batch_x["seq_len"][i])
            
            out_tags=out[i,1:length].tolist()
            sentence = batch_x["ori_words"][i]
            dataset_name = sentence[0]
            sentence=sentence[1:]
            context_list.append(sentence)
            #print(out_tags)
            assert is_dataset_tag(dataset_name)
            assert len(out_tags)==len(sentence)

            if write_out==True:
                obs_strings = to_tag_strings(i2t, out_tags)
                word_list = bmes_to_words(sentence, obs_strings)

                s_list=[]
                for i in word_list:
                    s_list.extend((len(i)-1)*[0])
                    s_list.append(1)
                split_or_not.append(s_list[:-1])
                
                raw_string=' '.join(word_list)
                res.append(raw_string)
    return res,split_or_not,context_list


model.load_state_dict(torch.load(options.old_model,map_location="cuda:0"))
        
for name,para in model.named_parameters():
    if name.find("task_embed")!=-1:
        tm=para.detach().cpu().numpy()
        print(tm.shape)
        np.save("{}/task.npy".format(root_dir),tm)                    
        break
        
test_batch=DataSetIter(test_set,options.batch_size)
res,split_or_not,context_list=tester(model,test_batch,True)

to_remain = []
to_del = []
candis=np.load('sentenceNum.npy',allow_pickle=True).item()
split_info = {}
for candi in candis.keys():
    sen_nums = candis[candi]
    count_left, count_mid, count_right = 0,0,0
    count_total=len(sen_nums)
    for sen_num in sen_nums:
        context = context_list[sen_num]
        split_value = split_or_not[sen_num]
        index=context.find(candi)
        assert index != -1
        count_left += 1 if index==0 else split_value[index-1]
        count_right += 1 if index+len(candi)==len(context) else split_value[index-1+len(candi)]
        count_mid += 1 if 1 in split_value[index:index+len(candi)-2] else 0
    #print(count_left, count_mid, count_right, count_total)
    split_left = count_left / count_total
    split_mid = count_mid / count_total
    split_right = count_right / count_total
    split_info[candi]=(count_left, count_mid, count_right, count_total, split_left, split_mid, split_right)
    #print(split_left, split_mid, split_right)
    #print()

    ### Here the filter law
    if split_left>0.8 and split_right>0.8 and split_mid<=0.5:
        to_remain.append(candi)
    else:
        to_del.append(candi)
#print(to_remain)
#print(to_del)

# split_info 写入 json 文件
j_word_split = json.dumps(split_info)
j_word_File = open('word_split.json','w',encoding='utf-8')
j_word_File.write(j_word_split)
j_word_File.close()

# to_remain = []
# to_del = []
# context_list = np.load('context.npy').tolist()
# candis=np.load('sentenceNum.npy',allow_pickle=True).item()
# for candi in candis.keys():
#     start = candis[candi][0]
#     end = candis[candi][1]
#     contexts = context_list[start:end]
#     split_values = split_or_not[start:end]
#     count_left, count_mid, count_right = 0,0,0
#     count_total=end-start
#     for i in range(count_total):
#         index=contexts[i].find(candi)
#         assert index != -1
#         count_left += 1 if index==0 else split_values[i][index-1]
#         count_right += 1 if index+len(candi)==len(contexts[i]) else split_values[i][index-1+len(candi)]
#         count_mid += 1 if 1 in split_values[index:index+len(candi)-2] else 0
#     split_left = count_left / count_total
#     split_mid = count_mid / count_total
#     split_right = count_right / count_total

#     if split_left>0.8 and split_right>0.8 and split_mid<=0.5:
#         to_remain.append(candi)
#     else:
#         to_del.append(candi)

Cremain = open('Cremain.txt','a',encoding='utf-8')
for w in to_remain:
    Cremain.write(w+'\n')
logger.info('len(to_remain): {}'.format(len(to_remain)))

Cdel = open('Cdel.txt','a',encoding='utf-8')
for w in to_del:
    Cdel.write(w+'\n')
logger.info('len(to_del): {}'.format(len(to_del)))

with open("{}/testout.txt".format(root_dir), 'w',encoding="utf-16") as raw_writer:
    for sent in res:
        raw_writer.write(sent)
        raw_writer.write('\n')
        


