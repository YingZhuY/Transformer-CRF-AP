from collections.abc import Iterable
from collections import Counter
from pygtrie import Trie
import pandas as pd
import argparse
import logging
import os
import time
import sys
import csv
import glob
import types
import re
import math
import json

# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
# Corpus file to use. You can choose one way from the following five: --default-csv, --txt-file, --csv-file, --txt-directory, --csv-directory
parser.add_argument("--default-csv", dest="default_csv", action="store_true", 
                    help="Default Mode. Use the giving CSV corpus to find new words.")
parser.add_argument("--txt-file", dest="txt_file", help="Path to your txt corpus.")
parser.add_argument("--csv-file", dest="csv_file", help="Path to your csv corpus.")
parser.add_argument("--txt-directory", dest="txt_directory", 
                    help="If you need to process two or more txt files, you can put them in the same directory. Give the directory here.")
parser.add_argument("--csv-directory", dest="csv_directory", 
                    help="If you need to process two or more csv files, you can put them in the same directory. Give the directory here.")

parser.add_argument("--BE-stop", dest="BE_stop", action="store_true", help="Filter with BE-stop mode.")
parser.add_argument("--wiki", dest="wiki", action="store_true", help="Filter with wiki mode.")

parser.add_argument("--min-n", default=2, dest="min_n", type=int, 
                    help="The min n of n-gram to extract. Default 2.")
parser.add_argument("--max-n", default=6, dest="max_n", type=int, 
                    help="The max n of n-gram to extract. Default 5.")
parser.add_argument("--min-freq", default=10, dest="min_freq", type=int, 
                    help="The frequency threshold. Default 10.")
parser.add_argument("--min-pmi", default=0, dest="min_pmi", type=float, 
                    help="The PMI threshold. Default 0. You can define your own min-pmi.")
parser.add_argument("--min-entropy", default=0, dest="min_entropy", type=float, 
                    help="The Entropy threshold. Default 0. You can define your own min-entropy.")

parser.add_argument("--restore-score", dest="restore_score", action="store_true", help="Restore score to a json file.")
parser.add_argument("--log-dir", default="result", dest="log_dir",
                    help="Directory where to write logs / saved results")
parser.add_argument("--task-name", default=time.strftime("%Y-%m-%d-%H-%M-%S"), dest="task_name",
                    help="Name for this task, you can change a comprehensive one. The result file will be stored in this directory.")

options = parser.parse_args()
task_name = options.task_name
task_dir = "{}/{}/{}".format(os.getcwd(), options.log_dir, task_name)

def init_logger():
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    log_formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    file_handler = logging.FileHandler("{0}/info.log".format(task_dir), mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger

logger = init_logger()      # set up logging
# log command and options about this run
logger.info(' '.join(sys.argv))
logger.info('')
logger.info(options)
logger.info('')

def union_word_freq(dic1,dic2):
    '''
    word_freq合并
    :param dic1:{'你':200,'还':2000,....}:
    :param dic2:{'你':300,'是':1000,....}:
    :return:{'你':500,'还':2000,'是':1000,....}
    '''
    keys = (dic1.keys()) | (dic2.keys())
    total = {}
    for key in keys:
        total[key] = dic1.get(key, 0) + dic2.get(key, 0)
    return total

def sentence_split_by_punc(corpus:str):     # 标点列表，分成小分句
    return re.split(r'[;；.。，,！\n!?？]',corpus)

def remove_irregular_chars(corpus:str):     # 去掉 非（中文字符、0-9、大小写英文）
    return re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", corpus)

def generate_ngram(corpus,n:int=2):
    """
    对一句话生成ngram并统计词频字典，n=token_length,
    返回: generator (节省内存)
    """
    def generate_ngram_str(text:str,n):
        for i in range(0, len(text)-n+1):
            yield text[i:i+n]
    if isinstance(corpus,str):
        for ngram in generate_ngram_str(corpus,n):
            yield ngram
    elif isinstance(corpus, (list, types.GeneratorType)):
        for text in corpus:
            for ngram in generate_ngram_str(text,n):
                yield ngram

def get_ngram_freq_info(corpus, ## list or generator
                        min_n:int=2,
                         max_n:int=4,
                         chunk_size:int=5000,
                         min_freq:int=2,
                         ):
    """
    :param corpus: 接受list或者generator
                   如果corpus是generator, 默认该generator每次yield一段长度为chunk_size的corpus_chunk
    """
    ngram_freq_total = {}  ## 记录词频
    ngram_keys = {i: set() for i in range(1, max_n + 2)}  ## 用来存储N=时, 都有哪些词, 形如 {1: {'应', '性', '送', '灰', '缚',...}, 2: {'术生', '哗吵', '面和', '上恐', '党就', '胁区', '受制', ...}, 3: {'卫生事', '重伤严', '包括教', '关科研',...}, 4: {'护妇女权', '标准的规', '本款第三', '种类后果', '生态效益',...}, 5: {'障国防教育', '管理规定关', '知是指犯罪', '红十字会工', '防护用品进',...}, 6: {'以由几个单位', '占滥用林地的', '规定的行政措', '放本条规定的', '引渡的具体依', '意伤害罪定罪',...}, 7: {}}

    def _process_corpus_chunk(corpus_chunk):
        ngram_freq = {}
        for ni in [1]+list(range(min_n,max_n+2)):
            ngram_generator = generate_ngram(corpus_chunk, ni)
            nigram_freq = dict(Counter(ngram_generator))
            ngram_keys[ni] = (ngram_keys[ni] | nigram_freq.keys())
            ngram_freq = {**nigram_freq, **ngram_freq}
        ngram_freq = {word: count for word, count in ngram_freq.items() if count >= min_freq}  ## 每个chunk的ngram频率统计
        return ngram_freq

    if isinstance(corpus,types.GeneratorType):
        ## 注意: 如果 corpus 是generator, 该function对chunk_size无感知
        for corpus_chunk in corpus:
            ngram_freq = _process_corpus_chunk(corpus_chunk)
            ngram_freq_total = union_word_freq(ngram_freq, ngram_freq_total)
    elif isinstance(corpus,list):
        len_corpus = len(corpus)
        for i in range(0,len_corpus,chunk_size):
            corpus_chunk = corpus[i:min(len_corpus,i+chunk_size)]
            ngram_freq = _process_corpus_chunk(corpus_chunk)
            ngram_freq_total = union_word_freq(ngram_freq,ngram_freq_total)     # 将每个 chunk 的 ngram：频率 对汇总
    for k in ngram_keys:
        ngram_keys[k] = ngram_keys[k] & ngram_freq_total.keys()
    return ngram_freq_total,ngram_keys

def _ngram_entropy_scorer(parent_ngrams_freq):
    """
    根据一个candidate的neighbor的出现频率, 计算Entropy具体值
    :param parent_ngrams_freq:
    :return:
    """
    _total_count = sum(parent_ngrams_freq)
    _parent_ngram_probas = map(lambda x: x/_total_count,parent_ngrams_freq)
    _entropy = sum(map(lambda x: -1 * x * math.log(x, 2),_parent_ngram_probas))
    return _entropy

def _calc_ngram_entropy(ngram_freq,
                        ngram_keys,
                        n,
                        min_entropy):
    """
    基于ngram频率信息计算熵信息
    :param ngram_freq:
    :param ngram_keys:
    :param n:
    :return:
    """
    if isinstance(n,Iterable): ## 一次性计算 len(N)>1 的 ngram
        entropy = {}
        for ni in n:
            entropy = {**entropy,**_calc_ngram_entropy(ngram_freq,ngram_keys,ni,min_entropy)}
        return entropy

    ngram_entropy = {}
    target_ngrams = ngram_keys[n]
    parent_candidates = ngram_keys[n+1]

    ## 对 n+1 gram 进行建Trie处理
    left_neighbors = Trie()
    right_neighbors = Trie()

    for parent_candidate in parent_candidates:
        right_neighbors[parent_candidate] = ngram_freq[parent_candidate]
        left_neighbors[parent_candidate[1:]+parent_candidate[0]] = ngram_freq[parent_candidate]

    ## 计算
    for target_ngram in target_ngrams:
        try:  ## 一定情况下, 一个candidate ngram 没有左右neighbor
            right_neighbor_counts = (right_neighbors.values(target_ngram))
            right_entropy = _ngram_entropy_scorer(right_neighbor_counts)
        except KeyError:
            right_entropy = 0
        try:
            left_neighbor_counts = (left_neighbors.values(target_ngram))
            left_entropy = _ngram_entropy_scorer(left_neighbor_counts)
        except KeyError:
            left_entropy = 0
        if left_entropy > min_entropy and right_entropy > min_entropy:
            ngram_entropy[target_ngram] = (left_entropy,right_entropy)
    return ngram_entropy


def _calc_ngram_pmi(ngram_freq,ngram_keys,n,threshold):
    """
    计算 Pointwise Mutual Information 与 Average Mutual Information
    :param ngram_freq:
    :param ngram_keys:
    :param n:
    :return:
    """
    if isinstance(n,Iterable):
        mi = {}
        for ni in n:
            mi = {**mi,**_calc_ngram_pmi(ngram_freq,ngram_keys,ni,threshold)}
        return mi
    n1_totalcount = sum([ngram_freq[k] for k in ngram_keys[1] if k in ngram_freq])      # 总字数
    mi = {}
    for target_ngram in ngram_keys[n]:
        target_flag = True
        for cut in range(n-1):
            pmi = math.log(n1_totalcount*ngram_freq[target_ngram] / ngram_freq[target_ngram[:n-1-cut]] / ngram_freq[target_ngram[n-1-cut:]],2)
            if pmi <= threshold:
                target_flag = False
                break
        if target_flag:
            mi[target_ngram] = (pmi)
    return mi


def get_scores(corpus,
               min_n:int = 2,
               max_n: int = 4,
               chunk_size:int=5000,
               min_freq:int=0,
               min_pmi:int=0,
               min_entropy:int = 0):
    """
    基于corpus, 计算所有候选词汇的相关评分.
    :param corpus:
    :param max_n:
    :param chunk_size:
    :param min_freq:
    :return: 为节省内存, 每个候选词的分数以tuble的形式返回.
    """
    ngram_freq, ngram_keys = get_ngram_freq_info(corpus,min_n,max_n,
                                                 chunk_size=chunk_size,
                                                 min_freq=min_freq)


    left_right_entropy = _calc_ngram_entropy(ngram_freq,ngram_keys,range(min_n,max_n+1),min_entropy)
    mi = _calc_ngram_pmi(ngram_freq,ngram_keys,range(min_n,max_n+1),min_pmi)
    joint_phrase = mi.keys() & left_right_entropy.keys()
    word_liberalization = lambda le,re: math.log((le * 2 ** re + re * 2 ** le+0.00001)/(abs(le - re)+1),1.5)
    word_info_scores = {word: (mi[word],     #point-wise mutual information
                 left_right_entropy[word][0],   #left_entropy
                 left_right_entropy[word][1],   #right_entropy
                 min(left_right_entropy[word][0],left_right_entropy[word][1]),    #branch entropy  BE=min{left_entropy,right_entropy}
                 word_liberalization(left_right_entropy[word][0],left_right_entropy[word][1])+mi[word]   #our score
                     )
              for word in joint_phrase}

    # word_info_scores 写入 json 文件
    # j_word_scores = json.dumps(word_info_scores)
    # jsonFile = open('scorePie.json','w',encoding='utf-8')
    # jsonFile.write(j_word_scores)
    # jsonFile.close()

    # with open('afterStop.txt','w',encoding='utf-8') as fA:
    #     fA.write('\n'.join(word_info_scores.keys()))

    return word_info_scores


def load_stop():        # load 停用字列表
    stop_Zi = []
    f_stop = open('stopZi.txt','r',encoding='utf-8')
    for line in f_stop:
        if len(line.strip())==1:
            stop_Zi.append(line.strip())
    return stop_Zi


def remove_BE_Repeat(word_info_scores):
    """
        Way 1: 对首尾字的处理

        对在 candidate ngram 中, 首字或者尾字出现次数特别多的进行筛选, 如 "XX的,美丽的,漂亮的" 剔出字典
        -> 测试中发现这样也会去掉很多合理词，如 “法*”
        solve: 处理为对带筛选首尾字进行限制，要求其在停用词表内

        p.s. 停用词来自 [中文停用词](https://github.com/goto456/stopwords)。程序选择 `cn_stopwords.txt`，
            取其单字项构成文件 `stopZi.txt`，共 237 项。
    """
    stop_Zi = load_stop()
    target_ngrams = word_info_scores.keys()
    start_chars = Counter([n[0] for n in target_ngrams])
    end_chars = Counter([n[-1] for n in target_ngrams])
    threshold = int(len(target_ngrams) * 0.004)
    threshold = max(50,threshold)
    invalid_start_chars = set([char for char, count in start_chars.items() if char in stop_Zi and count > threshold])
    invalid_end_chars = set([char for char, count in end_chars.items() if char in stop_Zi and count > threshold])

    invalid_target_ngrams = set([n for n in target_ngrams if (n[0] in invalid_start_chars or n[-1] in invalid_end_chars)])
    for n in invalid_target_ngrams:  ## 按照不合适的字头字尾信息删除一些
        word_info_scores.pop(n)
    return word_info_scores

### more to add, 近期填坑

def extract_phrase(corpus,
                   top_k: float = 200,
                   chunk_size: int = 1000000,
                   min_n:int = 2,
                   max_n:int=4,
                   min_freq:int = 5,
                   min_pmi:int = 0,
                   min_entropy:int = 0):
    #取前k个new words或前k%的new words
    if isinstance(corpus,str):
        corpus_splits = [remove_irregular_chars(sent) for sent in sentence_split_by_punc(corpus)]
    if isinstance(corpus,list):
        corpus_splits = [remove_irregular_chars(sent) for news in corpus for sent in
                                sentence_split_by_punc(str(news)) if len(remove_irregular_chars(sent)) != 0]
    word_info_scores = get_scores(corpus_splits,min_n,max_n,chunk_size,min_freq,min_pmi,min_entropy)
    word_info_scores = remove_BE_Repeat(word_info_scores)
    new_words = [item[0] for item in sorted(word_info_scores.items(),key=lambda item:item[1][-1],reverse = True)]
    if top_k > 1:              #输出前k个词
        return new_words[:top_k]
    elif top_k < 1:            #输出前k%的词
        return new_words[:int(top_k*len(new_words))]

def CSVcombine(menu_path,corpus):
    files = glob.glob(menu_path + '/' + '*.csv')
    for filename in files:
        print(filename)
        logger.info(filename)
        data=pd.read_csv(filename,encoding='utf-8')
        data[u'标题']=data[u'标题'].astype(str)
        corpus.extend(data[u'标题'])
        data[u'文本']=data[u'文本'].astype(str)
        corpus.extend(data[u'文本'])
    return corpus

if __name__=='__main__':
    corpus=[]
    if options.default_csv:
        logger.info("default_csv")
        logger.info("Loading the default csv files...")
        # os.getcwd()
        #os.chdir('C:/Users/Lucky/Desktop/corpus')
        menu_path = os.getcwd() + '/corpus'
        corpus = CSVcombine(menu_path,corpus)
    if options.txt_file is not None:
        logger.info("txt_file")
        logger.info("Loading the txt file...")
        data = open(options.txt_file, 'r', encoding = 'utf-8')
        corpus.extend(data)
    if options.csv_file is not None:
        logger.info("csv_file")
        logger.info("Loading the csv file...")
        data = pd.read_csv(options.csv_file, encoding = 'utf-8')
        data[u'文本']=data[u'文本'].astype(str)
        corpus.extend(data[u'文本'])
    if options.txt_directory is not None:
        logger.info("txt_directory")
        logger.info("Loading the txt files...")
        files = glob.glob(options.txt_directory + '/' + '*.txt')
        for filename in files:
            logger.info(filename)
            data = open(filename, 'r', encoding = 'utf-8')
            corpus.extend(data)
    if options.csv_directory is not None:
        logger.info("csv_directory")
        logger.info("Loading the csv files...")
        files = glob.glob("{}/*.csv".format(options.csv_directory))
        for filename in files:
            logger.info(filename)
            data = pd.read_csv(filename, encoding = 'utf-8')
            data[u'文本']=data[u'文本'].astype(str)
            corpus.extend(data[u'文本'])
    if len(corpus)==0:
        print("Lacking corpus!!!\nYou should give your own corpus. Choose one way from the following five: --default-csv, --txt-file, --csv-file, --txt-directory, --csv-directory.")
        exit(0)
    else:
        logger.info("Corpus Ready...\n")
    result=extract_phrase(corpus,top_k=240000,min_n=options.min_n,max_n=options.max_n,min_freq=options.min_freq,min_pmi=options.min_pmi,min_entropy=options.min_entropy)
    print('Extract Ready...')
    with open("{}/newWord.txt".format(task_dir),'w',encoding='utf-8') as fw:
        fw.write('\n'.join(result))
    print(result)
    logger.info("New words have been saved to " + task_dir)



