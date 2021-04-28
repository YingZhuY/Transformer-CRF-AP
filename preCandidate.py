from collections.abc import Iterable
from collections import Counter
from pygtrie import Trie
import pandas as pd
import numpy as np
import argparse
import logging
import os
import time
import sys
import csv
import glob
import types
import math
import re
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

parser.add_argument("--pair-punc", dest="pair_punc", action="store_true",help="Special treatment of paired punctuation. Like 、__、 《__》 （__） “__”")
parser.add_argument("--BE-stop", dest="BE_stop", action="store_true", help="Filter with BE-stop mode.")
parser.add_argument("--wiki", dest="wiki", action="store_true", help="Filter with wiki mode.")
parser.add_argument("--Re-pattern", dest="Re_pattern", action="store_true", help="Filter with Re-pattern mode.")
parser.add_argument("--selectN", dest="selectN", type=int, default=100, help="Top N to select in the Re-pattern mode (without PUNC)")
parser.add_argument("--separate",dest="separate",action="store_true",help="Separate mode. Extract topK from each ")

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
parser.add_argument("--topK", dest="top_k", type=float, default=200,
                    help="Output the top k new words.1. topK>1: the top k words; 2. topK<1: the top k percent words; 3. topK=1: all words")

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
    file_handler = logging.FileHandler("{0}/info.log".format(task_dir), mode='w', encoding='utf-8')
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


keep_dict = [u'（', u'）', u'：', u'、', u'《', u'》', u'“', u'”', u'—', u'…']
PUNC_SET = set(u'!#$%&()*+,-./:;<=>?@[\\]^_`{|}~＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～､　、〜〟〰〾〿–—„…‧﹏﹑﹔·！？｡。□．×℃°ｔ′ｘ…∶∠→．△Ｉ≤‖±●Δ∞∈％≥。，？√﹟<"《》“”．％—…』『／’‘Ｉ（）．％「」─〔？【】．…⦅\'')


def containsAny(seq, aset):                 # 判断字符串是否包含 list 中的元素
    return True if any(i in seq for i in aset) else False


def remove_irregular_chars(corpus:str):     # 去掉 非（中文字符、0-9、大小写英文）
    # 定义一些经常出现的重要符号，避免其前后位置连接成不合理的候选 ngram
    # keep_dict_unicode = ['\uff08', '\uff09', '\uff1a', '\u3001', '\u300a', '\u300b', '\u201c', '\u201d', '\u2014', '\u2026']
    # \uff08\uff09\uff1a\u3001\u300a\u300b\u201c\u201d\u2014\u2026
    return re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a\uff08\uff09\uff1a\u3001\u300a\u300b\u201c\u201d\u2014\u2026])", "", corpus)


def generate_ngram(corpus,n:int=2):
    """
    对一句话生成ngram并统计词频字典，n=token_length,
    返回: generator (节省内存)
    """
    def generate_ngram_str(text:str,n):
        for i in range(0, len(text)-n+1):
            if not containsAny(text[i:i+n], keep_dict):
                yield text[i:i+n]
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
    word_info_scores = {word: [mi[word],     #point-wise mutual information
                 left_right_entropy[word][0],   #left_entropy
                 left_right_entropy[word][1],   #right_entropy
                 min(left_right_entropy[word][0],left_right_entropy[word][1]),    #branch entropy = min{left_entropy,right_entropy}
                 word_liberalization(left_right_entropy[word][0],left_right_entropy[word][1])+mi[word]   #our score
                ]
              for word in joint_phrase}
    if options.restore_score:       #word_info_scores 写入 json 文件
        j_word_scores = json.dumps(word_info_scores)
        jsonFile = open('score.json','w',encoding='utf-8')
        jsonFile.write(j_word_scores)
        jsonFile.close()

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


def remove_all_digit_alpha(target_dict):       # 去掉目标词典中所有含英文或数字项（Way 2）
    # 此处只保留中文项 remind error: RuntimeError: dictionary changed size during iteration
    words = [ key for key,value in target_dict.items() ]
    for w in words:
        if re.search('[\u4e00-\u9fa5]*',w).group(0) != w:             # 匹配第一个连续的中文片段
        #if "".join(re.compile('[^\u4e00-\u9fa5]').split(line)) != line:    # 匹配所有中文
            target_dict.pop(w)
    return target_dict


def remove_only_digit_alpha(target_dict):       # 去掉目标词典中仅英文或数字项
    words = [ key for key,value in target_dict.items() ]
    for w in words:
        if re.search('^[a-zA-Z]+$', w) or re.search('^[0-9]+$', w):             # 匹配第一个连续的中文片段
            target_dict.pop(w)
    return target_dict


def scale_by_wiki_index(word_info_scores, min_n, max_n):
    """
        Way 3：WiKi_index Mode

        `WiKi_index.txt` 是从 `zhwiki-20210301-pages-articles-multistream-index.txt.bz2` 中提取、简化、预处理得到的，其每一项都为一个中文维基百科词条。
        维基百科词条是确定可以成词的，我们提高它对应的权重（这里设置为一个经验值 1.2），有助于区别脏词。
    """
    wiki_index_file = open('WiKi_index.txt','r',encoding='utf-8')
    wiki_index = wiki_index_file.readlines()
    for index in wiki_index:
        index = index.strip()
        if len(index)<min_n or len(index)>max_n:
            continue
        if index in word_info_scores:
            word_info_scores[index][-1] = word_info_scores[index][-1] * 1.2
    return word_info_scores


def load_temp(n=100):       # 导入 random_select_aver.txt 文件前 n 个不含标点的模式到字典 temp_dict 中（Way 4）
    temp_list = []
    count = 0
    for line in open('random_select_aver.txt','r',encoding='utf-8').readlines():
        line = line.strip()
        segs = line.split('\t')
        if any([a in PUNC_SET for a in segs[0].split('__')]):
            continue
        temp_list.extend(segs[0])
        count += 1
        if count >= n:
            break
    return temp_list


def pattern_filter(result):
    """
        Way 4：Re_pattern based filter

        使用 random_select_aver.txt 文件中的 pattern（取不含 PUNC 的 Top N）。TopN 不推荐设置很大，太消耗内存。
        对匹配到的模式产生的词，如从 `从大分式开始dfdsf` 匹配到 `大分式`
        1. 若 `大分式` 不在 result 中，用子词 `大分式` 替换其父词 `从大分式开始dfdsf`
        2. 若 `大分式` 在 result 中，删除其父词

        p.s. 对于匹配到的子词，注意限制 >= min_n （上限不用判断，因为不可能越剪越长）

    # 版本一
    # for corn in result:
    #     for temp in reversed(pattern_list):
    #         ts = temp.split('__')
    #         m = re.finditer(ts[0] + '.*?' + ts[1], corn)    # 一种模式可能有多个匹配项，选择第一项（简化计算）；一个字符串可能匹配到多个模式，选择最后一个（概率高，reversed）
    #         for a in m:
    #             w = a.group(0)[len(ts[0]):-len(ts[1])]
    #             if len(w) >= 2 and w not in result:
    #                 toPrint.append(w)
    #                 result[i]=w
    #             else:
    #                 result.remove(corn)
    #             break
    #     i+=1
    """
    toPrint=[]
    pattern_list = load_temp(options.selectN)
    i = 0
    for corn in result:
        for temp in pattern_list:
            ts = temp.split('__')
            m = re.search(ts[0] + '.*?' + ts[1], corn)    # 字符串 corn 可能匹配到多个模式 pattern / temp，考虑下简化计算，取第一个（越前边概率越高）
            if m:       # 当发生匹配
                w = m.group(0)[len(ts[0]):-len(ts[1])]
                if len(w) >= 2 and w not in result:
                    toPrint.append(w)
                    result[i]=w
                else:
                    result.remove(corn)
                break
        i+=1
    logger.info("You used Way 4 [Re_pattern based filter] and obtained the following words:")
    logger.info(toPrint)
    logger.info('')
    return result


def extract_phrase(corpus,
                   top_k: float = 200,
                   chunk_size: int = 1000000,
                   min_n:int = 2,
                   max_n:int = 5,
                   min_freq:int = 10,
                   min_pmi:int = 0,
                   min_entropy:int = 0):
    #取前k个new words或前k%的new words
    if isinstance(corpus,str):
        corpus_splits = [remove_irregular_chars(sent) for sent in
                        sentence_split_by_punc(corpus) if len(remove_irregular_chars(sent)) != 0]
    elif isinstance(corpus,list):
        corpus_splits = [remove_irregular_chars(sent) for news in corpus for sent in
                        sentence_split_by_punc(str(news)) if len(remove_irregular_chars(sent)) != 0]
    else:
        print("Wrong file type!")
    """
        Way 2：对成对标点包括的内容单独处理 **生成部分**

        对预处理后的小句子 corpus_splits 进行匹配，利用成对的标点，pair_mode 定义为 {mode：weight}
        pair_mode={'“__”': 1, '《__》': 2, '（__）': 3, '、__、': 4}
        -> 将其包括起来的内容符合 [min_n,max_n] 的给予更高的候选词权重

        p.s. 匹配得到的词中可能含有标点，记得筛去
    """
    if options.pair_punc: 
        pair_punc_dict={}
        pair_mode={'《__》': 2, '（__）': 1.1, '、__、': 1.5, '“__”': 1.5}
        for corn in corpus_splits:
            for temp in pair_mode:
                ts = temp.split('__')
                m = re.finditer(ts[0] + '.*?' + ts[1], corn)    # 可能有多个匹配项
                for a in m:
                    word = a.group(0)[1:-1]
                    if len(word)>=min_n and len(word)<=max_n and word not in pair_punc_dict and not containsAny(word, keep_dict):
                        pair_punc_dict[word] = pair_mode[temp]
        pair_punc_dict = remove_all_digit_alpha(pair_punc_dict)
        # j_pair = json.dumps(pair_punc_dict)
        # je = open('pair.json','w',encoding='utf-8')
        # je.write(j_pair)
        # je.close()
        # exit(0)

    word_info_scores = get_scores(corpus_splits,min_n,max_n,chunk_size,min_freq,min_pmi,min_entropy)
    
    """
        Way 2：对成对标点包括的内容单独处理 **执行部分**

        1. 对于在 `word_info_scores` 中的词，将其 score 乘以对应的经验值（按成对标点分类），此处设的经验值有依据对应 mode 频数 and 总体质量
        2. 对于不在 `word_info_scores` 中的词，直接将其 score 置为固定值（程序中置为 score 平均值）并加入 word_info_scores。
    """
    if options.pair_punc:
        scores = [ value[-1] for key, value in word_info_scores.items() ]
        avg_score = np.mean(scores)
        logger.info("Words extracted from PAIR_PUNC mode that not in word_info_scores are setting to Score: " + str(avg_score) + '\n')
        pair_punc_words = [ key for key,value in pair_punc_dict.items() ]
        for ppw in pair_punc_words:
            if ppw in word_info_scores:
                word_info_scores[ppw][-1] = word_info_scores[ppw][-1] * pair_punc_dict.get(ppw)
            else:
                word_info_scores[ppw]=[0,0,0,0,avg_score]

    word_info_scores = remove_only_digit_alpha(word_info_scores)

    if options.BE_stop:     ### Way 1
        word_info_scores = remove_BE_Repeat(word_info_scores)

    if options.wiki:                    ### Way 3
        word_info_scores = scale_by_wiki_index(word_info_scores, min_n, max_n)

    # 排序取 TOP
    new_words = [item[0] for item in sorted(word_info_scores.items(),key=lambda item:item[1][-1],reverse = True)]
    assert top_k > 0
    if options.separate:
        top_k = min(20, 0.1*len(new_words))
    if top_k > 1:               # 输出前 k 个词
        return new_words[:top_k]
    elif top_k < 1:             #输出前 k% 的词
        return new_words[:int(top_k*len(new_words))]
    else:
        return new_words


def CSVcombine(menu_path,corpus):
    files = glob.glob(menu_path + '/' + '*.csv')
    for filename in files:
        logger.info(filename)
        data=pd.read_csv(filename,encoding='utf-8')
        data[u'标题']=data[u'标题'].astype(str)
        corpus.extend(data[u'标题'])
        data[u'文本']=data[u'文本'].astype(str)
        corpus.extend(data[u'文本'])
    return corpus


if __name__=='__main__':
    corpus=[]
    tloadS=time.time()
    if options.default_csv:
        logger.info("default_csv")
        logger.info("Loading the default csv files...")
        # os.getcwd()
        #os.chdir('C:/Users/Lucky/Desktop/corpus')
        menu_path = os.getcwd() + '/corpus'
        if not os.path.exists(menu_path):
            print("Please cd the right directory, or put default corpus in the right way.")
            exit(0)
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
    tloadE=time.time()
    logger.info("Loading time: {} loss: {} \nExample: {}\n".format(tloadE-tloadS,len(corpus),corpus[0]))

    textractS=time.time()
    if options.separate:
        result = []
        logger.info("Using the separate Mode, we will do [extract_phrase] {} times. For some considerations, the final words are disordered.\n".format(len(corpus)))
        for each in corpus:
            result.extend(extract_phrase(corpus,min_n=options.min_n,max_n=options.max_n,min_freq=options.min_freq,min_pmi=options.min_pmi,min_entropy=options.min_entropy))
        result = list(set(result))
    else:    
        result = extract_phrase(corpus,top_k=options.top_k,min_n=options.min_n,max_n=options.max_n,min_freq=options.min_freq,min_pmi=options.min_pmi,min_entropy=options.min_entropy)
    
    if options.Re_pattern: 
        result = pattern_filter(result)    

    print('Extract Ready...')
    textractE=time.time()
    logger.info("Extracting time: {} len(New Words): {}".format(textractE-textractS,len(result)))

    with open("{}/newWord.txt".format(task_dir),'w',encoding='utf-8') as fw:
        fw.write('\n'.join(result))
    logger.info(result)
    logger.info("New words have been saved to " + task_dir)



