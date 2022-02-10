import numpy as np
def LCS(str1, str2):
    '''    计算str1和str2之间的最长公共子串的长度    '''
    if str1 == '' or str2 == '':
        return 0
    len1 = len(str1)
    len2 = len(str2)
    c = np.zeros((len2+1,),dtype=np.int32)
    max_len = 0
    for i in range(len1):
        for j in range(len2-1,-1,-1):
            if str1[i] == str2[j]:
                c[j+1] = c[j] + 1
                if c[j+1]>=max_len:
                    max_len = c[j+1]
            else:
                c[j+1] = 0
    return max_len

def _levenshtein_distance(str1, str2):
    '''    计算str1和str2之间的莱文斯坦距离，即编辑距离    '''
    if not isinstance(str1, str):
        str1 = "null"
    x_size = len(str1) + 1
    y_size = len(str2) + 1
    matrix = np.zeros((x_size, y_size), dtype=np.int_)
    for x in range(x_size):
        matrix[x, 0] = x
    for y in range(y_size):
        matrix[0, y] = y
    for x in range(1, x_size):
        for y in range(1, y_size):
            if str1[x - 1] == str2[y - 1]:
                matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x - 1, y - 1], matrix[x, y - 1] + 1)
            else:
                matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x - 1, y - 1] + 1, matrix[x, y - 1] + 1)
    return matrix[x_size - 1, y_size - 1]

def get_title_in_query(query_list,title_list):
    ret_status=[]
    for i in range( len(query_list) ):
        query = query_list[i]
        title = title_list[i]
        status=0
        if query != ' ' and query!='' and title in query:
            status=1
        ret_status.append(status)
    return ret_status

def get_query_in_title(query_list,title_list):
    ret_status=[]
    for i in range( len(query_list) ):
        query = query_list[i]
        title = title_list[i]
        status=0
        if query != ' ' and query!='' and   query in title:
            status=1
        ret_status.append(status)
    return ret_status
def get_LCS_list(query_list,title_list):
    ret_status=[]
    for i in range( len(query_list) ):
        distance=LCS(query_list[i], title_list[i])
        ret_status.append(distance)
    return ret_status

def get_evenshtein_distance_list(query_list,title_list):
    ret_status=[]
    for i in range( len(query_list) ):
        distance=_levenshtein_distance(query_list[i], title_list[i])
        ret_status.append(distance)
    return ret_status

from nltk import word_tokenize
from gensim import corpora
import numpy as np


document_split = ['.', ',', '?', '!', ';']
dict_file = 'dict_file.txt'
def load_dataset(filename='Alice.txt'):
    # 读入文件
    with open(file=filename, mode='r') as file:
        document = []
        lines = file.readlines()
        for line in lines:
            # 删除非内容字符
            value = clear_data(line)
            if value != '':
                # 对一行文本进行分词
                for str in word_tokenize(value):
                    # 跳过章节标题
                    if str == 'CHAPTER':
                        break
                    else:
                        document.append(str.lower())

        return document
def clear_data(str):
    # 删除字符串中的特殊字符或换行符
    value = str.replace('\ufeff', '').replace('\n', '')
    return value
# 生成词云
def show_word_cloud(document):
    # 需要清楚的标点符号
    left_words = ['.', ',', '?', '!', ';', ':', '\'', '(', ')']
    # 生成字典
    dic = corpora.Dictionary([document])

    print( 'dic.items:', list( dic.items()) )
    # 计算得到每个单词的使用频率
    words_set = dic.doc2bow(document)

    print('words_set:',  words_set )
    # 生成单词列表和使用频率列表
    words, frequences = [], []
    for item in words_set:
        key = item[0]
        frequence = item[1]
        word = dic.get(key=key)
        if word not in left_words:
            words.append(word)
            frequences.append(frequence)
    # 使用pyecharts生成词云
    word_cloud = WordCloud()#width=1000, height=620
    word_zip= list( zip(words, frequences ) )
    print( word_zip[:10]  )
    word_cloud.add( '',words_set, shape='circle', word_size_range=[20, 100]) #name='Alice\'s word cloud', attr=words,
    word_cloud.render()

def word_to_integer(document):
    # 生成字典
    dic = corpora.Dictionary([document])
    # 保存字典到文本文件
    dic.save_as_text(dict_file)
    dic_set = dic.token2id
    #print(dic)
    # 将单词转换为整数
    values = []
    for word in document:
        # 查找每个单词在字典中的编码
        values.append(dic_set[word])
    return values


# 数据清洗
def deal_row_text(row_data, add_stoplist, whitelist, flags):
    # 繁转简
    texts = row_data.iloc[:, 0].apply(lambda x: opencc.OpenCC('t2s').convert(x))
    # 逐句分词
    text = [jp.cut(i) for i in texts]
    # 去停用词
    stoplist = open('cn_stopwords.txt', 'r', encoding='utf-8').read().split('\n') + add_stoplist
    # 加白名单
    for i in whitelist:
        jieba.add_word(i)
    data = [[w.word for w in words if (w.word not in stoplist and w.flag in flags and len(w.word) > 1)] for words in text]
    return data


# 词频统计
def words_frequent(data):
    frequency = defaultdict(int)
    for i in data:
        for j in i:
            frequency[j] += 1
    sort_frequency = list(frequency.items())
    sort_frequency.sort(key=lambda x: x[1], reverse=True)
    return sort_frequency
def word_to_integer(document):
    # 生成字典
    dic = corpora.Dictionary([document])
    # 保存字典到文本文件
    dic.save_as_text(dict_file)
    dic_set = dic.token2id
    #print(dic)
    # 将单词转换为整数
    values = []
    for word in document:
        # 查找每个单词在字典中的编码
        values.append(dic_set[word])
    return values
def load_dataset(filename):
    # 读入文件
    with open(file=filename, mode='r') as file:
        document = []
        lines = file.readlines()
        for line in lines:
            # 删除非内容字符
            value = clear_data(line)
            if value != '':
                # 对一行文本进行分词
                for str in word_tokenize(value):
                    # 跳过章节标题
                    if str == 'CHAPTER':
                        break
                    else:
                        document.append(str.lower())

        return document
def clear_data(str):
    # 删除字符串中的特殊字符或换行符
    value = str.replace('\ufeff', '').replace('\n', '')
    return value

# 生成词云
def show_word_cloud(document):
    # 需要清楚的标点符号
    left_words = ['.', ',', '?', '!', ';', ':', '\'', '(', ')']
    # 生成字典
    dic = corpora.Dictionary([document])
    # 计算得到每个单词的使用频率
    words_set = dic.doc2bow(document)

    # 生成单词列表和使用频率列表
    words, frequences = [], []
    for item in words_set:
        key = item[0]
        frequence = item[1]
        word = dic.get(key=key)
        if word not in left_words:
            words.append(word)
            frequences.append(frequence)
    # 使用pyecharts生成词云
    word_cloud = WordCloud(width=1000, height=620)
    word_cloud.add(name='Alice\'s word cloud', attr=words, value=frequences, shape='circle', word_size_range=[20, 100])
    word_cloud.render()

import time
import jieba
import tushare as ts
import pandas as pd


if __name__ == '__main__':
    print( ts.get_concept_classified() )
    print(ts.get_latest_news())
    #print( ts.get_latest_news() )

    print(word_tokenize('this is a good example.!'))
    print( list( jieba.cut('我们怎么样啊') ) )

    document = load_dataset()
    print(document[:20])
    # eixt()
    show_word_cloud(document)
    values = word_to_integer(document)

    num_topics = 15
    texts = pd.read_excel('past.xlsx', encoding='utf-8', header=None).astype(str)
    flags = ['n', 'nr', 'v', 'ns', 'nt', 'd']
    add_stoplist = ['知道', '了解', '感受', '矿泉水', '公司', '东北', '案例', '包装', '作者', '看到', '转载', '商品', '日本', '消费者', '心理', '使用', '越来越', '购买', '一定', '个人', '企业', '没', '做', '说', '非常', '可能', '原', '美国', '中国', '产品', '问题']
    whitelist = ['iPhone 11', '泸州老窖']
    data = deal_row_text(texts, add_stoplist, whitelist, flags)
    # record_data(data)
    # model_result(num_topics=num_topics, passes=100)
