
import jieba
def jieba_word_cut_(doc):
    seg = [jieba.lcut(w) for w in doc]
    return seg

#gensim做自然语言处理的一般思路是：使用（处理）字典 -> 生成（处理）语料库 -> 自然语言处理（tf-idf的计算等)
from gensim import corpora
def gen_corpora_dic(texts,  dic_path='test.dict'  ):
    # corpora.Dictionary()方法，括号里面的对象的形式是这样的：[ [list1] , [list2] ]
    dictionary = corpora.Dictionary(texts)
    if dic_path !='':
        dictionary.save( dic_path )
    return dictionary

def load_corpora_dic( dic_path ):
    dictionary = corpora.Dictionary.load( dic_path )   # 加载字典
    return dictionary

from gensim import models
def gen_TfidfModel( corpus, dict, path=''):
    tfidf_model = models.TfidfModel(corpus=corpus, dictionary=dict)
    if path!='':
        tfidf_model.save( path ) #保存模型到本地
    return tfidf_model

def load_TfidfModel( path ):
    tfidf_model = models.TfidfModel.load( path )  # 载入模型
    return tfidf_model

from gensim import models, similarities
def sample( texts,dictionary,tfidf_model  ):
    #  dict.token2id    返回字典：{ 单词：id }
    #  dict.id2token	返回字典：{ id： 单词 }
    #  dict.items()     返回对象：[(id, 词)]，查看需要循环
    #  dict.dfs	        返回字典：{ 单词id：在多少文档中出现 }
    #  dict.num_docs	返回文档数目
    #  词袋形象地解释便是：将文本中的词放入一个袋子里，在袋子中，词拥有两个属性编号和数目（id，num）
    ## 语料库对象（corpus） , doc2bow 是建立词袋向量
    bow_corpus  = [ dictionary.doc2bow(text) for text in texts] #词袋是指词在本文本中出现的次数，一个词袋就是一篇文章
    tfidf       = models.TfidfModel( bow_corpus, dictionary=dictionary )
    tfidf_model = gen_TfidfModel(bow_corpus, dictionary=dictionary  ) ##TfidfModel 可以把bow转化为 tfid特征

    # tfidf_model = models.TfidfModel(corpus)
    # corpus_tfidf = [tfidf_model[doc] for doc in corpus]
    return tfidf

from gensim import corpora, models

def gen_ldamodel(corpus,dictionary,topcnt):
    ldamodel = models.LdaModel(corpus, id2word=dictionary, num_topics=topcnt )
    ldamodel.print_topics()
    return ldamodel

def gen_lsimodel( corpus_tfidf,dictionary,topcnt=50  ): #input shi tfid 向量
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=topcnt)
    return lsi

import pandas as pd
def get_words_probability( query_list,title_list,label_list  ):
    word_label=[]
    for i in range( len(query_list) ):
        query_token= query_list[i].split(' ')
        title_token= title_list[i].split(' ')
        for j in query_token:
           word_label.append([j,label_list[i]  ])
        for j in title_token:
            word_label.append( [j,label_list[i] ])
    word_label_df = pd.DataFrame(data=word_label,columns=['word','label'])
    word_label_df['cnt'] =1
    word_groupby_df = word_label_df.groupby('word').sum()
    word_groupby_df['rate'] =  word_groupby_df['label']/word_groupby_df['cnt']
    key_word_click_rate={}
    for i in word_groupby_df.reset_index().values:
        key_word_click_rate[i[0]] = i[3]

    return word_groupby_df,key_word_click_rate


def process_click_inqure():
    df = pd.read_csv('./click_pred/train_data.sample', sep=',', header=None)
    df.columns = ['query_id', 'query', 'query_title_id', 'title', 'label']
    df['LCS'] = get_LCS_list(df['query'].tolist(), df['title'].tolist())
    df['evenshtein_distance'] = get_evenshtein_distance_list(df['query'].tolist(), df['title'].tolist())
    df['title_in_query'] = get_title_in_query(df['query'].tolist(), df['title'].tolist())
    df['query_in_title'] = get_query_in_title(df['query'].tolist(), df['title'].tolist())
    return


from gensim import corpora
from gensim.models import LdaModel
from gensim import models
from gensim.corpora import Dictionary
import pandas as pd
import jieba.posseg as jp, jieba
import opencc
from collections import defaultdict




# 词袋编码
def record_data(data):
    global dictionary, corpus, corpus_tfidf
    dictionary = corpora.Dictionary(data)
    dictionary.filter_extremes(no_below=3, no_above=0.4, keep_n=500)
    corpus = [dictionary.doc2bow(text) for text in data]
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]


# 训练模型
def model_result(num_topics=5, passes=100):
    global lda, doc_topic
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)
    doc_topic = [a for a in lda[corpus]]


# 标准化结果展示
def format_result(on_of=False, num_topics=5, num_words=5):
    key_word = []
    on_of = True
    for i in lda.show_topics(num_topics=num_topics, num_words=num_words, log=False, formatted=False):
        word = []
        for j in i[1]:
            word.append(j[0])
            if on_of == True:
                word.append(j[1])
        key_word.append(word)
    return key_word


# 文章分类
def classify_article(num_topics):
    topic_list = [i for i in range(num_topics)]
    article_class = defaultdict()
    for i in topic_list:
        article_class[i] = []
    for i in range(len(doc_topic)):
        doc_topic[i].sort(key=lambda x: x[1], reverse=True)
        article_class[topic_list[doc_topic[i][0][0]]].append(texts.iloc[i, 0])
    return article_class




if __name__ == '__main__':
    documents = ['工业互联网平台的核心技术是什么',
                 '工业现场生产过程优化场景有哪些']
    texts = jieba_word_cut_(documents)
    for k in texts:
        print(k)