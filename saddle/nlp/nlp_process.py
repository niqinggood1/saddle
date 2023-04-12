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

def levenshtein_distance(str1, str2):
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


# # 词频统计
# def words_frequent( docs ): #input docs ,output word:frequence
#     frequency = defaultdict(int)
#     for i in docs:
#         for j in i:
#             frequency[j] += 1
#     sort_frequency = list(frequency.items())
#     sort_frequency.sort(key=lambda x: x[1], reverse=True)
#     return sort_frequency

def str_strip(str):
    # 删除字符串中的特殊字符或换行符
    value = str.replace('\ufeff', '').replace('\n', '')
    return value


def wordbag_to_dummy( features ):
    ret_list=[]
    for f in features:
        fsubs= [  1 if fsub>0 else 0  for fsub in f ]
        ret_list.append(  fsubs )
    return ret_list


import os
import jieba
class jiebaX():
    def __init__(self,stopwords_file):
        self.stopwords =[]
        if os.path.exists( stopwords_file ):
            self.stopwords = stopwordslist( stopwords_file )
        else:
            print(stopwords_file,'not exists')

    def load_userdict(self,files):
        for f in files:
            if os.path.exists( f ):
                jieba.load_userdict( f )
                print('jieba load ',f )
            else:
                print(f,'does not exists')
            # jieba.load_userdict('./data/nl/stkcode_name2.txt')
            # jieba.load_userdict('./data/nl/stock_name.txt')
            # jieba.load_userdict('./data/nl/bankuai.txt')
            # jieba.load_userdict('./data/nl/stopwords.txt')
            # jieba.load_userdict('./data/nl/Intelligent_customer_service.csv')
    def add_jieba_words(self,words):
        for w in words:
            jieba.add_word(w)
        return
    def cut_words(self,docs):
        seg =  [ jieba.lcut(d) for d in docs ]
        removed_seg=[]
        for s in seg:
            s   = [ w  for w in s   if w not in self.stopwords  ]
            removed_seg.append(  s )
        return  removed_seg




# crreate stop words list
def stopwordslist(filepath='J:\\work\\computational_Finance\\nlp_warehouse\\stopwords.txt'):
    stopwords =[' '] + [ line.strip() for line in open(filepath, 'r',encoding="utf-8").readlines()]
    return stopwords

#add myself stopwords






def en_word_cut( docs ):
    stoplist = set('for a of the and to in . ,'.split())
    texts = [[word for word in doc.lower().split() if word not in stoplist]
             for doc in docs]
    return texts

from collections import defaultdict
def frequency_statas( texts ): #Texts is cuted docs
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    return frequency

def remove_only_one( texts,frequency):
    texts = [ [token for token in text if frequency[token] > 1]
             for text in texts ]
    return texts

def load_corpora_dic( dic_path ):
    dictionary = corpora.Dictionary.load( dic_path )   # 加载字典
    return dictionary

from gensim import corpora
def gen_corpora_dic( seg_texts,  dic_path='test.dict'  ):
    # gensim做自然语言处理的一般思路是：使用（处理）字典 -> 生成（处理）语料库 -> 自然语言处理（tf-idf的计算等)
    # corpora.Dictionary()方法，括号里面的对象的形式是这样的：[ [list1] , [list2] ]
    # corpora是gensim中的一个基本概念，是文档集的表现形式，也是后续进一步处理的基础。从本质上来说，corpora其实是一种格式，其实就是一个二维矩阵
    # Dictionary 是一个工具，一个字典，里面给每个word都编码 子是token,编码是id
    #  dict.token2id    返回字典：{ 单词：id }
    #  dict.id2token	返回字典：{ id： 单词 }
    #  dict.items()     返回对象：[(id, 单词)]，查看需要循环
    #  dict.dfs	        返回字典：{ 单词id：在多少文档中出现 }
    #  dict.num_docs	返回文档数目
    #  dic.doc2bow      返回的是密集特征，变成词袋，Vec
    dictionary = corpora.Dictionary( seg_texts )
    if dic_path !='':
        dictionary.save( dic_path )
    return dictionary

def gen_texts_tfid_withDict( seged_texts, dictionary):
    #  词袋形象地解释便是：将文本中的词放入一个袋子里，在袋子中，词拥有两个属性编号和数目（id，num）
    ## 语料库对象（corpus） , doc2bow 是建立词袋向量; bag of world  等同于vectorizer.fit_transform( corpus )
    bowed_docs   = [ dictionary.doc2bow(text) for text in seged_texts] #词袋是指词在本文本中出现的次数，一个词袋就是一篇文章
    tfidf_model  = gen_TfidfModel(bowed_docs, dict=dictionary  ) ##TfidfModel 可以把bow转化为 tfid特征
    corpus_tfidf = [ tfidf_model[tmp] for tmp in bow_corpus]
    return corpus_tfidf, tfidf_model

def tfidf2vec(corpus_tfidf, length): # trans bow to index ordered vector
    #可以通过bow2vector进行转换;  max(dictionary)+1  length = max(dictionary) + 1
    vec = []
    for content in  corpus_tfidf :
        sentense_vectors = np.zeros(length)
        for co in content:
            sentense_vectors[co[0]] = co[1]
        vec.append(sentense_vectors)
    return vec

def gen_tfid_with_mode(seged_texts, dictionary,tfidf_model):
    # dictionary use to gen bow   tfidf_model used to gen vectors
    bowed_text = [dictionary.doc2bow(text) for text in seged_texts]  # doc2bow 变成词袋 bag of word词袋是指词在本文本中出现的次数，一个词袋就是一篇文章
    corpus_tfidf = [tfidf_model[tmp] for tmp in bowed_text]
    return corpus_tfidf


from gensim import models
def gen_TfidfModel( bowed_docs, dict, path=''):
    tfidf_model = models.TfidfModel(corpus=bowed_docs, dictionary=dict)
    if path!='':
        tfidf_model.save( path ) #保存模型到本地
    return tfidf_model

def load_TfidfModel( path ):
    tfidf_model = models.TfidfModel.load( path )  # 载入模型
    return tfidf_model

from gensim import models,corpora,  similarities
def gen_ldamodel(corpus,dictionary,topcnt):
    ldamodel = models.LdaModel(corpus, id2word=dictionary, num_topics=topcnt )
    ldamodel.print_topics()
    return ldamodel

def gen_lsimodel( corpus_tfidf,dictionary,topcnt=50  ): #input shi tfid 向量
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=topcnt)
    return lsi



#### based on  sklearn is more comfortable
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

def create_sklearn_CountVectorizer( corpus  ):
    vectoerizer = CountVectorizer(min_df=1, max_df=1.0, token_pattern='\\b\\w+\\b')  ##创建词袋数据结构
    vectoerizer.fit( corpus )   #转成词袋模型
    #vectoerizer.vocabulary_ 此特征及索引
    #bag_of_words = vectoerizer.get_feature_names()
    # print(X.toarray())#看到词频矩阵
    return vectoerizer

def get_SKlearn_BOW( vectoerizer, corpus ):
    x       = vectoerizer.transform( corpus )
    xarray  = x.toarray( )
    return  xarray

def get_sklearn_Tfidf_model( count_list ):
    # TfidfTransformer 继续在CountVectorizer基础上计算tfidf
    #输入词袋  由vectorizer.fit_transform 后返回得到
    tfidf         = TfidfTransformer()
    tf_idf_model  = tfidf.fit( count_list )
    transform
    return  transformer,   # transformer 和 稀疏的tfidf的矩阵

def get_sklearn_tfid(tfid_model,count_list):
    tfidf_matrix = tfid_model.transform( count_list )
    return tfidf_matrix.toarray()

#TfidfVectorizer 是CountVectorizer和TfidfTransformer的组合
from sklearn.feature_extraction.text import TfidfVectorizer
def get_sk_TfidfVectorizer( docs,max_df=0.8, min_df=4, ):  #wrong code
    tfidf_vec  = TfidfVectorizer(use_idf=True, smooth_idf=True, norm=None,token_pattern=r"(?u)\b\w+\b" ,
                                 max_df = max_df,
                                 min_df = min_df
                                 )
    tv_fit     = tfidf_vec.fit(docs) #_transform
    return  tv_fit   #tfidf_matrix.toarray()

#    #tfidf_vec.get_feature_names() 可以得到features的key
    #tfidf_vec.vocabulary_ 得到map

def get_tfidvec_withModel(tv_fit,raw_docs):
    tfidf_matrix  = tv_fit.transform( raw_docs )
    matrix        = tfidf_matrix.toarray()
    return matrix

def get_pos_neg_df(file_path):
    import pandas as pd
    df = pd.read_csv(file_path)
    pos = df['Positive'].tolist()
    neg= df['Negative'].tolist()
    return pos,neg

def test2():
    documents = ['工业互联网平台的核心技术是什么',
                 '工业现场生产过程优化场景有哪些']
    texts = jieba_cut_word(documents)
    for k in texts:
        print(k)
    texts2 = [ ' '.join(t)   for t in texts ]
    print( texts2 )
    tv_fit = get_sk_TfidfVectorizer( texts2  )
    v      = get_tfidvec_withModel(  tv_fit, texts2  )
    print( v[0] )
    print( v[1] )
    return


def save_mode(model,filepath):
    joblib.dump(model, filepath)
    return

import joblib
def load_model(filepath):
    model = joblib.load( filepath )
    return model

class PosNegDic():
    def __init__(self, file_path):
        print('初始化方法')
        self.file_path = file_path
        pos,neg =  get_pos_neg_df( file_path )
        self.pos= set( pos )
        self.neg= set( neg )

import pandas as pd
class Sentiment():
    def __init__(self):
        #load corpus
        self.label_text_df =pd.DataFrame({'label':[], 'text':[] })
        return
    def load_sentiment_XS_30k(self,file_path):
        # sentiment_XS_30k.txt  #sentiment_XS_30k.txt
        tmp_df = pd.read_csv( file_path   )
        tmp_df = tmp_df.rename( columns={'labels':'label' ,'text': 'text' } )
        self.label_text_df = self.label_text_df.append(  tmp_df )
        return

    def load_sentiment_5type(self, copor_dir):
        for sub in ['clothing','fruit','hotel','pda','shampoo']:
            # n_df = pd.read_csv(copor_dir+'\\neg.txt',header=None);n_df.columns=['']
            # p_df = pd.read_csv(copor_dir+'\\pos.txt',header=None )
            negs = [ line.strip() for line in open(copor_dir+'\\'+sub+'\\neg.txt','r', encoding="UTF-8-SIG").readlines()]
            poss = [ line.strip() for line in open(copor_dir+'\\'+sub+'\\pos.txt','r', encoding="UTF-8-SIG").readlines()]
            n_df = pd.DataFrame({'text':negs}); n_df['label'] = 'negative'
            p_df = pd.DataFrame({'text':poss}); p_df['label'] = 'positive'

            self.label_text_df = self.label_text_df.append( n_df )
            self.label_text_df = self.label_text_df.append( p_df )
        return

    def load_news_sample(self, news_sample_file=''):
            tmp_df         = pd.read_csv(news_sample_file)
            tmp_df['label']= tmp_df['label'].apply( lambda x: {'0':'positive','1':'neutral','2':'negative'}[str(x)] )
            self.label_text_df = self.label_text_df.append(tmp_df[['text','label']])
            return

class StksBasic():
    def __init__(self, file_path='./data/stk_basic.csv'):
        print('初始化方法')
        import pandas as pd
        self.file_path      = file_path
        stk_basic_df        = pd.read_csv( file_path )
        stk_basic_df['code']= stk_basic_df['code'].apply( lambda x: str(x)[0:6])
        self.name_ind_dic  =  dict(zip(stk_basic_df['name'], stk_basic_df['industry'])) #stk_basic_df[['name','industry']].to_dict()
        self.name_code_dic =  dict(zip(stk_basic_df['name'], stk_basic_df['code'])) #stk_basic_df[['name', 'code']].to_dict()
        self.code_name = dict(zip(stk_basic_df['code'],stk_basic_df['name'] ))
        self.stks_name = set(  self.name_ind_dic.keys() )
        self.industry_name = set(self.name_ind_dic.values())
        self.concept_name = set( self.name_code_dic.values())
        self.stk_concept={}
        print( 'name_ind_dic', self.name_ind_dic )
        print( 'name_code_dic',self.name_code_dic )

    def update_stk_concept(self,stk_concept_file):
        stk_basic_df = pd.read_csv(file_path)
        self.stk_concept =  dict(zip(stk_basic_df['stk'],stk_basic_df['concept'] ))
        return



if __name__ == '__main__':
    exit()
    stk_basic = StksBasic( file_path='J:\\work\\computational_Finance\\data\\stk_basic.csv' )
    exit()
    pos_neg = PosNegDic( file_path='J:\\work\\computational_Finance\\data\\neg_pos.csv')
    print( pos_neg.pos  )
    print( pos_neg.neg  )

    # new_file = '../python/data/aggregated_news_data.csv', sep = '|' ):
    # df = pd.read_csv(new_file, sep=sep)
    # df = df[df['Title'].str.contains('<p>') == False]
    # print(df[['Date', 'Title', 'Keywords', 'Summary']])

    exit( )
    test2(   )
    #sklearn_TfidfVectorizer( )

    exit()
    import time
    import jieba
    print( word_tokenize('this is a good example.!'))
    print( list( jieba.cut('我们怎么样啊') ) )

    exit()

    values = word_to_integer(document)

    num_topics = 15
    texts = pd.read_excel('past.xlsx', encoding='utf-8', header=None).astype(str)
    flags = ['n', 'nr', 'v', 'ns', 'nt', 'd']
    add_stoplist = ['知道', '了解', '感受', '矿泉水', '公司', '东北', '案例', '包装', '作者', '看到', '转载', '商品', '日本', '消费者', '心理', '使用', '越来越', '购买', '一定', '个人', '企业', '没', '做', '说', '非常', '可能', '原', '美国', '中国', '产品', '问题']
    whitelist = ['iPhone 11', '泸州老窖']
    data = deal_row_text(texts, add_stoplist, whitelist, flags)
    # record_data(data)
    # model_result(num_topics=num_topics, passes=100)



# from gensim import corpora
# from gensim.models import LdaModel
# from gensim import models
# from gensim.corpora import Dictionary
# import pandas as pd
#
# from collections import defaultdict
# # 词袋编码
# def record_data(data):
#     global dictionary, corpus, corpus_tfidf
#     dictionary = corpora.Dictionary(data)
#     dictionary.filter_extremes(no_below=3, no_above=0.4, keep_n=500)
#     corpus = [dictionary.doc2bow(text) for text in data]
#     tfidf = models.TfidfModel(corpus)
#     corpus_tfidf = tfidf[corpus]


# # 训练模型
# def model_result(num_topics=5, passes=100):
#     global lda, doc_topic
#     lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)
#     doc_topic = [a for a in lda[corpus]]
#     return doc_topic


# # 标准化结果展示
# def format_result(on_of=False, num_topics=5, num_words=5):
#     key_word = []
#     on_of = True
#     for i in lda.show_topics(num_topics=num_topics, num_words=num_words, log=False, formatted=False):
#         word = []
#         for j in i[1]:
#             word.append(j[0])
#             if on_of == True:
#                 word.append(j[1])
#         key_word.append(word)
#     return key_word
# # 文章分类
# def classify_article(num_topics):
#     topic_list = [i for i in range(num_topics)]
#     article_class = defaultdict()
#     for i in topic_list:
#         article_class[i] = []
#     for i in range(len(doc_topic)):
#         doc_topic[i].sort(key=lambda x: x[1], reverse=True)
#         article_class[ topic_list[ doc_topic[i][0][0]] ].append(texts.iloc[i, 0])
#     return article_class


# #不知道干嘛的
# def CountVectorizer(  corpus  ):
#     # corpus  也可以称作语料库 what is coupus， 里面是词：编码
#     #输出是词袋(词袋就是每个word 出现的次数) ； 词袋的问题是生成的特征特比长，很稀疏，不利于模型进行判断处理
#     from sklearn.feature_extraction.text import CountVectorizer
#     vectorizer  = CountVectorizer()
#     count_freq  = vectorizer.fit_transform( corpus )   # 转化成为词频统计 [ (0,1) (0,2)
#     #vectorizer.get_feature_names() 可以得到feature key, vectorizer.vocabulary_可以到key value的map
#     # x.toarray( )变成词袋 # [ [1,2,1,2],  [1,1,1,1]  ]
#     return count_freq, count_freq.toarray()


# import numpy as np
# # 生成词云
# def show_word_cloud(seged_docs):
#     from gensim import corpora
#     # 需要清楚的标点符号
#     left_words = ['.', ',', '?', '!', ';', ':', '\'', '(', ')']
#     # 生成字典
#     dic = corpora.Dictionary([seged_docs])
#     print( 'dic.items:', list( dic.items()) )
#     # 计算得到每个单词的使用频率
#     words_set = dic.doc2bow(document)
#     print('words_set:',  words_set )
#     # 生成单词列表和使用频率列表
#     words, frequences = [], []
#     for item in words_set:
#         key = item[0]
#         frequence = item[1]
#         word = dic.get(key=key)
#         if word not in left_words:
#             words.append(word)
#             frequences.append(frequence)
#     # 使用pyecharts生成词云
#     word_cloud = WordCloud()#width=1000, height=620
#     word_zip= list( zip(words, frequences ) )
#     print( word_zip[:10]  )
#     word_cloud.add( '',words_set, shape='circle', word_size_range=[20, 100]) #name='Alice\'s word cloud', attr=words,
#     word_cloud.render()

# def gensim_lsi( docs ):
#     seg_docs = en_word_cut(docs)
#     fre = frequency_statas( seg_docs )
#     dictionary = corpora.Dictionary( seg_docs )
#     bow_corpus = [ dictionary.doc2bow(text) for text in seg_docs ]
#     corpus_tfidf, tfidf_model = gen_texts_tfid_withDict(seg_docs, dictionary)
#     vector     = bow2vec(corpus_tfidf, max(dictionary) + 1)
#     lsi        = gen_lsimodel(corpus_tfidf, dictionary, topcnt=50)
#     vec_lsi = [lsi[doc] for doc in corpus_tfidf]
#     return  vec_lsi