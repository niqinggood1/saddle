#!/usr/local/bin/python
#coding=utf8
import pandas as pd
import numpy as np
import sys

def text_emotion_data_processing(filename='./text_emotion/train.csv'):
    train_file = open(filename) 
    data_map={'id':[],'review':[],'label':[]}
    str_buf =''
    old_line=''
    for line in train_file:
        line=line.replace('|','')
        if (line.strip()=='ID,review,label') or line.strip()=='ID,review':    
            str_buf+=line.replace(',','|')
            continue
        arg_list = line.split(',')
        if  arg_list[0].isdigit():
            
            str_buf+= old_line 
            old_line= line.replace( arg_list[0]+',', arg_list[0]+'|').replace(',Negative','|Negative').replace(',Positive','|Positive')
        else:
            old_line=old_line.strip()+ line.replace(',Negative','|Negative').replace(',Positive','|Positive')
    str_buf+= old_line
    train_file = open(filename+'tmp',"w")
    #train_file.write('id|review|label\n')
    train_file.write(str_buf)
    train_file.close()
    
#text_emotion_data_processing(filename='./text_emotion/20190610_test.csv')


##data processing 
#train_file = open('./text_emotion/train.csv') 
#data_map={'id':[],'review':[],'label':[]}
#str_buf =''
#old_line=''
#for line in train_file:
    #line=line.replace('|','')
    #if (line.strip()=='ID,review,label'):    
        ##str_buf+=line
        #continue
    
    #arg_list = line.split(',')
    #if  arg_list[0].isdigit():
        #str_buf+= old_line 
        #old_line= line.replace( arg_list[0]+',', arg_list[0]+'|').replace(',Negative','|Negative').replace(',Positive','|Positive')
    #else:
        #old_line=old_line.strip()+ line.replace(',Negative','|Negative').replace(',Positive','|Positive')
    
#train_file = open('./text_emotion/train_tmp.csv',"w")
#train_file.write('id|review|label\n')
#train_file.write(str_buf)
#train_file.close()
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy  as np
df = pd.read_csv('./text_emotion/train_tmp.csv',sep='|',encoding='utf8',lineterminator='\n').rename(columns={'label\r':'label'})
df['label']=df['label'].apply(lambda x:x.strip())
print df.head()

words_set=set()
for review in df['review'].tolist():
    tmp_set = set( review.strip().split(' ') )
    words_set = words_set|tmp_set-set(u'')

print len( words_set )
print words_set

words_set=words_set-set(u'') ;print len(words_set)
words_list = list(words_set)
len_words_list=len(words_list)
words_idx_dic={}
for k in xrange(len_words_list):
    words_idx_dic[ words_list[k] ]=k

featues_list=[]
for review in df['review'].tolist():
    tmp_set = set(  review.strip().split(' ') )
    feature = np.zeros( len_words_list )
    for k in tmp_set:
        idx = words_idx_dic[k]
        feature[idx] =1
    featues_list.append(feature)   
print featues_list[0]
print featues_list[1]
print len(  featues_list)

y =  df['label'].apply( lambda x:x.strip() ).apply( lambda x:  1 if x=='Positive' else 0)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(np.array(featues_list), y, test_size = 0.1, random_state = 0)


#lr
lm = LogisticRegression(penalty='l2',C=0.05) # logestic model construction
lm.fit(x_train,y_train)  # fitting the data
lr_y_train_pred = lm.predict_proba(x_train)
print 'LR train text emotion confusion_matrix:',    confusion_matrix(y_train, [1 if i[1]>0.5 else 0 for i in lr_y_train_pred ], labels=[0, 1]) 
print 'LR train text emotion roc and auc:',    roc_auc_score(y_train, [ i[1] for i in lr_y_train_pred ]) 

lr_y_test_pred = lm.predict_proba(x_test)
print 'LR test text emotion confusion_matrix:',    confusion_matrix(y_test, [1 if i[1]>0.5 else 0 for i in lr_y_test_pred ], labels=[0, 1]) 
print 'LR test text emotion roc and auc:',    roc_auc_score(y_test, [ i[1] for i in lr_y_test_pred ]) 

exit()
#bayes
from sklearn.naive_bayes import GaussianNB,BernoulliNB 
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB(alpha=0.5)  #GaussianNB()
clf = clf.fit(x_train,y_train) 
nb_y_train_pred=clf.predict_proba( x_train )
nb_y_test_pred=clf.predict_proba( x_test )

print nb_y_test_pred[:10]
print 'MutlnomialNB naive_bayes train text emotion confusion_matrix:',    confusion_matrix(y_train, [1 if i[1] >0.5 else 0 for i in nb_y_train_pred ], labels=[0, 1]) 
print 'MutlnomialNB naive_bayes train text emotion roc and auc:',    roc_auc_score(y_train,[ i[1] for i in  nb_y_train_pred ])

print 'MutlnomialNB naive_bayes test text emotion confusion_matrix:',    confusion_matrix(y_test, [1 if i[1] >0.5 else 0 for i in nb_y_test_pred ], labels=[0, 1]) 
print 'MutlnomialNB naive_bayes test text emotion roc and auc:',     roc_auc_score(y_test, [ i[1] for i in  nb_y_test_pred ])

eval_df = pd.read_csv('./text_emotion/20190610_test.csvtmp',sep='|')
eval_features=list()
   
for review in eval_df['review'].tolist():
    tmp_set = set(  review.strip().split(' ') )
    feature = np.zeros( len_words_list )
    for k in tmp_set:
        if k not in words_idx_dic:
            continue
        idx = words_idx_dic[k]
        feature[idx] =1
    eval_features.append(feature) 

bayes_pred = clf.predict_proba( np.array(eval_features)  )
eval_df['Pred'] = [i[1] for i in  bayes_pred]
eval_df['ID'] =  eval_df['ID']
eval_df[[ 'ID','Pred'] ].to_csv('patricks_MultinomialNB_text_emotion_pred.csv',index=False)


import numpy as np
for alpha in np.arange(0.1,3,0.1):
    clf = MultinomialNB(alpha=alpha)  #GaussianNB()
    clf = clf.fit(x_train,y_train) 
    nb_y_train_pred=clf.predict( x_train )
    nb_y_test_pred=clf.predict_proba( x_test )  
    print 'multinomianLNB alpha %.2f'%alpha, 'test auc:',roc_auc_score(y_test, [ i[1] for i in  nb_y_test_pred ] ) 
    
 
####
clf = BernoulliNB(alpha=1.0, binarize=.0, fit_prior=True, class_prior=None)  #GaussianNB()
clf = clf.fit(x_train,y_train) 
nb_y_train_pred=clf.predict( x_train )
nb_y_test_pred=clf.predict( x_test )

print nb_y_test_pred[:10]


print 'BernoulliNB naive_bayes train text emotion confusion_matrix:',    confusion_matrix(y_train, [1 if i >0.5 else 0 for i in nb_y_train_pred ], labels=[0, 1]) 
print 'BernoulliNB naive_bayes train text emotion roc and auc:',    roc_auc_score(y_train, nb_y_train_pred)

print 'BernoulliNB naive_bayes test text emotion confusion_matrix:',    confusion_matrix(y_test, [1 if i >0.5 else 0 for i in nb_y_test_pred ], labels=[0, 1]) 
print 'BernoulliNB naive_bayes test text emotion roc and auc:',     roc_auc_score(y_test, nb_y_test_pred)

 

import xgboost as xgb 
xlf = xgb.XGBClassifier(max_depth=10,  learning_rate=0.1,  n_estimators=101,silent=True, objective='reg:linear',nthread=-1, 
	                    gamma=0,	 min_child_weight=1, max_delta_step=0, subsample=0.85,  colsample_bytree=0.7,  colsample_bylevel=1, 
	                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=1440, missing=None)    

xlf.fit( x_train  , y_train, eval_metric='auc')
xgb_ = xlf.predict( x_train )
xgb_test_pred = xlf.predict( x_test )


 
print 'xgboost train text emotion confusion_matrix:',    confusion_matrix(y_train, [1 if i>0.5 else 0 for i in xgb_ ], labels=[0, 1]) 
print 'xgboost train text emotion roc and auc:',    roc_auc_score(y_train, xgb_ )  

print 'xgboost test text emotion confusion_matrix:',    confusion_matrix(y_test, [1 if i>0.5 else 0 for i in xgb_test_pred ], labels=[0, 1]) 
print 'xgboost test text emotion roc and auc:',    roc_auc_score(y_test, xgb_test_pred ) 

# RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=10, random_state=0,n_estimators=101)
regr.fit( x_train  , y_train)
rf_train_pred = regr.predict( x_train )
rf_test_pred  = regr.predict( x_test )

print 'rf train text emotion confusion_matrix:',    confusion_matrix(y_train, [1 if i>0.5 else 0 for i in rf_train_pred ], labels=[0, 1]) 
print 'rf train  text emotion roc and auc:',    roc_auc_score(y_train, rf_train_pred )  

print 'rf test text emotion confusion_matrix:',    confusion_matrix(y_test, [1 if i>0.5 else 0 for i in rf_test_pred ], labels=[0, 1]) 
print 'rf test text emotion roc and auc:',    roc_auc_score(y_test, rf_test_pred )  

###ligtGBM to much dimension
import lightgbm as lgb
lgb_train = lgb.Dataset( np.array(x_train), y_train)
#lgb_eval  = lgb.Dataset(X_test, y_test, reference=lgb_train)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss'},
    'num_leaves': 64,
    'num_trees': 100,
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
num_leaf = 64

print('Start training...')
gbm = lgb.train(params, lgb_train, num_boost_round=100,  valid_sets=lgb_train)
gbm_y_pred_train = gbm.predict(x_train) 


print 'gbm train sample pred',gbm_y_pred_train[:4]
print 'light GBM text emotion confusion_matrix:', confusion_matrix(y_train, [1 if i>0.5 else 0 for i in gbm_y_pred_train ], labels=[0, 1])
print 'light GBM  text emotion roc and auc:',    roc_auc_score(y_train, gbm_y_pred_train )  

gbm_y_pred_test = gbm.predict(x_test) 
print 'light GBM text emotion confusion_matrix:', confusion_matrix(y_test, [1 if i>0.5 else 0 for i in gbm_y_pred_test ], labels=[0, 1])
print 'light GBM  text emotion roc and auc:',    roc_auc_score(y_test, gbm_y_pred_test )  


y_pred = gbm.predict(x_train, pred_leaf=True)
transformed_training_matrix = np.zeros([len(y_pred), len(y_pred[0]) * num_leaf],
                                       dtype=np.int64)  # N * num_tress * num_leafs
for i in range(0, len(y_pred)):
    temp = np.arange(len(y_pred[0])) * num_leaf + np.array(y_pred[i])
    transformed_training_matrix[i][temp] += 1
print 'transformed_training_matrix0',transformed_training_matrix[0]
lm = LogisticRegression(penalty='l2',C=0.05) # logestic model construction
lm.fit(transformed_training_matrix,y_train)  # fitting the data
gbdtlr_y_train_pred = lm.predict_proba(transformed_training_matrix)   # Give the probabilty on each label

y_tmp = gbm.predict(x_test, pred_leaf=True)
transformed_testing_matrix = np.zeros([len(y_tmp), len(y_tmp[0]) * num_leaf], dtype=np.int64)
for i in range(0, len(y_tmp)):
    temp = np.arange(len(y_tmp[0])) * num_leaf + np.array(y_tmp[i])
    transformed_testing_matrix[i][temp] += 1

gbdtlr_y_test_pred = lm.predict_proba(transformed_testing_matrix)


print 'GBDT+LR train text emotion confusion_matrix:',    confusion_matrix(y_train, [1 if i[1]>0.5 else 0 for i in gbdtlr_y_train_pred ], labels=[0, 1]) 
print 'GBDT+LR train text emotion roc and auc:',    roc_auc_score(y_train, [ i[1] for i in gbdtlr_y_train_pred ])  

 
print 'GBDT+LR test text emotion confusion_matrix:',    confusion_matrix(y_test, [1 if i[1]>0.5 else 0 for i in gbdtlr_y_test_pred ], labels=[0, 1]) 
print 'GBDT+LR test text emotion roc and auc:',    roc_auc_score(y_test, [ i[1] for i in gbdtlr_y_test_pred ]) 

print '############'*10,'train data:'

#rf_train_pred = xlf.predict( x_train )
#rf_test_pred  = xlf.predict( x_test )

print 'GBDT+LR text emotion confusion_matrix:',    confusion_matrix(y_train, [1 if i[1]>0.5 else 0 for i in gbdtlr_y_train_pred ], labels=[0, 1]) 
print 'GBDT+LR text emotion roc and auc:',    roc_auc_score(y_train, [ i[1] for i in gbdtlr_y_train_pred ] )  


print '****'*10
from pyfm import pylibfm
from scipy.sparse import csr_matrix
fm = pylibfm.FM()
fm.fit(      csr_matrix(  x_train ),  y_train.values  )
fm_y_train_pred = fm.predict(  csr_matrix( x_train )  )

print 'fm train sample pred',fm_y_train_pred[:4]
print 'FM  train text emotion confusion_matrix:',    confusion_matrix(y_train, [1 if i>0.5 else 0 for i in fm_y_train_pred ], labels=[0, 1])
print 'FM  train text emotion roc and auc:',    roc_auc_score(y_train,  fm_y_train_pred )

fm_y_train_pred = fm.predict(  csr_matrix(  x_test )  )
print 'FM  test text emotion confusion_matrix:',    confusion_matrix(y_test, [1 if i>0.5 else 0 for i in fm_y_train_pred ], labels=[0, 1])
print 'FM  test text emotion roc and auc:',    roc_auc_score(y_test,  fm_y_train_pred )


#predict 

eval_df = pd.read_csv('./text_emotion/20190610_test.csvtmp',sep='|')
eval_features=list()
   
for review in eval_df['review'].tolist():
    tmp_set = set(  review.strip().split(' ') )
    feature = np.zeros( len_words_list )
    for k in tmp_set:
        if k not in words_idx_dic:
            continue
        idx = words_idx_dic[k]
        feature[idx] =1
    eval_features.append(feature) 

gbm_y_pred_eval = gbm.predict( np.array(eval_features)  )

eval_df['Pred'] = gbm_y_pred_eval
eval_df['ID'] =  eval_df['ID']
eval_df[[ 'ID','Pred'] ].to_csv('patricks_text_emotion_pred.csv',index=False)


fm_y_eval_pred = fm.predict(  csr_matrix( x_train )  )

print 'finished'




exit()  

#data processing  20190610_test.csv



   
 