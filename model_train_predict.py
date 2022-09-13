#!/usr/local/bin/python
#coding=utf8
import datetime
import joblib

from sklearn import tree
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn  import  linear_model#LinearRegression
# from sklearn.linear_model import Ridge
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import ElasticNet
from xgboost  import XGBRegressor
from lightgbm import LGBMRegressor

def build_regressor_model(name='RandomForestRegressor'):
    if name == 'LR' :   model = linear_model.LinearRegression()
    if name == 'Lasso':    model = linear_model.Lasso()  ### 线性回归 ###
    if name == 'Ridge':    model = linear_model.Ridge()  ### 线性回归 ###
    if name == 'ElasticNet':model = linear_model.ElasticNet()  ### 线性回归 ###
    if name == 'DT' :   model = tree.DecisionTreeRegressor(  ) #criterion="mae"
    if name == 'SVM':   model = svm.SVR()                        ### SVM回归 ###
    if name == 'KNN':   model = neighbors.KNeighborsRegressor() ### KNN回归 ###
    if name == 'RF':    model = ensemble.RandomForestRegressor(n_estimators=21)  # 用20个决策树 ### 随机森林回归 ###
    if name == 'AbaR':  model = ensemble.AdaBoostRegressor(n_estimators=50)  # 用50个决策树### Adaboost回归 ###
    if name == 'GB':    model = ensemble.GradientBoostingRegressor(n_estimators=100)  # 用100个决策树 ### GBRT回归 ###
    if name == 'BR':    model = BaggingRegressor() ### Bagging回归 ###
    if name == 'ER':    model = ExtraTreeRegressor()    ### ExtraTree极端随机树回归 ###
    if name == 'Xgboost':
        model = XGBRegressor(max_depth=3,n_estimators=101,learning_rate=0.1)
    if name == 'LightGBM':   model = LGBMRegressor(n_estimators=100)
    return model

# from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import BernoulliRBM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB,BernoulliNB

from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier
import lightgbm as lgb
def build_classifier_model(name):
    if name == 'NearestNeighbors':  model = KNeighborsClassifier( 7 )
    if name == 'MLP':               model = MLPClassifier(hidden_layer_sizes=(50,50),
                                                          activation='relu',max_iter=300,alpha=0.01)
    if name == 'LR':                model = LogisticRegression() #penalty='l1'
    if name == 'LinearSVM':         model =SVC(kernel="linear", C=0.025)
    if name == 'RBFSVM':            model =SVC(kernel='rbf') #,gamma=2, C=1
    if name == 'DecisionTree':      model =DecisionTreeClassifier(max_depth=5)
    if name == 'NaiveBayes':        model =GaussianNB()
    if name == 'ER':                model =QuadraticDiscriminantAnalysis()
    if name == 'GaussianProcess':   model =GaussianProcessClassifier()
    if name == 'BernoulliNB':       model =BernoulliNB()
    if name == 'MultinomialNB':     model =MultinomialNB(alpha=1,class_prior=None, fit_prior=True)
    if name == 'RandomForest':      model =ensemble.RandomForestClassifier(max_depth=5, n_estimators=101 )
    if name == 'AdaBoost':          model =ensemble.AdaBoostClassifier()
    if name =='GradientBoosting':   model =GradientBoostingClassifier()
    if name == 'Xgboost':
        model = XGBClassifier(max_depth=6, learning_rate=0.05,n_estimators=101,nthread=-1,objective='reg:logistic'
                              ,reg_alpha=0.04,gamma=0.02,min_child_weight=11) #objective ='reg:squarederror',  silent=0,objective='reg:linear'
        # gamma=0,min_child_weight=1, max_delta_step=0, subsample=0.85,colsample_bytree=0.7,colsample_bylevel=1,
        # reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=1440, missing=None
    if name =='LightGBM':   model = LGBMClassifier() #objective='binary',metrics='binary_logloss',n_estimators=101
    # if name =='fm':          model = pylibfm.FM()
    return model


def gen_model_path( model_file ):
    if model_file == '':
        todaystr = str(datetime.date.today().date())
        model_file = './data/regressor_%s.m' % todaystr
    return model_file

def model_fit(model,x_train, y_train):
    model_fit = model.fit(x_train, y_train)
    return model_fit

def build_train_model(x_train, y_train, name='RandomForestRegressor',type='classifier'):
    if type in ['cl','classifier' ]:
        model = build_classifier_model(name )  # # 指定模型
    if type in ['reg','regressor' ]:
        model = build_regressor_model( name )  # # 指定模型
    model.fit(x_train, y_train)
    return model

from sklearn import model_selection
def multi_model_test(x,y,n_splits=7,score='accuracy'): #f1
    names  =   [  #'LR','DecisionTree','NaiveBayes','ER','GaussianProcess',
    'BernoulliNB', 'RandomForest',\
    'AdaBoost','GradientBoosting','Xgboost','LightGBM','NearestNeighbors','MLP','LinearSVM','RBFSVM' ] #'MultinomialNB',
    models=[]
    for k in names:
        tmp_model = build_classifier_model(  k  )
        models.append( tmp_model  )
    results=[]
    for idx in range( len(names) ):
        kfold       = model_selection.KFold( n_splits= n_splits)
        cv_results  = model_selection.cross_val_score( models[idx] ,x, y,cv=kfold,scoring=score )#accuray  f1
        results.append( cv_results )
        if  names[idx] =='xgb':
            print('xgb:',cv_results)
        msg= "%s: %.3f(%.3f)" %( names[idx] , round(np.median(  cv_results ),3) ,round(cv_results.std() ,3)    )
        print(  msg )

    import matplotlib.pyplot as plt
    fig = plt.figure( figsize=(6,6) )
    fig.suptitle( 'Algorithm(Comparison)' )
    ax = fig.add_subplot(111)
    plt.boxplot( results )
    ax.set_xticklabels(names,rotation=15,fontsize=12)
    plt.show()

    return



#这个没啥意义啊
from  . model_eval  import model_evl
from sklearn.model_selection import train_test_split
def train_evl_model_(train_df,features=[], target='', postive_mutiply=3,modelname='RandomForestRegressor' ,model_file='tmp.model',type='regressor' ):
    print( 'target:', target,'feature:', features )
    model_file = gen_model_path(model_file )
    x_train,x_test, y_train,y_test = train_test_split( train_df[features].values, train_df[target], test_size=0.3, random_state=0)

    model   = build_train_model( x_train,y_train,name=modelname,type=type )
    p_test  = model.predict(  x_test     )
    report  = model_evl(type,y_test,p_test )
    joblib.dump( model, model_file)
    return  model

def get_lr_model_weight(var_woe_name,LR_model_fit):
    ###保存模型的参数用于计算评分,基于做了拟合后的LR模型
    var_woe_name.append('intercept')
    ##提取权重
    weight_value = list(LR_model_fit.coef_.flatten()) #flatten 返回一维数组
    ##提取截距项
    weight_value.extend(list(LR_model_fit.intercept_))
    dict_params     = dict(zip(var_woe_name, weight_value))
    return dict_params

#accuracy_score函数利用 y_test 和 y_predict 计算出得分  score(X_test,y_test) score是自动预测计算得分
from io import StringIO
import pandas  as pd
import numpy   as np



#要区分 回归还是分类
def model_predict( df, feature_list=[], model_file=''):
    model = joblib.load(model_file)
    features = df[feature_list].as_matrix()
    pred = model.predict(features)
    return pred

default_lr_parm={'C': [0.01, 0.1, 0.2, 0.5, 1, 1.5, 2],
                 'class_weight': [{1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}]}
def grib_lr_search(x_train,y_train,lr_param = default_lr_parm):
    ########logistic模型
    ##初始化网格搜索
    lr_gsearch = GridSearchCV(
        estimator=LogisticRegression(random_state=0, fit_intercept=True, penalty='l2', solver='saga'),
        param_grid=lr_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)
    ##执行超参数优化
    lr_gsearch.fit(x_train, y_train)
    print('logistic model best_score_ is {0},and best_params_ is {1}'.format(lr_gsearch.best_score_,
                                                                             lr_gsearch.best_params_))
    LR_model = LogisticRegression(C=lr_gsearch.best_params_['C'], penalty='l2', solver='saga',
                                  class_weight=lr_gsearch.best_params_['class_weight'])
    return lr_gsearch, LR_model

def grib_dt_search(x_train,y_train):
    ########决策树模型
    ##设置待优化的超参数
    DT_param = {'max_depth': np.arange(2, 10, 1),
                'class_weight': [{1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}]}
    ##初始化网格搜索
    DT_gsearch = GridSearchCV(
        estimator=DecisionTreeClassifier(),
        param_grid=DT_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)
    ##执行超参数优化
    DT_gsearch.fit(x_train, y_train)
    print('DecisionTreeClassifier model best_score_ is {0},and best_params_ is {1}'.format(DT_gsearch.best_score_,
                                                                                           DT_gsearch.best_params_))

    DT_model_1 = DecisionTreeClassifier(max_depth=DT_gsearch.best_params_['max_depth'],
                                        class_weight=DT_gsearch.best_params_['class_weight'])
    ##训练决策树模型
    # DT_model_fit = DT_model_1.fit(x_train, y_train)
    return DT_model_1,DT_gsearch

def grib_lineSVM_search(x_train,y_train):
    ########线性支持向量机模型
    ##设置待优化的超参数
    lin_svm_param = {'C': np.arange(0.1, 5, 0.1),
                     'class_weight': [{1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}]}
    ##初始化网格搜索
    lin_svm_gsearch = GridSearchCV(estimator=LinearSVC(), param_grid=lin_svm_param,
                                   cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
    ##执行超参数优化
    lin_svm_gsearch.fit(x_train, y_train)
    print('linearSVC model best_score_ is {0},and best_params_ is {1}'.format(lin_svm_gsearch.best_score_,
                                                                              lin_svm_gsearch.best_params_))

    ##用最优参数，初始化模型
    lin_svm_model = LinearSVC(C=lin_svm_gsearch.best_params_['C'],
                              class_weight=lin_svm_gsearch.best_params_['class_weight'])
    ##模型训练
    lin_svm_model_fit = lin_svm_model.fit(x_train, y_train)
    return lin_svm_model,lin_svm_gsearch


def grib_RbfSVM_search(x_train,y_train):
    #########非线性支持向量机模型
    ##设置待优化的超参数
    svm_param = {'C': np.arange(0.1, 5, 0.1),
                 'gamma': np.arange(0.01, 10, 1),
                 'class_weight': [{1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}]}
    ##初始化网格搜索
    svm_gsearch = GridSearchCV(estimator=SVC(kernel='rbf'), param_grid=svm_param,
                               cv=3, scoring='roc_auc', n_jobs=-1, verbose=2)
    ##执行超参数优化
    svm_gsearch.fit(x_train, y_train)
    print('DecisionTreeClassifier model best_score_ is {0},and best_params_ is {1}'.format(svm_gsearch.best_score_,
                                                                                           svm_gsearch.best_params_))

    ##用最优参数，初始化模型
    svm_model = SVC(kernel='rbf', C=svm_gsearch.best_params_['C'], gamma=svm_gsearch.best_params_['gamma'],
                    class_weight=svm_gsearch.best_params_['class_weight'], probability=True)
    return svm_model,svm_gsearch

from sklearn.ensemble import RandomForestClassifier
def grib_RandomForest_search(x_train,y_train):
    ########随机森林模型
    ##设置待优化的超参数
    rf_param = {'n_estimators': list(range(50, 400, 50)),
                'max_depth': list(range(2, 10, 1)),
                'class_weight': [{1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}]}
    ##初始化网格搜索
    rf_gsearch = GridSearchCV(estimator=RandomForestClassifier(random_state=0, criterion='entropy',
                                                               max_features=0.8, bootstrap=True),
                              param_grid=rf_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)
    ##执行超参数优化
    rf_gsearch.fit(x_train, y_train)
    print('RandomForest model best_score_ is {0},and best_params_ is {1}'.format(rf_gsearch.best_score_,
                                                                                 rf_gsearch.best_params_))
    ##模型训练
    ##用最优参数，初始化随机森林模型
    RF_model_2 = RandomForestClassifier(random_state=0, n_jobs=-1, criterion='entropy',
                                        n_estimators=rf_gsearch.best_params_['n_estimators'],
                                        max_depth=rf_gsearch.best_params_['max_depth'],
                                        max_features=0.8,
                                        min_samples_split=50,
                                        class_weight=rf_gsearch.best_params_['class_weight'],
                                        bootstrap=True)
    return RF_model_2,rf_gsearch

from sklearn.ensemble import AdaBoostClassifier
def grib_adboost_search(x_train, y_train):
    ###Adaboost模型
    ##设置待优化的超参数
    ada_param = {'n_estimators': list(range(50, 500, 50)),
                 'learning_rate': list(np.arange(0.1, 1, 0.2))}
    ##初始化网格搜索
    ada_gsearch = GridSearchCV(estimator=AdaBoostClassifier(algorithm='SAMME.R', random_state=0),
                               param_grid=ada_param, cv=3, n_jobs=-1, verbose=2)
    ##执行超参数优化
    ada_gsearch.fit(x_train, y_train)
    print('AdaBoostClassifier model best_score_ is {0},and best_params_ is {1}'.format(ada_gsearch.best_score_,
                                                                                       ada_gsearch.best_params_))
    ##模型训练
    ##用最优参数，初始化Adaboost模型
    ada_model_2 = AdaBoostClassifier(n_estimators=ada_gsearch.best_params_['n_estimators'],
                                     learning_rate=ada_gsearch.best_params_['learning_rate'],
                                     algorithm='SAMME.R', random_state=0)
    return ada_model_2,ada_gsearch

def grib_gbdt_search(x_train, y_train):
    ####GBDT模型
    ##设置待优化的超参数
    gbdt_param = {'n_estimators': list(range(50, 400, 50)),
                  'max_depth': list(range(2, 5, 1)),
                  'learning_rate': list(np.arange(0.01, 0.5, 0.05))}
    ##初始化网格搜索
    gbdt_gsearch = GridSearchCV(estimator=GradientBoostingClassifier(subsample=0.8, max_features=0.8, validation_fraction=0.1,
                                                                     n_iter_no_change=3, random_state=0), param_grid=gbdt_param,
                                cv=3, scoring='f1', n_jobs=-1, verbose=2)
    ##执行超参数优化
    gbdt_gsearch.fit(x_train, y_train)
    print('gbdt model best_score_ is {0},and best_params_ is {1}'.format(gbdt_gsearch.best_score_,
                                                                         gbdt_gsearch.best_params_))
    ##模型训练
    ##用最优参数，初始化GBDT模型
    GBDT_model = GradientBoostingClassifier(subsample=0.8, max_features=0.8, validation_fraction=0.1,
                                            n_iter_no_change=3, random_state=0,
                                            n_estimators=gbdt_gsearch.best_params_['n_estimators'],
                                            max_depth=gbdt_gsearch.best_params_['max_depth'],
                                            learning_rate=gbdt_gsearch.best_params_['learning_rate'])
    return GBDT_model,gbdt_gsearch

from xgboost import XGBClassifier
def grib_xgboost_search(x_train,y_train):
    ###xgboost模型
    ##设置待优化的超参数
    xgb_param = {'max_depth': list(range(2, 6, 1)), 'min_child_weight': list(range(1, 4, 1)),
                 'learning_rate': list(np.arange(0.01, 0.3, 0.05)), 'scale_pos_weight': list(range(1, 5, 1))}
    ##初始化网格搜索
    xgb_gsearch = GridSearchCV(
        estimator=XGBClassifier(random_state=0, n_estimators=500, subsample=0.8, colsample_bytree=0.8),
        param_grid=xgb_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)
    ##执行超参数优化
    xgb_gsearch.fit(x_train, y_train)
    print('xgboost model best_score_ is {0},and best_params_ is {1}'.format(xgb_gsearch.best_score_,
                                                                            xgb_gsearch.best_params_))
    ##用最优参数，初始化xgboost模型
    xgboost_model = XGBClassifier(random_state=0, n_jobs=-1,
                                  n_estimators=500,
                                  max_depth=xgb_gsearch.best_params_['max_depth'],
                                  subsample=0.8, colsample_bytree=0.8,
                                  learning_rate=xgb_gsearch.best_params_['learning_rate'],
                                  scale_pos_weight=xgb_gsearch.best_params_['scale_pos_weight'])
    return xgboost_model,xgb_gsearch


def grib_lightGBM(x_train, y_train):
    grib_lightGBM_params = {'n_estimators':range(20,200,20),"max_depth":range(3,20,2),'num_leaves':range(3,20,2)}
    gbm = LGBMClassifier()
                             # max_depth = 6,num_leaves = 40,

    gsearch = GridSearchCV( gbm, param_grid=grib_lightGBM_params, scoring='accuracy', cv=17) #roc_auc
    gsearch.fit(x_train, y_train)
    print('The best params:{0}'.format(gsearch.best_params_))
    print('The best model score:{0}'.format(gsearch.best_score_))
    print(gsearch.cv_results_['mean_test_score'])
    print(gsearch.cv_results_['params'])

    ##用最优参数，初始化xgboost模型
    gbm = LGBMClassifier(objective = 'binary',
                         is_unbalance = True,
                         metric = 'binary_logloss,auc',
                         n_estimators =gsearch.best_params_['n_estimators'] ,
                         max_depth = gsearch.best_params_['max_depth'] , num_leaves = gsearch.best_params_['max_depth']   ,
                         learning_rate = 0.1,
                         feature_fraction = 0.7,
                         min_child_samples=21,min_child_weight=0.001,
                         bagging_fraction = 1,bagging_freq = 2,
                         reg_alpha = 0.001,reg_lambda = 8,
                         cat_smooth = 0,num_iterations = 200)
    return gbm,gsearch

def build_train_model_stack(x_train_temp, y_train_temp   ):
    RF_model_1 = RandomForestClassifier(random_state=0, n_jobs=-1, criterion='entropy',
                                        n_estimators=100,
                                        max_depth=3,
                                        max_features=0.8,
                                        min_samples_split=50,
                                        class_weight={1: 2, 0: 1},
                                        bootstrap=True)
    RF_model_fit_1 = RF_model_1.fit(x_train_temp, y_train_temp)

    ##xgboost模型
    xgboost_model_1 = XGBClassifier(random_state=0, n_jobs=-1,
                                    n_estimators=100,
                                    max_depth=2,
                                    min_child_weight=2,
                                    subsample=0.8,
                                    learning_rate=0.02,
                                    scale_pos_weight=2)
    xgboost_model_fit_1 = xgboost_model_1.fit(x_train_temp, y_train_temp)
    ##给出概率预测


    ##svm模型
    svm_model_1 = SVC(kernel='rbf', C=0.5, gamma=0.1, class_weight={1: 2, 0: 1}, probability=True)
    svm_model_fit_1 = svm_model_1.fit(x_train_temp, y_train_temp)
    ##给出概率预测
    return RF_model_fit_1, xgboost_model_fit_1, svm_model_fit_1




#y_pred_all = np.vstack([y_pred_1, y_pred_2,y_pred_3]).T

def lr_predict_proba(LR_model,x_train,x_test):
    ## ##模型预测
    y_score_train = LR_model.predict_proba(x_train)[:, 1]
    y_score_test  = LR_model.predict_proba(x_test )[:, 1]
    return y_score_train,y_score_test

##############  text  Model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def build_word_vector( word_list  ):
    vectorizer = CountVectorizer()                  #生成词典向量  也就是字符变数字
    X = vectorizer.fit_transform(word_list)  # word cnt
    #print  vectorizer.get_feature_names() #word list
    transformer = TfidfTransformer()       #该类变换Tfidf，字频变Tfid这个量
    tfidf = transformer.fit_transform(X)    #TfidfVectorizer 可以讲 countVectorizer 和 TfidfTransformer合起来
    return tfidf

################################### 深度学习
def build_net():
    from keras.models import Model
    from keras.layers import Input, LSTM, Dense
    from keras import backend as K

    num_encoder_tokens = 15
    max_item = 1000 * 10000

    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    query_em = Embedding(output_dim=512, input_dim=max_item, input_length=num_encoder_tokens)(encoder_inputs)
    lstm_out1 = LSTM(32)(query_em)

    x = Dense(64, activation='relu')(lstm_out1)
    x = Dense(64, activation='relu')(x)
    main_output = Dense(max_item, activation='softmax', name='main_output')(x)

    train_model = model(encoder_inputs, main_output)
    train_model.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

def tradation(iris):
    #iris = load_iris()  # 载入鸢尾花数据集
    data = iris.data
    target = iris.target
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    # 加载你的数据
    # print('Load data...')
    # df_train = pd.read_csv('../regression/regression.train', header=None, sep='\t')
    # df_test = pd.read_csv('../regression/regression.test', header=None, sep='\t')
    #
    # y_train = df_train[0].values
    # y_test = df_test[0].values
    # X_train = df_train.drop(0, axis=1).values
    # X_test = df_test.drop(0, axis=1).values

    # 创建成lgb特征的数据集格式
    lgb_train = lgb.Dataset(X_train, y_train)  # 将数据保存到LightGBM二进制文件将使加载更快
    lgb_eval  = lgb.Dataset(X_test, y_test, reference=lgb_train)  # 创建验证数据

    # 将参数写成字典下形式
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'regression',  # 目标函数
        'metric': {'l2', 'auc'},  # 评估函数
        'num_leaves': 31,  # 叶子节点数
        'learning_rate': 0.05,  # 学习速率
        'feature_fraction': 0.9,  # 建树的特征选择比例
        'bagging_fraction': 0.8,  # 建树的样本采样比例
        'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': 1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    print('Start training...')
    # 训练 cv and train
    gbm = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)  # 训练数据需要参数列表和数据集

    print('Save model...')

    gbm.save_model('model.txt')  # 训练后保存模型到文件

    print('Start predicting...')
    # 预测数据集
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)  # 如果在训练期间启用了早期停止，可以通过best_iteration方式从最佳迭代中获得预测
    # 评估模型
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)  # 计算真实值和预测值之间的均方根误差

    return

from sklearn.model_selection import KFold
def build_train_stack_model(x_train,y_train):
    ########stacking模型融合
    kfold = KFold(n_splits=5)
    ##存放概率预测结果
    y_pred_1 = np.zeros(x_train.shape[0])
    y_pred_2 = np.zeros(x_train.shape[0])
    y_pred_3 = np.zeros(x_train.shape[0])
    for train_index, test_index in kfold.split(x_train):
        ##K折中的建模部分
        x_train_temp = x_train[train_index, :]
        y_train_temp = y_train[train_index]
        ##K折中的预测部分
        x_pre_temp = x_train[test_index, :]
        y_pre_temp = y_train[test_index]

        ##一级模型
        ##随机森林模型
        RF_model_1 = RandomForestClassifier(random_state=0, n_jobs=-1, criterion='entropy',
                                            n_estimators=100,
                                            max_depth=3,
                                            max_features=0.8,
                                            min_samples_split=50,
                                            class_weight={1: 2, 0: 1},
                                            bootstrap=True)
        RF_model_fit_1 = RF_model_1.fit(x_train_temp, y_train_temp)
        ##给出概率预测
        y_pred_1[test_index] = RF_model_fit_1.predict_proba(x_pre_temp)[:, 1]

        ##xgboost模型
        xgboost_model_1 = XGBClassifier(random_state=0, n_jobs=-1,
                                        n_estimators=100,
                                        max_depth=2,
                                        min_child_weight=2,
                                        subsample=0.8, #colsample_bytree=0.8,
                                        learning_rate=0.02,
                                        scale_pos_weight=2)
        xgboost_model_fit_1 = xgboost_model_1.fit(x_train_temp, y_train_temp)
        ##给出概率预测
        y_pred_2[test_index] = xgboost_model_fit_1.predict_proba(x_pre_temp)[:, 1]

        ##svm模型
        svm_model_1 = SVC(kernel='rbf', C=0.5, gamma=0.1, class_weight={1: 2, 0: 1}, probability=True)
        svm_model_fit_1 = svm_model_1.fit(x_train_temp, y_train_temp)
        ##给出概率预测
        y_pred_3[test_index] = svm_model_fit_1.predict_proba(x_pre_temp)[:, 1]

    ##将预测概率输出与原特征合并，构造新的训练集
    y_pred_all = np.vstack([y_pred_1, y_pred_2, y_pred_3]).T
    x_train_1 = np.hstack([x_train, y_pred_all])
    ##训练二级模型
    ##设置待优化的超参数
    lr_param = {'C': [0.01, 0.1, 0.2, 0.5, 1, 1.5, 2],
                'class_weight': [{1: 1, 0: 1}, {1: 2, 0: 1}, {1: 3, 0: 1}]}
    ##初始化网格搜索
    lr_gsearch = GridSearchCV(
        estimator=LogisticRegression(random_state=0, fit_intercept=True, penalty='l2', solver='saga'),
        param_grid=lr_param, cv=3, scoring='f1', n_jobs=-1, verbose=2)
    ##执行超参数优化
    lr_gsearch.fit(x_train_1, y_train)
    print('logistic model best_score_ is {0},and best_params_ is {1}'.format(lr_gsearch.best_score_,
                                                                             lr_gsearch.best_params_))
    ##最有参数训练模型
    ##用最优参数，初始化Logistic回归模型
    LR_model_2 = LogisticRegression(C=lr_gsearch.best_params_['C'], penalty='l2', solver='saga',
                                    class_weight=lr_gsearch.best_params_['class_weight'])
    ##训练Logistic回归模型
    LR_model_fit = LR_model_2.fit(x_train_1, y_train)
    return  RF_model_fit_1,xgboost_model_fit_1,svm_model_fit_1,LR_model_fit

def stack_model_predict( model11, model12, model13, model2, x_test):
    ##先用一级模型进行概率预测 ,测试集预测结果    ##给出概率预测
    ytest_pred_1 = model11.predict_proba(x_test)[:, 1]
    ytest_pred_2 = model12.predict_proba(x_test)[:, 1]
    ytest_pred_3 = model13.predict_proba(x_test)[:, 1]
    ##构造新特征矩阵
    ytest_pred_all  = np.vstack([ytest_pred_1, ytest_pred_2, ytest_pred_3]).T
    x_test_1        = np.hstack([x_test, ytest_pred_all])
    y_pred          = model2.predict(x_test_1)
    y_pred_proba    = model2.predict_proba(x_test_1)[:, 1]
    return  y_pred,y_pred_proba

#########评分卡转为分数
def score_params_cal(base_point, odds, PDO):
    ##给定预期分数，与翻倍分数，确定参数A,B
    B = PDO / np.log(2)
    A = base_point + B * np.log(odds)
    return A, B
def myfunc(x):
    return str(x[0]) + '_' + str(x[1])

##生成评分卡 score =  A - Blog( Odds ) = A - B(wo + w1 + w2+ )
def create_score(dict_woe_map, dict_params, dict_cont_bin, dict_disc_bin):
    ##假设Odds在1:60时对应的参考分值为600分，分值调整刻度PDO为20，则计算得到分值转化的参数B = 28.85，A= 481.86。
    # dict_woe_map 为分箱的WOE值; dic_params逻辑回归后各变量的权重(包含截距); dict_cont_bin 连续变量分箱; dict_disc_bin离散变量分箱
    #返回，df_score 是各个变量的得分
    # dict_bin_score,分箱和对应的score
    params_A, params_B = score_params_cal(base_point=600, odds=1 / 60, PDO=20)
    # 计算基础分
    base_points = round(params_A - params_B * dict_params['intercept'])
    df_score = pd.DataFrame()
    dict_bin_score = {}
    for k in dict_params.keys():
        #        k='duration_BIN'
        #        k = 'foreign_worker_BIN'
        if k != 'intercept':
            df_temp = pd.DataFrame([dict_woe_map[ k.split(sep='_woe')[0] ]] ).T  # 变成了 列是 bin, woe值
            df_temp.reset_index(inplace=True)
            df_temp.columns = ['bin', 'woe_val']
            ##计算分值
            df_temp['score'] = round(-params_B * df_temp.woe_val * dict_params[k])
            dict_bin_score[k.split(sep='_BIN')[0]] = dict( zip(df_temp['bin'], df_temp['score']) )
            ##连续变量的计算
            if k.split(sep='_BIN')[0] in dict_cont_bin.keys():
                df_1 = dict_cont_bin[k.split(sep='_BIN')[0]]
                df_1['var_name'] = df_1[['bin_low', 'bin_up']].apply(myfunc, axis=1)
                df_1 = df_1[['total', 'var_name']]
                df_temp = pd.merge(df_temp, df_1, on='bin')
                df_temp['var_name_raw'] = k.split(sep='_BIN')[0]
                df_score = pd.concat([df_score, df_temp], axis=0)  # 这里是拼接上范围
            ##离散变量的计算
            elif k.split(sep='_BIN')[0] in dict_disc_bin.keys():
                df_temp = pd.merge(df_temp, dict_disc_bin[k.split(sep='_BIN')[0]], on='bin')
                df_temp['var_name_raw'] = k.split(sep='_BIN')[0]
                df_score = pd.concat([df_score, df_temp], axis=0) # 这里也是拼接上范围

    df_score['score_base'] = base_points
    return df_score, dict_bin_score, params_A, params_B, base_points

from . import  variable_bin_methods  as varbin_meth
##计算样本分数
def cal_score(df_1, dict_bin_score, dict_cont_bin, dict_disc_bin, base_points):
    ##先对原始数据分箱映射，然后，用分数字典dict_bin_score映射分数，基础分加每项的分数就是最终得分
    df_1.reset_index(drop=True, inplace=True)
    df_all_score = pd.DataFrame()
    ##连续变量分箱
    for i in dict_cont_bin.keys():
        if i in dict_bin_score.keys():
            df_all_score = pd.concat([df_all_score, varbin_meth.cont_var_bin_map( df_1[i], dict_cont_bin[i]).map(dict_bin_score[i])], axis=1)
    ##离散变量分箱
    for i in dict_disc_bin.keys():
        if i in dict_bin_score.keys():
            df_all_score = pd.concat([df_all_score, varbin_meth.disc_var_bin_map(df_1[i], dict_disc_bin[i]).map(dict_bin_score[i])], axis=1)

    df_all_score.columns = [x.split(sep='_BIN')[0] for x in list(df_all_score.columns)]
    df_all_score['base_score'] = base_points
    df_all_score['score'] = df_all_score.apply(sum, axis=1)
    df_all_score['target'] = df_1.target
    return df_all_score


def score_statas( df_all_score ):
    df_all_score.score.max()
    df_all_score.score.min()
    ##简单的分数区间计算
    score_bin = np.arange(330, 660, 30)
    good_total = sum(df_all_score.target == 0)
    bad_total = sum(df_all_score.target == 1)
    bin_rate = []
    bad_rate = []
    ks = []
    good_num = []
    bad_num = []
    score_index=[]
    for i in range(len(score_bin) - 1):
        score_index.append( '%d-%d'%(score_bin[i] ,score_bin[i + 1]) )
        ##取出分数区间的样本
        if score_bin[i + 1] == 900:
            index_1 = (df_all_score.score >= score_bin[i]) & (df_all_score.score <= score_bin[i + 1])
        else:
            index_1 = (df_all_score.score >= score_bin[i]) & (df_all_score.score < score_bin[i + 1])
        df_temp = df_all_score.loc[index_1, ['target', 'score']]
        ##计算该分数区间的指标
        good_num.append(sum(df_temp.target == 0))
        bad_num.append(sum(df_temp.target == 1))
        ##区间样本率
        bin_rate.append(df_temp.shape[0] / df_all_score.shape[0] * 100)
        ##坏样本率
        bad_rate.append(df_temp.target.sum() / df_temp.shape[0] * 100)
        ##以该分数为注入分数的ks值
        ks.append(sum(bad_num[0:i + 1]) / bad_total - sum(good_num[0:i + 1]) / good_total)

    df_result = pd.DataFrame({'score_bin': score_index , 'good_num': good_num, 'bad_num': bad_num, 'bin_rate': bin_rate,
                              'bad_rate': bad_rate, 'ks': ks})
    print('######计算结果如下###########')
    print(df_result)
    return df_result


def build_FM(fm, x_train, y_train):
    from pyfm import pylibfm
    fm = pylibfm.FM(num_factors=5, num_iter=500, verbose=True, task="classification",
                    initial_learning_rate=0.0001, learning_rate_schedule="optimal")
    return fm
def  build_train_FM(x_train,y_train):
    fm=build_FM()
    fm.fit(x_train, y_train)
    return fm

def fm_predict(fm,x_test):
    y_score_test = fm.predict(x_test)
    return y_score_test


import pickle
def pkl_save(filename,file):
    output = open(filename, 'wb')
    pickle.dump(file, output)
    output.close()

def pkl_load(filename):
    pkl_file = open(filename, 'rb')
    file = pickle.load(pkl_file)
    pkl_file.close()
    return file


def save_lightGBM(model ,model_file="dota_model.txt"):
    #clf.booster_.savemodel(file)
    joblib.dump( model, model_file)
    return

def load_lightGBM(model_file='dota_model.txt'):
    # clf = lgb.Booster(model_file)
    model = joblib.load(model_file)
    return model