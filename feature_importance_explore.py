#!/usr/local/bin/python
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost.sklearn import XGBClassifier,XGBRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression

####################### 特征重要性筛选
def randomforest_importance(train_x, train_y,feats, classifier_regressor=1):
    forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1) if classifier_regressor==1 else  RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    forest.fit(train_x, train_y)
    importances = forest.feature_importances_     #rfr.score(test_x, test_y)
    tmp_df = pd.DataFrame(data={'feature':feats,  'importances':importances    }).sort_values(by=['importances'],ascending=False)
    return tmp_df

def linergressor_importance(train_x, train_y,feats, classifier_regressor=1):
    model = LinearRegression() if classifier_regressor==1 else LogisticRegression()  
    model.fit(train_x, train_y)  
    importance = model.coef_  
    for i,v in enumerate(importance):  
        print('Feature: %0d, Score: %.5f ,%s  ' % (feats[i],v,feats[i]))
    return tmp_df

def XGB_importance(train_x, train_y,feats, classifier_regressor=1):
    model = XGBRegressor() if classifier_regressor==1 else XGBClassifier()  
    model.fit(train_x, train_y)  
    importance =  model.feature_importances_
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f ,%s ' % (i,v,feats[i]))
    return

import statsmodels.api as sm
def smLogit_importance(train_x, train_y,feats,label='y', classifier_regressor=1,save_file="./smlogit_importance.csv"):

    # print('x:',train_x[:5])
    # print('y:',train_y[:5],np.unique(train_y) )
    # train_x = sm.add_constant(train_x)
    # train_y = sm.add_constant(train_y)
    print('x:',train_x[:5])
    print('y:',train_y[:5],np.unique(train_y) )

    clf = sm.Logit(train_y,train_x) if classifier_regressor==0 else sm.OLS(train_y , train_x)
    clf = clf.fit( method='bfgs' if classifier_regressor==0 else 'pinv'  )
    importance_ = clf.summary(yname=label,xname=feats)
    importance_ = importance_.as_csv()
    import sys
    fw = open(save_file,"w",encoding ='utf-8')
    fw.write( importance_ ,)
    fw.close()
    return importance_

def get_correlation( src_df ,drop_keys=[],label=''):
    corrwith_df =  src_df.drop( drop_keys,axis=1).corrwith(  src_df[ label ],method='pearson' ).to_frame().rename(columns={0:'correlation'}).sort_values(by='correlation',ascending=False)
    return corrwith_df

from sklearn.feature_selection import RFE
def wraper_rfe(train_x, train_y,feats, classifier_regressor=1):
    RFC_ = RandomForestClassifier(n_estimators=101, random_state=0, n_jobs=-1) if classifier_regressor==1 else  RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
    selector = RFE(RFC_, n_features_to_select=2, step=50).fit(train_x,train_y)
    print( selector.support_.sum() )
    print( 'rank',selector.ranking_ )
    print( 'support',selector.support_ ) 
    X_wrapper = selector.transform(train_x)
    #print( cross_val_score(RFC_,X_wrapper,y,cv=5).mean()  )
    return 

# feature process
from sklearn.preprocessing import StandardScaler

def get_signal(feature_list):
    mu = np.mean(feature_list)
    sigma = np.std(feature_list)
    ret_list = [  (i-mu)/sigma    for i in  feature_list ]
    return ret_list

def data_range(df,cols): #*cols
    krange = []
    for col in cols:
        crange = df[col].max() - df[col].min()
        krange.append(crange)
    return(krange)

#src_iris_df['sepal width (cm)'].hist( bins=10 )
import os 
def gen_data_his_picture(  data, key, bins,picture_path='./feature_select/'):
    ax  = data[ key ].hist( bins=bins )
    fig = ax.get_figure()
    if not os.path.exists(picture_path): os.mkdir(picture_path)
    fig.savefig( picture_path + '%s.png'%key)    
    return 


#grr = pd.plotting.scatter_matrix(iris_dataframe,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8)
#plt.show() 
import pandas as pd
def gen_data_scatter_integration_picture(df,key,bins,picture_path='./feature_select/'):
    ax  =   pd.plotting.scatter_matrix(df,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=.8)
    fig = ax.get_figure()
    if not os.path.exists(picture_path): os.mkdir(picture_path)
    fig.savefig( picture_path + 'scatter_integration_'+'%s.png'%key)    
    return 

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
def gen_data_scatter_single_picture(x,y,key, picture_path='./feature_select/'):
    plt.scatter(x,y, alpha = 0.4, cmap = 'Reds')
    if not os.path.exists(picture_path): os.mkdir(picture_path)
    plt.savefig( picture_path + 'scatter_single_'+'%s.png'%key)
    return

def gen_box_plot( df,numerical_var,  picture_path='./feature_select/' ):
    ##对于连续数据绘制箱线图，观察是否有异常值
    ###连续变量不同类别下的分布
    for x  in numerical_var[::8]:
        plt.figure(figsize=(10, 6))  # 设置图形尺寸大小
        for j in range(1, len(numerical_var) + 1):
            plt.subplot(2, 4, j)
            df_temp = df[ numerical_var[j - 1]][~df[numerical_var[j - 1]].isnull()]
            plt.boxplot(df_temp,
                        notch=False,  # 中位线处不设置凹陷
                        widths=0.2,  # 设置箱体宽度
                        medianprops={'color': 'red'},  # 中位线设置为红色
                        boxprops=dict(color="blue"),  # 箱体边框设置为蓝色
                        labels=[numerical_var[j - 1]],  # 设置标签
                        whiskerprops={'color': "black"},  # 设置须的颜色，黑色
                        capprops={'color': "green"},  # 设置箱线图顶端和末端横线的属性，颜色为绿色
                        flierprops={'color': 'purple', 'markeredgecolor': "purple"}  # 异常值属性，这里没有异常值，所以没表现出来
                        )
            plt.savefig(picture_path + 'bar_%s.png'%(x)  )
    plt.show()

    return

import missingno as msno
def data_na_bar( df,  pic_path=''  ):
    if not os.path.exists(pic_path): os.mkdir(pic_path)
    msno.matrix( df, labels=True)
    plt.savefig(  os.path.join( pic_path, 'fillna_matrix.jpeg'  )     )
    msno.heatmap( df )
    plt.savefig(  os.path.join( pic_path, 'fillna_heatmap.jpeg'  )     )
    msno.bar(df, labels=True, figsize=(10, 6), fontsize=10)
    plt.savefig( os.path.join( pic_path, 'fillna_bar.jpeg'  )    )
    return


#######data 数据分析
from .data_preprocess import category_continue_separation
def data_anlalyse(df,label ,pic_path='./feature_select/'):
    feature_names = df.columns.tolist()
    categorical_var, numerical_var = category_continue_separation( df, feature_names )
    for s in set(numerical_var):
        print('数值变量' + s + '可能取值' + str(len(df[s].unique())))
    for s in set(categorical_var):
        print('离散变量' + s + '可能取值' + str(len(df[s].unique())))
    data_na_bar(df, pic_path )  # 绘制每个变量的连续、缺失值情况
    for k in df.columns:  #分箱干什么
        print(k)
        gen_data_his_picture(data=df, key=k, bins=10, picture_path=pic_path)

    for k in df.columns :
        print(k)
        if k=='label':continue
        gen_data_scatter_single_picture( df[k], df[label], key=k, picture_path=pic_path)

    # 分箱看异常值情况
    categorical_var, numerical_var = category_continue_separation( df,feature_names=df.columns.tolist() )
    gen_box_plot( df,numerical_var ,  picture_path='./feature_select/' )

    corr_df     = get_correlation(src_iris_df, drop_keys=[label], label=label);  print(corr_df.head())
    df_des1     = src_iris_df.describe().T
    print(df_des1.index) ;  print(df_des1.columns)

    skew_df     = df.skew().to_frame().rename(columns={0: 'skew'})
    kurtosis_df = df.kurtosis().to_frame().rename(columns={0: 'kurtosis'})
    des_df      = pd.concat([corr_df, df_des1, skew_df, kurtosis_df], axis=1)
    des_df.to_csv( pic_path + 'description.csv' )
    return des_df

if  __name__ == '__main__':
    from sklearn import datasets
    iris                    = datasets.load_iris()
    src_iris_df             = pd.DataFrame( data=iris.data, columns = iris.feature_names  )
    src_iris_df['label']    = iris.target
    print( src_iris_df.head() )
    importance_desc = smLogit_importance( iris.data, iris.target, feats=iris.feature_names, classifier_regressor=0,
                                        save_file="./smlogit_importance.csv" )
    print(  importance_desc )

    importance_df = randomforest_importance(iris.data, iris.target, iris.feature_names, classifier_regressor=0)
    print(importance_df)

    wraper_rfe(iris.data, iris.target, feats=iris.feature_names, classifier_regressor=1)
    desc_df = data_anlalyse( src_iris_df,label='label' )
    print(  desc_df  )

    exit()

##from sklearn import datasets
#def load_data():
    #iris = datasets.load_iris()
    #print(iris.keys())
    #n_samples, n_features = iris.data.shape
    #print((n_samples, n_features))
    #print(iris.data[0])
    #print(iris.target.shape)
    #print(iris.target)
    #print(iris.target_names)
    #print("feature_names:", iris.feature_names)