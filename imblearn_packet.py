import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours

def underSample(x_data, y_data, replacement=True):
    #随机下采样，并不考虑多数样本的数据分布，等概率采样使正负样本一致
    #input:x_data可以是dataframe ,y_data series  return: dataframe结构
    randUnderSample_model = RandomUnderSampler(random_state=10, replacement=replacement) #replacement True有放回,False无放回
    X_resample, y_resample = randUnderSample_model.fit_resample(x_data, y_data)
    ret_df           = pd.DataFrame(X_resample, columns=x_data.columns)
    ret_df['target'] = y_resample
    return ret_df, randUnderSample_model

def near_miss( x_data, y_data, version,n_neighbors=3,replacement=True):
    ##nearmiss方法
    near_miss_1 = NearMiss( version=version, n_neighbors=n_neighbors) #3
    X_resample, y_resample = near_miss_1.fit_resample(x_data, y_data)
    ret_df = pd.DataFrame(X_resample, columns=x_data.columns)
    ret_df['target'] = y_resample
    return ret_df

def tomek_links(x_data, y_data,sampling_strategy='auto'):
    ##TomekLinks方法
    tom_link = TomekLinks(sampling_strategy= sampling_strategy )
    X_resample, y_resample = tom_link.fit_resample( x_data, y_data )
    #tom_link.sample_indices_
    ret_df              = pd.DataFrame(X_resample, columns=x_data.columns)
    ret_df['target']    = y_resample
    return  ret_df,tom_link

def edit_nearst_neighbours(x_data, y_data):
    ##EditedNearestNeighbours方法
    Enn = EditedNearestNeighbours(n_neighbors=3, kind_sel='mode')
    X_resample, y_resample = Enn.fit_resample(x_data, y_data)
    ret_df              = pd.DataFrame(X_resample, columns=x_data.columns)
    ret_df['target']    = y_resample
    return ret_df,Enn



import math
from imblearn.ensemble import EasyEnsembleClassifier,BalancedBaggingClassifier #EasyEnsemble, BalanceCascade
def easy_ensemble(x_data, y_data):
    ##bagging方法
    sub_num = math.ceil(sum(y_data == 0) / sum(y_data == 1))
    easy_en = EasyEnsembleClassifier( n_subsets=sub_num, replacement=True)
    # X_resample, y_resample = easy_en.fit_resample(x_data, y_data)
    # ret_df = pd.DataFrame(X_resample, columns=x_data.columns)
    # ret_df['target'] = y_resample
    return easy_en

def boost(x_cont_data, y_data):
    #boosting方法
    boost_balance = BalancedBaggingClassifier(random_state=42)
    X_resample, y_resample = boost_balance.fit_resample(x_cont_data, y_data)
    return boost_balance

def random_over_sampler(x_data, y_data,sampling_strategy):
    #############数据层上采样方法###################################################
    ##随机上采样
    rand_over_sample = RandomOverSampler(sampling_strategy=sampling_strategy)
    X_resample, y_resample = rand_over_sample.fit_resample(x_data, y_data)
    ret_df = pd.DataFrame(X_resample, columns=x_data.columns)
    ret_df['target'] = y_resample
    return ret_df

from imblearn.over_sampling import RandomOverSampler, SMOTE
def smote(x_data, y_data,sampling_strategy=1, k_neighbors=5, kind='regular'):
    ##Smote样本生成方法,插值法
    sm_sample = SMOTE(random_state=10, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)
    X_resample, y_resample = sm_sample.fit_resample(x_data, y_data)
    ret_df = pd.DataFrame(X_resample, columns=x_data.columns)
    ret_df['target'] = y_resample
    return ret_df

from sklearn.decomposition import PCA
def build_pca_model(n_components,x_cont_data):
    ##PCA降维，到2维
    model_pca = PCA(n_components=n_components).fit(x_cont_data)
    # X_pca_raw = model_pca.transform(x_cont_data)
    # X_pca_1 = model_pca.transform(X_resample_1)
    return model_pca

def process_unbalance(df,target,mutiply=3):
    target_set = set(df[target])
    for k in target_set:
        print('%s-%s len:' % (k, target), len(df[df[target] == k]))
    sample_df = df[df[target] == 1];
    postive_len = len(sample_df);
    sample_df = sample_df.append(  df[df[target] != 1].sample( int(mutiply * postive_len) )  )
    return sample_df

def plot_undersample(x_cont_data,y_data, X_resample,y_resample):
    ##看underSample
    ##PCA降维，到2维
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    model_pca = PCA(n_components=2).fit(x_cont_data)
    X_pca_raw = model_pca.transform(x_cont_data)
    X_pca_1   = model_pca.transform(X_resample )

    ##降维后,用前两维的
    plt.figure(figsize=(15, 8))
    fontsize_1 = 15
    ##原始数据降维结果
    plt.subplot(211)
    index_1 = y_data == 0
    plt.scatter(X_pca_raw[index_1, 0], X_pca_raw[index_1, 1], c='grey', marker='o', label='负样本')
    plt.scatter(X_pca_raw[~index_1, 0], X_pca_raw[~index_1, 1], c='black', marker='+', alpha=0.5, label='正样本')
    plt.title('raw_data', fontsize=fontsize_1)
    plt.xticks(fontsize=fontsize_1)
    plt.yticks(fontsize=fontsize_1)
    plt.legend()
    ##采样数据降维结果
    plt.subplot(212)
    index_2 =y_resample == 0
    plt.scatter(X_pca_1[index_2, 0], X_pca_1[index_2, 1], c='grey', marker='o', label='负样本')
    plt.scatter(X_pca_1[~index_2, 0], X_pca_1[~index_2, 1], c='black', marker='+', alpha=0.5, label='正样本')
    plt.title('RandomUnderSampler', fontsize=fontsize_1)
    plt.xticks(fontsize=fontsize_1)
    plt.yticks(fontsize=fontsize_1)
    plt.legend()
    plt.show()
    return

def plot_enn(x_cont_data,y_data,X_resample,y_resample):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    ##PCA降维，到2维
    model_pca = PCA(n_components=2).fit(x_cont_data)
    X_pca_raw = model_pca.transform(x_cont_data)
    X_pca_1 = model_pca.transform(X_resample)
    ##降维后,用前两维的
    plt.figure(figsize=(15, 8))
    fontsize_1 = 15
    ##原始数据降维结果
    plt.subplot(211)
    index_1 = y_data == 0
    plt.scatter(X_pca_raw[index_1, 0], X_pca_raw[index_1, 1], c='grey', marker='o', label='负样本')
    plt.scatter(X_pca_raw[~index_1, 0], X_pca_raw[~index_1, 1], c='black', marker='+', alpha=0.5, label='正样本')
    plt.title('raw_data', fontsize=fontsize_1)
    plt.xticks(fontsize=fontsize_1)
    plt.yticks(fontsize=fontsize_1)
    plt.legend()
    ##采样数据降维结果
    plt.subplot(212)
    index_2 = y_resample == 0
    plt.scatter(X_pca_1[index_2, 0], X_pca_1[index_2, 1], c='grey', marker='o', label='负样本')
    plt.scatter(X_pca_1[~index_2, 0], X_pca_1[~index_2, 1], c='black', marker='+', alpha=0.5, label='正样本')
    plt.title('EditedNearestNeighbours', fontsize=fontsize_1)
    plt.xticks(fontsize=fontsize_1)
    plt.yticks(fontsize=fontsize_1)
    plt.legend()
    plt.show()
    return

def plot_near_miss(x_cont_data,y_data,Xresample1,Xresample2,Xresample3,Yresample1,Yresample2,Yresample3):
    #NearMiss 效果
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

    model_pca = PCA(n_components=2).fit(x_cont_data)
    X_pca_raw = model_pca.transform(x_cont_data)
    X_pca_1 = model_pca.transform(Xresample1)
    X_pca_2 = model_pca.transform(Xresample2)
    X_pca_3 = model_pca.transform(Xresample3)

    ##降维后,用前两维的
    plt.figure(figsize=(15, 8))
    fontsize_1 = 15
    ##原始数据降维结果
    plt.subplot(221)
    index_1 = y_data == 0
    plt.scatter(X_pca_raw[index_1, 0], X_pca_raw[index_1, 1], c='grey', marker='o', label='负样本')
    plt.scatter(X_pca_raw[~index_1, 0], X_pca_raw[~index_1, 1], c='black', marker='+', alpha=0.5, label='正样本')
    plt.title('raw_data', fontsize=fontsize_1)
    plt.xticks(fontsize=fontsize_1)
    plt.yticks(fontsize=fontsize_1)
    plt.legend()
    ##NearMiss_1降维结果
    plt.subplot(222)
    index_2 = Yresample1 == 0
    plt.scatter(X_pca_1[index_2, 0], X_pca_1[index_2, 1], c='grey', marker='o', label='负样本')
    plt.scatter(X_pca_1[~index_2, 0], X_pca_1[~index_2, 1], c='black', marker='+', alpha=0.5, label='正样本')
    plt.title('NearMiss_1', fontsize=fontsize_1)
    plt.xticks(fontsize=fontsize_1)
    plt.yticks(fontsize=fontsize_1)
    plt.legend()
    ##NearMiss_2降维结果
    plt.subplot(223)
    index_3 = Yresample2 == 0
    plt.scatter(X_pca_2[index_3, 0], X_pca_2[index_3, 1], c='grey', marker='o', label='负样本')
    plt.scatter(X_pca_2[~index_3, 0], X_pca_2[~index_3, 1], c='black', marker='+', alpha=0.5, label='正样本')
    plt.title('NearMiss_2', fontsize=fontsize_1)
    plt.xticks(fontsize=fontsize_1)
    plt.yticks(fontsize=fontsize_1)
    plt.legend()
    ##NearMiss_3降维结果
    plt.subplot(224)
    index_4 = Yresample3 == 0
    plt.scatter(X_pca_3[index_4, 0], X_pca_3[index_4, 1], c='grey', marker='o', label='负样本')
    plt.scatter(X_pca_3[~index_4, 0], X_pca_3[~index_4, 1], c='black', marker='+', alpha=0.5, label='正样本')
    plt.title('NearMiss_3', fontsize=fontsize_1)
    plt.xticks(fontsize=fontsize_1)
    plt.yticks(fontsize=fontsize_1)
    plt.legend()
    plt.show()
    return

def plot_tok(x_cont_data,y_data,Xresample,Yresample):
    #tokek_link  PCA降维，到2维
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    model_pca = PCA(n_components=2).fit(x_cont_data)
    X_pca_raw   = model_pca.transform(x_cont_data)
    X_pca_1     = model_pca.transform(Xresample )
    ##降维后,用前两维的
    plt.figure(figsize=(15, 8))
    fontsize_1 = 15
    ##原始数据降维结果
    plt.subplot(211)
    index_1 = y_data == 0
    plt.scatter(X_pca_raw[index_1, 0], X_pca_raw[index_1, 1], c='grey', marker='o', label='负样本')
    plt.scatter(X_pca_raw[~index_1, 0], X_pca_raw[~index_1, 1], c='black', marker='+', alpha=0.5, label='正样本')
    plt.title('raw_data', fontsize=fontsize_1)
    plt.xticks(fontsize=fontsize_1)
    plt.yticks(fontsize=fontsize_1)
    plt.legend()
    ##采样数据降维结果
    plt.subplot(212)
    index_2 = Yresample == 0
    plt.scatter(X_pca_1[index_2, 0], X_pca_1[index_2, 1], c='grey', marker='o', label='负样本')
    plt.scatter(X_pca_1[~index_2, 0], X_pca_1[~index_2, 1], c='black', marker='+', alpha=0.5, label='正样本')
    plt.title('TomekLinks Sample', fontsize=fontsize_1)
    plt.xticks(fontsize=fontsize_1)
    plt.yticks(fontsize=fontsize_1)
    plt.legend()
    plt.show()
    return


if __name__ == '__main__':
    import data_preprocess as dp
    import variable_bin_methods  as varbin_meth
    import variable_encode as var_encode
    import model_train_predict as mtp
    import model_eval          as meval

    df_train, df_test = dp.tmpdata_reads('./code/chapter10/data/', 'german.csv')
    feature_names = list(df_train.columns)

    data_df = pd.concat([df_train, df_test], axis=0)
    data_df = data_df.reset_index(drop=True)
    sum(data_df.target == 1)
    sum(data_df.target == 0)
    x_data = data_df.loc[:, data_df.columns != 'target']
    y_data = data_df.target

    cont_name = ['duration', 'amount', 'income_rate', 'residence_info',
                 'age', 'num_credits', 'dependents']
    x_cont_data = data_df[cont_name]

    underSample_df,randUnderSample_model = underSample(  x_data, y_data )
    print(underSample_df.head(10))
    plot_undersample(x_cont_data, y_data,underSample_df[cont_name], underSample_df.target )

    resample_1 =  near_miss( x_cont_data, y_data, 1,n_neighbors=3,replacement=True)  # NearMiss(random_state=10, version=1, n_neighbors=3, ratio=1)
    resample_2 =  near_miss( x_cont_data, y_data, 2,n_neighbors=3,replacement=True)    #NearMiss(random_state=10, version=2, n_neighbors=3, ratio=1)
    resample_3 =  near_miss( x_cont_data, y_data, 3,n_neighbors=3,replacement=True)    #NearMiss(random_state=10, version=3, n_neighbors=3, ratio=1)
    plot_near_miss(x_cont_data, y_data,resample_1[cont_name],resample_2[cont_name],resample_3[cont_name],
                    resample_1.target,resample_2.target,resample_3.target)

    ##TomekLinks方法
    tom_df,tom_link = tomek_links(x_cont_data,y_data, sampling_strategy='auto' )
    plot_tok(x_cont_data, y_data, tom_df[cont_name], tom_df.target )

    ##EditedNearestNeighbours方法
    Enn_df, Enn = edit_nearst_neighbours( x_cont_data,y_data         )
    plot_enn( x_cont_data,y_data, Enn_df[cont_name] , Enn_df.target  )














