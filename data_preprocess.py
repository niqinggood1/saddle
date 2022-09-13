# descrirption :
# category_continue_separation
#
def category_continue_separation(data_df, feature_names):
    ##区分出离散变量与连续变量, 首先要剔除掉target\ label特征;
    #输入是dataframe和 feature_lists
    categorical_var = []
    numerical_var = []
    if 'target' in feature_names :
        feature_names.remove('target')
    if 'label' in feature_names :
        feature_names.remove('label')
    ##先判断类型，如果是int或float就直接作为连续变量 select_dtypes
    numerical_var   = list( data_df[feature_names].select_dtypes(include=['int', 'float', 'int32', 'float32', 'int64', 'float64']).columns.values)
    categorical_var = [x for x in feature_names if x not in numerical_var]
    return categorical_var, numerical_var

def contiues_tranfer_to_category(data_train,data_test,numerical_var,categorical_var,limit_num=10):
    for s in set(numerical_var):
        print('变量' + s + '可能取值' + str(len(data_train[s].unique())))
        if len(data_train[s].unique()) <= limit_num:
            categorical_var.append(s)
            numerical_var.remove(s)
            ##对数值少于一定值的数值变量转为离散特征变量，同时将后加的数值变量转为字符串
            #handle both train and test
            index_1 = data_train[s].isnull()
            if sum(index_1) > 0:
                data_train.loc[~index_1, s] = data_train.loc[~index_1, s].astype('str')
            else:
                data_train[s] = data_train[s].astype('str')
            index_2 = data_test[s].isnull()
            if sum(index_2) > 0:
                data_test.loc[~index_2, s] = data_test.loc[~index_2, s].astype('str')
            else:
                data_test[s] = data_test[s].astype('str')
    return data_train,data_test,numerical_var,categorical_var

#
def get_disc_bin(df, numerical_var,categorical_var ):
    dict_cont_bin = {}  #dict_cont_bin 是每个变量i,后面对应的分箱
    for i in numerical_var:
        print(i)
        dict_cont_bin[i], gain_value_save_train1, gain_rate_save1 = varbin_meth.cont_var_bin(df[i], df.target, method=2, mmin=3, mmax=12,
                                                                                     bin_rate=0.01, stop_limit=0.05, bin_min_num=20)
    ###离散变量分箱
    dict_disc_bin = {} ; del_key = []
    for i in categorical_var:
        dict_disc_bin[i], gain_value_save2, gain_rate_save2, del_key_1 = varbin_meth.disc_var_bin(df[i], df.target, method=2, mmin=3,
                                                                                                mmax=8, stop_limit=0.05, bin_min_num=20)
        if len( del_key_1 ) > 0:
            del_key.extend(del_key_1)
            ###删除分箱数只有1个的变量
    return  dict_cont_bin, dict_disc_bin,del_key,gain_value_save_train1, gain_rate_save1,gain_value_save2, gain_rate_save2


from . import variable_bin_methods  as varbin_meth
def cont_disc_bin_merge( df, dict_cont_bin,dict_disc_bin ):
    ##训练数据分箱，连续变量分箱映射
    df_cont_bin     = pd.DataFrame()
    for i in dict_cont_bin.keys():
        print(i)
        df_cont_bin = pd.concat([df_cont_bin, varbin_meth.cont_var_bin_map(df[i], dict_cont_bin[i])], axis=1)
    ##离散变量分箱映射
    df_disc_bin     = pd.DataFrame()
    for i in dict_disc_bin.keys():
        print(i)
        df_disc_bin = pd.concat([df_disc_bin, varbin_meth.disc_var_bin_map(df[i], dict_disc_bin[i])], axis=1)

    if 'target' in df.columns:
        df_disc_bin['target'] = df.target
    data_bin = pd.concat([df_cont_bin, df_disc_bin], axis=1)

    return  df_cont_bin,df_disc_bin,data_bin

import datetime
def date_str(delat, fmt="%Y-%m-%d", origin_date=datetime.date.today()):
    oneday = datetime.datetime.now().strftime('%Y-%m-%d')
    date = origin_date + datetime.timedelta(delat)
    return date.strftime(fmt)

def sigmod(X):
    return 1.0 / (1 + np.exp(-X))
def Z_ScoreNormalization(x, mu, sigma):
    x = (x - mu) / sigma;
    return x
def als_recommend():
    return
def describe(df):

    df.status_account.unique()
    df.describe()
    #df.drop_duplicates(subset=['order_id'], keep='first', inplace=True)
    #数据清理前期好像用ipynb

    return

def data_clear(df):
    df.status_account.unique()
    return

import pandas as pd
def get_dumies(df,one_hot_feature):
    dummies_df = pd.get_dummies(df, columns=one_hot_feature).fillna(0)
    return dummies_df


from sklearn.preprocessing import StandardScaler
def scaler(data_train,data_test,var_all):
    ####变量归一化,进一步完善
    scaler              = StandardScaler().fit( data_train[var_all])
    data_train[var_all] = scaler.transform( data_train[var_all])
    data_test[var_all]  = scaler.transform( data_test[var_all] )
    return  data_train,data_test



def num_to_ca( data_train,s ):
    index_1 = data_train[s].isnull()
    if sum(index_1) != 0:
        data_train.loc[~index_1, s] = data_train.loc[~index_1, s].astype('str')
    else:
        data_train[s]               = data_train[s].astype('str')
    return data_train

import os
from sklearn.model_selection import train_test_split
def tmpdata_reads(data_path, file_name):
    df = pd.read_csv(os.path.join(data_path, file_name), delim_whitespace=True, header=None)
    ##变量重命名
    columns = ['status_account', 'duration', 'credit_history', 'purpose', 'amount',
               'svaing_account', 'present_emp', 'income_rate', 'personal_status',
               'other_debtors', 'residence_info', 'property', 'age',
               'inst_plans', 'housing', 'num_credits',
               'job', 'dependents', 'telephone', 'foreign_worker', 'target']
    df.columns = columns
    ##将标签变量由状态1,2转为0,1;0表示好用户，1表示坏用户
    df.target = df.target - 1
    ##数据分为data_train和 data_test两部分，训练集用于得到编码函数，验证集用已知的编码规则对验证集编码
    data_train, data_test = train_test_split(df, test_size=0.2, random_state=0, stratify=df.target)
    return data_train, data_test


if __name__ == '__main__':

    print('data_preprocess')