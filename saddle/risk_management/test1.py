# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
import warnings
warnings.filterwarnings("ignore")  ##忽略警告


def func_s(x):
    return str(x[0]) + '_Cross_' + str(x[1])

import data_preprocess as dp
import variable_bin_methods  as varbin_meth
import variable_encode as var_encode
import model_train_predict as mtp
import model_eval          as meval




def credit_score_card_test():
    ##评分卡模型测试
    df_train, df_test = dp.tmpdata_reads('./data/', 'german.csv')
    feature_names = list(df_train.columns)
    feature_names.remove('target')
    categorical_var, numerical_var = dp.category_continue_separation(df_train, feature_names)
    df_train, df_test, numerical_var, categorical_var = dp.contiues_tranfer_to_category(df_train, df_test, numerical_var, categorical_var)

    params = dp.get_disc_bin(df_train, numerical_var, categorical_var)
    dict_cont_bin, dict_disc_bin, del_key, gain_value_save_train1, gain_rate_save1, gain_value_save2, gain_rate_save2 = params

    ###删除分箱数只有1个的变量
    if len(del_key) > 0:
        for j in del_key:
            del dict_disc_bin[j]

    # 训练集分箱，测试集分箱
    df_cont_bin_train, df_disc_bin_train, df_train_bin = dp.cont_disc_bin_merge(df_train, dict_cont_bin, dict_disc_bin)
    df_cont_bin_test, df_disc_bin_test, df_test_bin    = dp.cont_disc_bin_merge(df_test, dict_cont_bin, dict_disc_bin)

    df_train_bin.reset_index(inplace=True, drop=True)
    df_test_bin.reset_index(inplace=True, drop=True)

    ###WOE编码
    var_all_bin = list(df_train_bin.columns)
    var_all_bin.remove('target')
    ##训练集WOE编码
    df_train_woe, dict_woe_map, dict_iv_values, var_woe_name = var_encode.woe_encode(df_train_bin, './data', var_all_bin, df_train_bin.target, 'dict_woe_map', flag='train')
    ##测试集WOE编码
    df_test_woe, var_woe_name = var_encode.woe_encode(df_test_bin, './data', var_all_bin, df_test_bin.target, 'dict_woe_map', flag='test')

    ####取出训练数据与测试数据
    x_train = df_train_woe[var_woe_name]
    x_train = np.array(x_train)
    y_train = np.array(df_train_bin.target)

    x_test = df_test_woe[var_woe_name]
    x_test = np.array(x_test)
    y_test = np.array(df_test_bin.target)

    lr_gsearch, LR_model = mtp.grib_lr_search(x_train, y_train)
    lr_model_fit = LR_model.fit(x_train, y_train)
    dict_params_weights = mtp.get_lr_model_weight(var_woe_name, lr_model_fit)

    # 查看训练集、验证集与测试集   predict_proba 得出0的概率，1的概率
    y_score_train = lr_model_fit.predict_proba(x_train)[:, 1]
    y_score_test  = lr_model_fit.predict_proba(x_test)[:, 1]
    y_pred = lr_model_fit.predict(x_test)

    report = meval.model_evl(type='classifier', y_true=y_test, y_pred=y_pred)
    print(report)

    ####生成评分卡
    df_score, dict_bin_score, params_A, params_B, score_base = mtp.create_score(dict_woe_map, dict_params_weights, dict_cont_bin, dict_disc_bin)
    print('df_score\n', df_score.head(8))
    df_score.to_csv('./data/german_create_score.csv')
    ##计算样本评分
    df_all = pd.concat([df_train, df_test], axis=0)
    df_all_score = mtp.cal_score(df_all, dict_bin_score, dict_cont_bin, dict_disc_bin, score_base)
    score_statas_df = mtp.score_statas(df_all_score)
    # print(score_statas_df.head(20))
    return


from feature_selector import FeatureSelector,process_multicollinearity
def multiModels_Test():
    ##多模型测试
    df_train, df_test = dp.tmpdata_reads('./data/', 'german.csv')
    feature_names = list(df_train.columns)
    feature_names.remove('target')
    categorical_var, numerical_var = dp.category_continue_separation(df_train, feature_names)
    df_train, df_test, numerical_var, categorical_var = dp.contiues_tranfer_to_category(df_train, df_test, numerical_var, categorical_var)

    params = dp.get_disc_bin(df_train, numerical_var, categorical_var)
    dict_cont_bin, dict_disc_bin, del_key, gain_value_save_train1, gain_rate_save1, gain_value_save2, gain_rate_save2 = params

    ###删除分箱数只有1个的变量
    if len(del_key) > 0:
        for j in del_key:
            del dict_disc_bin[j]

    # 训练集分箱，测试集分箱
    df_cont_bin_train, df_disc_bin_train, df_train_bin = dp.cont_disc_bin_merge(df_train, dict_cont_bin, dict_disc_bin)
    df_cont_bin_test, df_disc_bin_test, df_test_bin = dp.cont_disc_bin_merge(df_test, dict_cont_bin, dict_disc_bin)

    df_train_bin.reset_index(inplace=True, drop=True)
    df_test_bin.reset_index(inplace=True, drop=True)

    ###WOE编码
    var_all_bin = list(df_train_bin.columns)
    var_all_bin.remove('target')
    ##训练集WOE编码
    df_train_woe, dict_woe_map, dict_iv_values, var_woe_name = var_encode.woe_encode(df_train_bin, './data', var_all_bin, df_train_bin.target, 'dict_woe_map', flag='train')
    ##测试集WOE编码
    df_test_woe, var_woe_name = var_encode.woe_encode(df_test_bin, './data', var_all_bin, df_test_bin.target, 'dict_woe_map', flag='test')

    sel_var = process_multicollinearity(df_train_woe, dict_iv_values)
    ##随机森林排序
    ##特征选择
    fs = FeatureSelector(data=df_train_woe[sel_var], labels=df_train_bin.target)
    ##一次性去除所有的不满足特征
    fs.identify_all(selection_params={'missing_threshold': 0.9,
                                      'correlation_threshold': 0.8,
                                      'task': 'classification',
                                      'eval_metric': 'binary_error',
                                      'max_depth': 2,
                                      'cumulative_importance': 0.90})
    df_train_woe = fs.remove(methods='all')
    df_train_woe['target'] = df_train_bin.target

    ####取出训练数据与测试数据
    x_train = df_train_woe[var_woe_name]
    x_train = np.array(x_train)
    y_train = np.array(df_train_bin.target)

    x_test = df_test_woe[var_woe_name]
    x_test = np.array(x_test)
    y_test = np.array(df_test_bin.target)

    # lr_gsearch, LR_model = mtp.grib_lr_search(  x_train,y_train   )
    # lr_model_fit         = LR_model.fit(        x_train,y_train   )
    # dict_params_weights  = mtp.get_lr_model_weight( var_woe_name,  lr_model_fit  )

    # 创建多个分类模型进行，查看效果
    mtp.multi_model_test(x_train, y_train, n_splits=7)
    return
if __name__ == '__main__':
    credit_score_card_test(  )  #信用评分卡模型
    multiModels_Test(        )  #多个分类模型测试举例
    exit()



