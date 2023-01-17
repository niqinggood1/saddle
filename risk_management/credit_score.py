##生成评分卡
import pandas as pd
import numpy as np
def score_params_cal(base_point, odds, PDO):
    ##给定预期分数，与翻倍分数，确定参数A,B
    B = PDO / np.log(2)
    A = base_point + B * np.log(odds)
    return A, B
def myfunc(x):
    return str(x[0]) + '_' + str(x[1])
def create_score(dict_woe_map, dict_params, dict_cont_bin, dict_disc_bin):
    ##假设Odds在1:60时对应的参考分值为600分，分值调整刻度PDO为20，则计算得到分值转化的参数B = 28.85，A= 481.86。
    #dict_woe_map    woe字典，详细
    # dict_params
    # dict_cont_bin
    # dict_disc_bin
    params_A, params_B = score_params_cal(base_point=600, odds=1 / 60, PDO=20)
    # 计算基础分
    base_points = round(params_A - params_B * dict_params['intercept'])
    df_score = pd.DataFrame()
    dict_bin_score = {}
    for k in dict_params.keys():
        #        k='duration_BIN'
        #        k = 'foreign_worker_BIN'
        if k != 'intercept':
            df_temp = pd.DataFrame([dict_woe_map[k.split(sep='_woe')[0]]]).T
            df_temp.reset_index(inplace=True)
            df_temp.columns = ['bin', 'woe_val']
            ##计算分值
            df_temp['score'] = round(-params_B * df_temp.woe_val * dict_params[k])
            dict_bin_score[k.split(sep='_BIN')[0]] = dict(zip(df_temp['bin'], df_temp['score']))
            ##连续变量的计算
            if k.split(sep='_BIN')[0] in dict_cont_bin.keys():
                df_1 = dict_cont_bin[k.split(sep='_BIN')[0]]
                df_1['var_name'] = df_1[['bin_low', 'bin_up']].apply(myfunc, axis=1)
                df_1 = df_1[['total', 'var_name']]
                df_temp = pd.merge(df_temp, df_1, on='bin')
                df_temp['var_name_raw'] = k.split(sep='_BIN')[0]
                df_score = pd.concat([df_score, df_temp], axis=0)
            ##离散变量的计算
            elif k.split(sep='_BIN')[0] in dict_disc_bin.keys():
                df_temp = pd.merge(df_temp, dict_disc_bin[k.split(sep='_BIN')[0]], on='bin')
                df_temp['var_name_raw'] = k.split(sep='_BIN')[0]
                df_score = pd.concat([df_score, df_temp], axis=0)

    df_score['score_base'] = base_points
    return df_score, dict_bin_score, params_A, params_B, base_points

##计算样本分数
def cal_score(df_1, dict_bin_score, dict_cont_bin, dict_disc_bin, base_points):
    ##先对原始数据分箱映射，然后，用分数字典dict_bin_score映射分数，基础分加每项的分数就是最终得分
    df_1.reset_index(drop=True, inplace=True)
    df_all_score = pd.DataFrame()
    ##连续变量
    for i in dict_cont_bin.keys():
        if i in dict_bin_score.keys():
            df_all_score = pd.concat([df_all_score, varbin_meth.cont_var_bin_map(df_1[i], dict_cont_bin[i]).map(dict_bin_score[i])], axis=1)
    ##离散变量
    for i in dict_disc_bin.keys():
        if i in dict_bin_score.keys():
            df_all_score = pd.concat([df_all_score, varbin_meth.disc_var_bin_map(df_1[i], dict_disc_bin[i]).map(dict_bin_score[i])], axis=1)

    df_all_score.columns = [x.split(sep='_BIN')[0] for x in list(df_all_score.columns)]
    df_all_score['base_score'] = base_points
    df_all_score['score'] = df_all_score.apply(sum, axis=1)
    df_all_score['target'] = df_1.target
    return df_all_score


def evl_score(df_all_score):
    df_all_score.score.max()
    df_all_score.score.min()
    ##简单的分数区间计算
    score_bin   = np.arange(330, 660, 30)
    good_total  = sum(df_all_score.target == 0)
    bad_total   = sum(df_all_score.target == 1)
    bin_rate    = []
    bad_rate    = []
    ks = []
    good_num = []
    bad_num = []
    for i in range(len(score_bin) - 1):
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

    df_result = pd.DataFrame({'good_num': good_num, 'bad_num': bad_num, 'bin_rate': bin_rate,
                              'bad_rate': bad_rate, 'ks': ks})
    print('######计算结果如下###########')
    print(df_result)
    return

##计算整体PSI值
def cal_psi(df_raw, df_test, score_min, score_max, step):
    ##df_raw:pd.DataFrame训练集(线下数据)
    ##df_test:pd.DataFrame测试集(线上数据)
    score_bin = np.arange(score_min, score_max + step, step)
    total_raw = df_raw.shape[0]
    total_test = df_test.shape[0]
    psi = []
    total_all_raw = []
    total_all_test = []
    for i in range(len(score_bin) - 1):
        total_1 = sum((df_raw.score >= score_bin[i]) & (df_raw.score < score_bin[i + 1]))
        total_2 = sum((df_test.score >= score_bin[i]) & (df_test.score < score_bin[i + 1]))
        if total_2 == 0:
            total_2 = 1
        if total_1 == 0:
            total_1 = 1
        psi.append((total_1 / total_raw - total_2 / total_test) * (np.log((total_1 / total_raw) / (total_2 / total_test))))
        total_all_raw.append(total_1)
        total_all_test.append(total_2)
    totle_psi = sum(psi)
    return totle_psi, total_all_raw, total_all_test


##计算单调性指标
def cal_kendall_tau(df_1, score_min, score_max, step, label='target'):
    score_bin = np.arange(score_min, score_max + step, step)
    bin_num = []
    for i in range(len(score_bin) - 1):
        df_temp = df_1.loc[(df_1.score >= score_bin[i]) & (df_1.score < score_bin[i + 1])]
        bin_num.append(df_temp[label].sum())
    concordant_pair = 0
    discordant_pair = 0
    for j in range(0, len(bin_num) - 1):
        if bin_num[j] < bin_num[j + 1]:
            discordant_pair += 1
        else:
            concordant_pair += 1
    ktau = (concordant_pair - discordant_pair) / (len(bin_num) * (len(bin_num) - 1) / 2)
    return ktau


