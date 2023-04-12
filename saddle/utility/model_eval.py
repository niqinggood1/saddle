from sklearn.metrics import classification_report,precision_score

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    print(lines[0])
    d_l= []
    for line in lines[2:]:
        # print( line )
        row_data = line.lstrip(' ').split('      ')
        # print(row_data)
        if len( row_data ) == 5:  d_l.append( row_data )
    df = pd.DataFrame( d_l,columns=['class','precision','recall','f1_score','support'] )
    return df
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,recall_score
import pandas as pd
def model_evl( type='classifier',y_true=[], y_pred=[]  ):
    if type=='classifier':
        report      = classification_report( y_true, y_pred )  #confusion_matrix 混淆矩阵也要加上
        statadf     = classifaction_report_csv( report  )
        conf_mat    = confusion_matrix(y_true, y_pred  )
        #print('conf_mat:',conf_mat)
        mat_df      = pd.DataFrame( conf_mat, columns=['预测正例','预测负例'], index=['真实正例','真实负例'] ).reset_index()
        statadf['*']= '*'
        statadf     = pd.merge(statadf,mat_df,left_index=True,right_index=True,how='left')
        roc_auc     = roc_auc_score( y_true, y_pred   )
        statadf['roc']  =''
        statadf     =statadf.fillna('')
        statadf.loc[0,'auc']=roc_auc
        # ##计算fpr与tpr
        # fpr, tpr, thresholds = roc_curve(y_test, y_score_test)
        # ks = max(tpr - fpr)

        recall_value    = recall_score( y_true, y_pred)   #召回率 TP/(TP+FN)  就是正样本的覆盖率，与明敏度一个意思
        precision_value = precision_score(y_true, y_pred) #TP/(TP+FP)
        accuracy        = accuracy_score( y_true, y_pred)
        b2              =  1**2
        F_Score         =  (1+b2)*precision_value*recall_value/( b2*precision_value + recall_value   )   #F-Measure
        print(conf_mat)
        print('Validation set:  model recall is {0},and percision is {1}'.format(recall_value,
                                                                                 precision_value))
        statadf.loc[0,'recall']     =recall_value
        statadf.loc[0, 'precision'] = precision_value
        statadf.loc[0, 'accuracy']  = accuracy
        statadf.loc[0, 'bb']        = b2
        statadf.loc[0, 'F_Score']   = F_Score
        #  precision, recall           = precision_recall_curve( y_true, y_pred ) #这个是画P-R曲线
        # statadf.loc[0, 'precision'] = precision
        # statadf.loc[0, 'recall']    = recall
        # fpr, tpr, thresholds = roc_curve(y_test, y_score_test)
        # ##计算auc值
        # roc_auc = auc(fpr, tpr)
        # ar = 2 * roc_auc - 1
    if type=='regressor':
        R_squared           = r2_score(y_true, y_pred)
        meanSqureError      = mean_squared_error(  y_true, y_pred )
        meanAbsoluteError   = mean_absolute_error( y_true, y_pred )
        statadf= pd.DataFrame( [ [R_squared,meanSqureError,meanAbsoluteError] ] ,columns=['R_squared','mean_squared_error','mean_absolute_error'] )
    print(statadf)

    return  statadf


from sklearn.metrics import confusion_matrix
def confusion( label_true, label_test ):
    ret = confusion_matrix( label_true, label_test )
    return  ret

from sklearn.metrics import roc_curve,confusion_matrix,recall_score, auc
def plot_ks(y_test,y_score_test):
    import matplotlib.pyplot as plt
    fpr, tpr, thresholds = roc_curve(y_test, y_score_test)
    ####计算AR。gini等
    roc_auc = auc(fpr, tpr)
    ks = max(tpr - fpr)
    ar = 2 * roc_auc - 1
    gini = ar
    print('test set:  model AR is {0},and ks is {1}'.format(ar,ks))
    ####ks曲线
    plt.figure(figsize=(10, 6))
    fontsize_1 = 12
    plt.plot(np.linspace(0, 1, len(tpr)), tpr, '--', color='black', label='正样本洛伦兹曲线')
    plt.plot(np.linspace(0, 1, len(tpr)), fpr, ':', color='black', label='负样本洛伦兹曲线')
    plt.plot(np.linspace(0, 1, len(tpr)), tpr - fpr, '-', color='grey')
    plt.grid()
    plt.xticks(fontsize=fontsize_1)
    plt.yticks(fontsize=fontsize_1)
    plt.xlabel('概率分组', fontsize=fontsize_1)
    plt.ylabel('累积占比%', fontsize=fontsize_1)
    plt.legend(fontsize=fontsize_1)
    print('max(tpr - fpr):',max(tpr - fpr))
    plt.show()
    return

##计算整体PSI值
import numpy as np
def cal_psi(df_raw, df_test, score_min, score_max, step):
    ##df_raw:pd.DataFrame训练集(线下数据)
    ##df_test:pd.DataFrame测试集(线上数据)
    score_bin  = np.arange( score_min, score_max + step, step )
    total_raw  = df_raw.shape[0]
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