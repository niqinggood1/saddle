def cartesian():
    ##近似笛卡尔积：特征交叉
    var_cross = ['amount_BIN', 'income_rate_BIN', 'residence_info_BIN',
                 'age_BIN', 'num_credits_BIN', 'dependents_BIN', 'status_account_BIN',
                 'credit_history_BIN', 'purpose_BIN', 'svaing_account_BIN',
                 'present_emp_BIN', 'personal_status_BIN', 'property_BIN', 'housing_BIN', 'job_BIN']
    list_name = []
    for i in range(len(var_cross) - 1):
        print(var_cross[i])
        for j in range(i + 1, len(var_cross)):
            # print(var_1[i]+'_Cross_'+var_1[j])
            list_name.append(var_cross[i] + '_Cross_' + var_cross[j])
            data_train_bin[var_cross[i] + '_Cross_' + var_cross[j]] = data_train_bin[[var_cross[i], var_cross[j]]].apply(func_s, axis=1)
            data_test_bin[var_cross[i] + '_Cross_' + var_cross[j]] = data_test_bin[[var_cross[i], var_cross[j]]].apply(func_s, axis=1)

    return

def func_s(x):
    return str(x[0]) + '_Cross_' + str(x[1])

def gdbt_features():
    ####GBDT模型
    GBDT_model = GradientBoostingClassifier(subsample=0.8, max_features=0.8, validation_fraction=0.1,
                                            n_iter_no_change=3, random_state=0, n_estimators=20,
                                            max_depth=2, learning_rate=0.1)
    ##训练GBDT模型
    GBDT_model_fit = GBDT_model.fit(x_train, y_train)

    ###用apply方法得到树的映射结果
    train_new_feature = GBDT_model_fit.apply(x_train)[:, :, 0]
    test_new_feature = GBDT_model_fit.apply(x_test)[:, :, 0]
    np.unique(train_new_feature[:, 1])
    ##进行One-hot编码
    enc = OneHotEncoder(dtype='int').fit(train_new_feature)
    df_train = pd.DataFrame(enc.transform(train_new_feature).toarray())
    df_test = pd.DataFrame(enc.transform(test_new_feature).toarray())
    ##合并得到新的数据集
    x_train_1 = np.hstack([x_train, df_train])
    x_test_1 = np.hstack([x_test, df_test])

def gen_fm_feature(df_all):
    ###df转为字典
    df_all = df_all[var_all].to_dict(orient='records')
    x_train = data_train[var_all].to_dict(orient='records')
    x_test = data_test[var_all].to_dict(orient='records')
    ###字典转为稀疏矩阵
    model_dictV = DictVectorizer().fit(df_all)
    x_train = model_dictV.fit_transform(x_train)
    x_test = model_dictV.transform(x_test)
    return x_train,x_test

def fm( ):
    fm = pylibfm.FM(num_factors=5, num_iter=500, verbose=True, task="classification",
                    initial_learning_rate=0.0001, learning_rate_schedule="optimal")
    fm.fit(x_train, y_train)
    return

import variable_bin_methods as varbin_meth