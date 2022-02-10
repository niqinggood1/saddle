import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.holtwinters import ExponentialSmoothing,SimpleExpSmoothing
def get_predict_ets( a_in, T=1 ):
    model   = ExponentialSmoothing(a_in, seasonal= 'add', seasonal_periods=12  ).fit(optimized=True) #,use_basinhopping=True  mul add
    pred    = model.predict(start= len(a_in), end= len(a_in)+T-1 )
    return pred

def get_predict_ets_optimize(a_in, T=1):
    a_in    = np.array([i for i in a_in if math.isnan(i) == False])
    model   = ExponentialSmoothing(a_in, trend='additive').fit(optimized=True)  #optimized=True # ,smoothing_level=0.5 additive ExponentialSmoothing mul add  ,use_basinhopping=True
    pred    = model.predict(start=len(a_in), end=len(a_in)+T-1)
    return pred

import pandas as pd
from pyramid.arima import auto_arima
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
def decompose_arima_predict(x,n_periods,freq='M',begin_date='1/1/2014'):
    #时间序做分解,分别建模，做预测
    #x是建模的原始数据，n_periods是预测未来几期
    #返回未来n_periods的预测
    rng             = pd.date_range( begin_date, periods=len(x), freq=freq )
    x               = pd.Series(x, index=rng)
    decomposition   = seasonal_decompose(x)
    trend           = decomposition.trend.dropna()      # .fillna(0)
    seasonal        = decomposition.seasonal.dropna()   # .fillna(0)
    residual        = decomposition.resid.dropna()      # .fillna(0)
    # print('trend',trend)
    #     seasonal_model    = auto_arima( seasonal.values,trace=True,error_action='ignore', suppress_warnings=True )
    #     seasonal_model.fit(             seasonal.values,seasonal_order=(0, 0, 0, 12) )  #  order=(0, 0, 0) seasonal_order=(0, 0, 0, 1);
    #     seasonal_forecast = seasonal_model.predict( n_periods=6  )

    trend_model     = auto_arima(trend.values, trace=True, error_action='ignore', suppress_warnings=True)
    trend_model.fit(trend.values)  # order=(0, 0, 0) seasonal_order=(0, 0, 0, 1);
    trend_forecast  = trend_model.predict(n_periods=n_periods)

    residual_model  = auto_arima(residual, trace=True, error_action='ignore', suppress_warnings=True)
    residual_model.fit(residual)
    residual_forecast = residual_model.predict(n_periods=n_periods)

    len_seasonal = len(seasonal)
    seasonal_forecast = [seasonal.values[i + len_seasonal - 12] for i in range(n_periods)]

    pred_result = pd.Series(trend_forecast)  # #trend_forecast  trend_forecast
    pred_result =  pred_result.add(  seasonal_forecast ).add( residual_forecast  )
    return pred_result


def model_decompose(ts):
    decomposition = seasonal_decompose(ts, freq=7)
    ts_trend      = decomposition.trend
    ts_seasonal   = decomposition.seasonal
    ts_resid      = decomposition.resid
    """
    使用分位数进行预测
    """
    d = ts_resid.describe()
    delta = d['75%'] - d['25%']
    ts_low_error, ts_high_error = (d['25%'] - 1 * delta, d['75%'] + 1 * delta)

    """
    将分解出来的趋势数据单独建模,通过差分的方法对ts2_trend转化为平稳序列，对ts2_trend部分进行ARIMA建模
    """
    # 因为分解出来的trend前3项和后3项会有nan,1阶查分后第一个数也会是nan值，所以要去nan
    ts_trend.count()
    ts_trend_diff = ts_trend.diff()
    ts_trend_diff.dropna(inplace=True)
    return ts_trend_diff, ts_trend, ts_seasonal, ts_low_error, ts_high_error


def exponential_smoothing(alpha, s):
    '''
    一次指数平滑
    :param alpha:  平滑系数
    :param s:      数据序列， list
    :return:       返回一次指数平滑模型参数， list
    '''
    s = np.array([i for i in s if math.isnan(i) == False])
    s_temp = [0 for i in range(len(s))]
    s_temp[0] = ( s[0] + s[1] + s[2] ) / 3
    for i in range(1, len(s)):
        s_temp[i] = alpha * s[i] + (1 - alpha) * s_temp[i-1]
    return s_temp

def test_decompose_arima_predict():
    dir = './data/'
    file_name = '专家模型需要预测的数据.xlsm'
    df = pd.read_excel(dir + file_name, sheet_name='Sheet1')
    print(df.head(10))
    for k in df.columns[:20]:
        print(k, end=',')
    region = '机构'
    df = df[df['机构'] == region]
    df.index = [datetime(*xldate_as_tuple(i, 0)) if type(i) == int else i for i in df.年月]  ## df.index.apply(lambda x : datetime( *xldate_as_tuple(x,0) ) )
    print(df.tail())
    max_date = max(df.index);
    start_predict_date = max_date.strftime('%Y-%m-%d')  # time.strftime("%Y-%m-%d",max_date)
    print('max_date', max_date, start_predict_date)
    del df['年月'], df['机构']
    for diff in range(2, 7):
        final_predict_df = copy.deepcopy(df)
        old = max(final_predict_df.index)
        for date in range(diff):
            new = datetime(old.year + (old.month == 12), old.month == 12 or old.month + 1, 1, 0, 0, 0)
            for key in final_predict_df.columns: final_predict_df.loc[new, key] = 0
            old = new
        for key in final_predict_df.columns:
            print(' key--begin ' * 3, key)
            final_predict_df[key] = final_predict_df[key].rolling(25, 25).apply(lambda x: get_predict_arima_optimize(x, diff)[diff - 1]).shift(diff)
            print(final_predict_df.tail(8))
            print('key--end ' * 5, key)
            # break
        final_predict_df.index = final_predict_df.index.date
        final_predict_df.to_excel('arima预测T%d_' % (diff) + region + '_' + file_name, index_label='年月')


import time
from datetime import datetime
from xlrd import xldate_as_tuple
import copy
import math
if __name__ == '__main__':
    dir='./data/'
    file_name = '专家模型需要预测的数据.xlsm'
    df = pd.read_excel( dir+file_name, sheet_name='Sheet1')
    print('columns_len', len(df.columns))
    for k in df.columns[:20]:
        print(k, end=',')
    region= '机构'
    df =  df[df['机构']==region]
    df.index = [datetime(*xldate_as_tuple(i, 0)) if type(i)==int  else i for i in df.年月  ] ## df.index.apply(lambda x : datetime( *xldate_as_tuple(x,0) ) )
    print(df.tail())
    max_date = max( df.index ); start_predict_date=  max_date.strftime('%Y-%m-%d')#time.strftime("%Y-%m-%d",max_date)
    print('max_date', max_date, start_predict_date  )
    del df['年月'],df['机构']


    for diff in range(1,7):

        final_predict_df = copy.deepcopy(  df )
        old = max( final_predict_df.index )
        for  date in range(diff):
            new = datetime(old.year + (old.month == 12), old.month == 12 or old.month + 1, 1, 0, 0, 0)
            for key in final_predict_df.columns : final_predict_df.loc[new,  key ] = np.float('NaN')
            old=new
        for key  in final_predict_df.columns:  #['NBEV']: ['NBEV']
            print(' key--begin '*3,key)

            final_predict_df['shift'] = final_predict_df[key].shift(1)

            final_predict_df['shift'] = list(
                map(lambda x, y, z: 1.0 * y if (x.month == 1) else z, final_predict_df.index, final_predict_df['shift'], final_predict_df[key]))
            final_predict_df['shift'] = list(map(lambda x, y: y if math.isnan(x) else x, final_predict_df['shift'], final_predict_df[key]))

            if diff<=2:
                final_predict_df['seanson'] = final_predict_df[key].rolling(84, 44).apply(lambda x: get_predict_ets(x, diff)[diff - 1]).shift(diff)
                final_predict_df['trend'] = final_predict_df['shift'].rolling( 13,8).apply(  lambda x : get_predict_ets_optimize(x,diff )[diff-1]   ).shift(diff)#rolling( 14+(diff-1)*3, 9+(diff-1)*3).apply(lambda x: get_predict_ets_optimize(x,diff )[diff-1]).shift(diff)  # _optimize
                final_predict_df[key] = list(map(lambda x, y, z: y if (x.month in [12, 1, 2, 3] or math.isnan(y)) else z, final_predict_df.index,final_predict_df['seanson'], final_predict_df['trend']))
            else:
                final_predict_df['seanson'] = final_predict_df[key].rolling(35,35).apply(lambda x: get_predict_ets(x, diff)[diff - 1]).shift(diff)
                final_predict_df['trend']=  final_predict_df['seanson']
                final_predict_df[key]    =  final_predict_df['seanson']
            del final_predict_df['shift'],final_predict_df['seanson'],final_predict_df['trend']
            print( final_predict_df.tail(8) )
            print('key--end ' * 5, key)
        final_predict_df.index = final_predict_df.index.date
        final_predict_df.to_excel( 'ets_optimize预测T%d_'%(diff) + region+ '_'+file_name,index_label ='年月' )

from pandas import read_csv
from datetime import datetime
# load data
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')
dataset = read_csv('PRSA_data_2010.1.1-2014.12.31.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
# manually specify column names
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# mark all NA values with 0
dataset['pollution'].fillna(0, inplace=True)
# drop the first 24 hours
dataset = dataset[24:]
# summarize first 5 rows
print(dataset.head(5))
# save to file
dataset.to_csv('pollution.csv')

from pandas import read_csv
from matplotlib import pyplot
# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1

from pandas import read_csv
from matplotlib import pyplot
# load dataset
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# plot each column
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1

pyplot.show()

import pandas as pd
import numpy as np
from fbprophet import Prophet
df = pd.read_excel('C:/Users/yangge/Desktop/prophet-master/examples/example_wp_peyton_manning.xlsx')
df['y'] = np.log(df['y'])
playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))#季后赛和超级碗比赛特别日期
m = Prophet(holidays=holidays)#指定节假日参数，其它参数以默认值进行训练
m.fit(df)#对过去数据进行训练
future = m.make_future_dataframe(freq='D',periods=365)#建立数据预测框架，数据粒度为天，预测步长为一年
forecast =m.predict(future)
m.plot(forecast).show()#绘制预测效果图
m.plot_components(forecast).show()#