#!/usr/local/bin/python
#-*- coding:utf-8 -*-
#coding=utf8
import pandas as pd
import numpy as np
#from fbprophet import Prophet

import time
from datetime import datetime
from xlrd import xldate_as_tuple
import copy
if __name__ == '__main__':
    file_name = u'./data/专家模型需要预测的数据.xlsm'
    df = pd.read_excel(   file_name, sheet_name='Sheet1')
    
    region= u'机构';    df =  df[df[u'机构']==region]
    print( df.head(10) )
    df['ds'] =  df[u'年月'].apply( lambda i : datetime(*xldate_as_tuple(i, 0)) if type(i)==int else i )
    df['y']  = df['NBEV']
    print( df[['ds','y']].tail() ) 










#def get_predict_prophet_optimize(x,diff):
    #rng = pd.date_range('1/1/2014', periods=len(x), freq='M')
    #x = pd.DataFrame( {'y':x, 'ds':rng }   ) #index=rng ,columns=['y']
    ##x['ds'] = x.index;
    ##print(x.tail())
    #m = Prophet( yearly_seasonality=12,weekly_seasonality=False,daily_seasonality=False,\
                 #) #mcmc_samples=50 seasonality_mode='multiplicative'
    #m.fit(x)
    #future = m.make_future_dataframe(periods=diff,freq='M')
    #forecast = m.predict(future)
    #print('forecast *** :',forecast.tail(5))
    #pred_result= forecast.tail(diff)['yhat'].tolist()
    #return pred_result

#import time
#from datetime import datetime
#from xlrd import xldate_as_tuple
#import copy
#if __name__ == '__main__':
    #dir = './data/' #
    #file_name = '专家模型需要预测的数据.xlsm'
    #df = pd.read_excel(dir + file_name, sheet_name='Sheet1')
    #print( df.head(10) )
    #for k in df.columns[:20]:
        #print(k, end=',')
    #region= '机构'
    #df =  df[df['机构']==region]
    #df.index = [datetime(*xldate_as_tuple(i, 0)) if type(i)==int  else i for i in df.年月  ]
    #print(df.tail())
    #max_date = max( df.index ); start_predict_date=  max_date.strftime('%Y-%m-%d')#time.strftime("%Y-%m-%d",max_date)
    #print('max_date', max_date, start_predict_date  )
    #del df['年月'],df['机构']
    ## df =df[ df.index >='2015-1-1' ][['月末人力']]
    #for diff in range(1,7):

        #final_predict_df = copy.deepcopy(  df )
        #old = max( final_predict_df.index )
        #for  date in range(diff):
            #new = datetime(old.year + (old.month == 12), old.month == 12 or old.month + 1, 1, 0, 0, 0)
            #for key in final_predict_df.columns : final_predict_df.loc[new,  key ] = 0
            #old=new
        #for key in  final_predict_df.columns:#['NBEV']:
            #print(' key--begin '*3,key)
            #final_predict_df[key]   =   final_predict_df[key].rolling(47,47).apply(  lambda x : get_predict_prophet_optimize(x ,diff)[diff-1]   ).shift(diff)

            #print( final_predict_df.tail(8) )
            #print('key--end ' * 5, key)
            ##break

        #final_predict_df.index = final_predict_df.index.date
        #final_predict_df.to_excel( 'prophet预测T%d_'%(diff) + region+ '_'+file_name,index_label ='年月' )
        ##break






