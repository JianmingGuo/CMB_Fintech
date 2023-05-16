# 在原处理结果的基础上增加view时间戳相关信息

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle

def data_prep(x):
    if type(x) == float:
        assert np.isnan(x)
        return np.nan
    a, b = x.split(" ")
    yy, mm, dd = a.split("-")
    hh, mimi, _ = b.split(":")
    if '上午' in hh:
        hh = int(hh[2:])
    elif '下午' in hh:
        hh = int(hh[2:]) + 12
    return (int(yy)-1492)*12*30*24*60 + (int(mm)-8)*30*24*60 + int(dd)*24*60 + int(hh)*60 + int(mimi)

def time_feature(df, length):
    wid = df.groupby('cust_wid')
    tmin = wid.agg('min').rename(columns={'acs_tm':'time_min'})
    tmax = wid.agg('max').rename(columns={'acs_tm':'time_max'})
    tmean = wid.agg('mean').rename(columns={'acs_tm':'time_mean'})
    tmedian = wid.agg('median').rename(columns={'acs_tm':'time_median'})
    time = tmin.join([tmax, tmean, tmedian])
    time['time_range'] = time['time_max'] - time['time_min']
    time['frequence'] = time['time_range'] / length
    return time

def pages_feature(df, index, train=True):
    pages = df['page_id'].map(lambda x:str(x))
    users = df['cust_wid'].map(lambda x:str(x))
    # print(users.shape)
    if train:
        encoder_label_pages = LabelEncoder()
        pages_tmp = encoder_label_pages.fit_transform(pages)
        with open('/work/process_0505/encoder_label_pages.pkl', 'wb') as f:
            pickle.dump(encoder_label_pages, f)
        encoder_one_pages = OneHotEncoder()
        encoder_one_pages.fit(pages_tmp.reshape(-1,1))
        with open('/work/process_0505/encoder_one_pages.pkl', 'wb') as f:
            pickle.dump(encoder_one_pages, f)
        
        encoder_label_users = LabelEncoder()
        # 先用base的所有用户训练encoder，再在view上transform
        encoder_label_users.fit(index)
        users = encoder_label_users.transform(users)
        encoder_one_users = OneHotEncoder()
        users = encoder_one_users.fit_transform(users.reshape(-1,1))

    else:
        encoder_label_pages = pickle.load(open('/work/process_0505/encoder_label_pages.pkl', 'rb'))
        encoder_one_pages = pickle.load(open('/work/process_0505/encoder_one_pages.pkl', 'rb'))
        encoder_label_users = LabelEncoder()
        encoder_label_users.fit(index)
        users = encoder_label_users.transform(users)
        encoder_one_users = OneHotEncoder()
        users = encoder_one_users.fit_transform(users.reshape(-1,1)) 

    pages = encoder_label_pages.transform(pages)
    pages = encoder_one_pages.transform(pages.reshape(-1,1))


    cnt = users.T.dot(pages).toarray()

    
    df_pages = pd.DataFrame(data=cnt,
            columns = ['pages_id_' + str(i) for i in range(cnt.shape[1])],
            index = index )
    return df_pages

def feature_columns():
    # names = ['cty_cd_' + str(i) for i in range(32)]
    columns_base = ['cust_wid', 'label', 'age', 'gdr_1', 'gdr_2']
    # names = ['amt_choice_' + str(i) for i in range(20)]
    # names1 = ['trx_cd_' + str(i) for i in range(42)]
    columns_trx = ['trx_len', 'amt_max', 'amt_mean', 'amt_sum', 'amt_min'] 
    # names = ['page_id_' + str(i) for i in range(64)]
    columns_view = ['view_len']
    # time = ['time_min','time_max','time_mean','time_median','range']
    return columns_base+columns_trx+columns_view


def main():
    infoDf = pd.read_csv('/tasks/5758/preprocess/data_preprocess_train_all.csv')
    infoDf = infoDf[feature_columns()]
    infoDf = infoDf.set_index('cust_wid')
    infoDf = infoDf.sort_index()
    info_view = pd.read_csv('/data/train_view.csv', encoding='gbk')
    info_view = info_view.sort_values(by='cust_wid')
    info_view['acs_tm'] = info_view['acs_tm'].apply(data_prep)
    time = time_feature(info_view[['cust_wid', 'acs_tm']], infoDf['view_len'])
    print("time", time)
    print(time.shape)
    pages = pages_feature(info_view, index = infoDf.index)
    print("page", pages)
    print(pages.shape)

    infoDf = infoDf.join([time, pages])
    infoDf.to_csv('/work/process_0505/train_0505_onehot.csv')

if __name__ == '__main__':
    main()

