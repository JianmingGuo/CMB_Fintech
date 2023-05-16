from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
import pandas as pd
import numpy as np
import random
from gensim.models import Word2Vec
from tqdm import tqdm

def GetFeatureinTrx(infoDf, wid, encoder_trx_cd):
    # wid = wid.replace('A','T')
    if wid not in infoDf['cust_wid'].values:
        return [np.nan] * 67
    info_id = infoDf[infoDf['cust_wid'].isin([wid])]

    # cd one-hot之后累加
    cd = encoder_trx_cd.transform(info_id['trx_cd'].values.reshape(-1,1)).toarray()
    amt = info_id['trx_amt'].values
    tmp = np.multiply(cd, amt.reshape([-1,1]))
    trx_cd_list = tmp.sum(axis=0).tolist()
    cd_names = ['trx_cd_' + str(i) for i in range(len(trx_cd_list))]

    # 记录数
    trx_len = len(info_id)

    # amt max, mean, sum
    amt_max, amt_mean, amt_sum, amt_min = np.max(amt), np.mean(amt), np.sum(amt), np.min(amt)
    # amt 采样20个
    amt_choice = np.resize(amt, (20,))


    # trx_tm
    # TODO: 添加时间特征

    # 5 + 20 + 42
    return [trx_len, amt_max, amt_mean, amt_sum, amt_min] + list(amt_choice) + trx_cd_list

def assemble_x(w2vec:dict,sentences):
    sen_vs=[]
    for sen in sentences:
        # max_len=max(max_len,len(sen))
        v=np.vstack([w2vec[w] for w in sen])
        sen_v=v.mean(axis=0)
        sen_vs.append(sen_v)
    return np.array(sen_vs)

def GetFeatureInview(infoDf, wid, embedding):
    if wid not in infoDf['cust_wid'].values:
        return [np.nan] * 129
    info_id = infoDf[infoDf['cust_wid'].isin([wid])]

    # 记录数
    view_len = len(info_id)

    # page embedding sum, 64
    pages_id = list(map(str, info_id['page_id']))
    x = assemble_x(embedding.wv, pages_id)
    x = x.sum(axis=0).tolist()

    #  1+64
    return [view_len] + x

def columns():
    names = ['cty_cd_' + str(i) for i in range(32)]
    columns_base = ['cust_wid', 'label', 'age', 'gdr_1', 'gdr_2'] + names
    names = ['amt_choice_' + str(i) for i in range(20)]
    names1 = ['trx_cd_' + str(i) for i in range(42)]
    columns_trx = ['trx_len', 'amt_max', 'amt_mean', 'amt_sum', 'amt_min'] + names + names1
    names = ['page_id_' + str(i) for i in range(64)]
    columns_view = ['view_len'] + names
    return columns_base+columns_trx+columns_view


def preprocess(k):
    info_trx = pd.read_csv('/data/train_trx.csv')
    info_base = pd.read_csv('/data/train_base.csv')
    info_view = pd.read_csv('/data/train_view.csv', encoding='gbk')

    # 对trx_cd 进行one-hot编码
    # 因为one-hot只能对数值进行，所有先labelencoder
    encoder_trx_cd_label = LabelEncoder()
    info_trx['trx_cd'] = encoder_trx_cd_label.fit_transform(info_trx['trx_cd'].values)
    encoder_trx_cd = OneHotEncoder()
    encoder_trx_cd.fit(info_trx['trx_cd'].values.reshape(-1,1))

    with open('/work/preprocess/encoder_trx_cd_label.pkl', 'wb') as f:
        pickle.dump(encoder_trx_cd_label, f)
    with open('/work/preprocess/encoder_trx_cd_one_hot.pkl', 'wb') as f:
        pickle.dump(encoder_trx_cd, f)
    
    # 对view里的page_id 进行embedding编码
    page = info_view['page_id']
    page = list(map(str, page))
    embedding_page = Word2Vec(sentences=page, vector_size=64)
    embedding_page.save('/work/preprocess/embedding_page.pt')
    embedding_page.wv.save('/work/preprocess/embedding_page.wv')
    # from gensim.models import KeyedVectors
    # wv = KeyedVectors.load('embedding_page.wv',mmap='r')

    # 对base里的city进行embedding编码
    city = info_base['cty_cd']
    city = list(map(str, city))
    embedding_city = Word2Vec(sentences=city, vector_size=32)
    embedding_city.save('/work/preprocess/embedding_city.pt')
    embedding_city.wv.save('/work/preprocess/embedding_city.wv')

    n = len(info_base)
    columns_name = columns()
    for i in range(k):
        # tmp = info_base[int(i*n/k): int((i+1)*n/k)]
        tmp = info_base[20000:]
        # print(int(i*n/k), int((i+1)*n/k))

        data = []
        for index, row in tqdm(tmp.iterrows(), total=len(tmp)):
            if index%100==0:
                print(index)
            data_sub = [row['cust_wid'], row['label'], row['age']]
            if row['gdr_cd'] == 'F':
                data_sub += [1,0]
            elif row['gdr_cd'] == 'M':
                data_sub += [0,1]
            else:
                data_sub += random.choice([[0,1],[1,0]])

            # 32
            city = assemble_x(embedding_city.wv, str(row['cty_cd']))
            city = city.sum(axis=0).tolist()

            # print(data_sub)
            # ----
            # test = ['T00000', 'T00001', 'T00002', 'T00003', 'T00004', 'T00005']
            # tmp = GetFeatureinTrx(info_trx, test[index], encoder_trx_cd)
            # if index == 5:
            #     break
            # ----

            trx = GetFeatureinTrx(info_trx, row['cust_wid'], encoder_trx_cd)
            view = GetFeatureInview(info_view, row['cust_wid'], embedding_page)
            

            data_sub = data_sub + city + trx + view
            data.append(data_sub)
            
            # data_sub 5+32=37
            # trx 67
            # view 65
            assert len(data_sub) == 169
        df = pd.DataFrame(data, columns=columns_name)
        file = '/work/preprocess/data_preprocess_train_2_5.csv'
        df.to_csv(file, index=False)
    # return data

def main():
    data = preprocess()
    names = ['cust_wid', 'label', 'age', 'gdr_1', 'gdr_2', 'trx_len', 'amt_max', 'amt_mean', 'amt_sum']
    cd_names = ['trx_cd_' + str(i) for i in range(42)]
    names += cd_names
    df = pd.DataFrame(data, columns=names)
    df.to_csv("data_preprocess.csv", index=False)

if __name__ == '__main__':
    preprocess(1)

