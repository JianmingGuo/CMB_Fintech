import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
import numpy as np

# label: 0 or 1
def GetDataBase(train=True):
    feature_columns = ['age', 'gdr_cd', 'cty_cd']
    label_column = ['label']
    if train:
        info_base = pd.read_csv('/data/train_base.csv')
        # info_base = info_base.dropna()
        # task1 label
        info_base['label'] = info_base['label'].map(lambda s:0 if s==0 else 1)
        encoder = LabelEncoder()
        info_base['cty_cd'] = encoder.fit_transform(info_base['cty_cd'])
        with open('/work/baseline_reg/encoder_cty_cd.pkl', 'wb') as f:
            pickle.dump(encoder, f)
        encoder_gdr = LabelEncoder()
        info_base['gdr_cd'] = encoder_gdr.fit_transform(info_base['gdr_cd'])
        with open('/work/baseline_reg/encoder_gdr_cd.pkl', 'wb') as f:
            pickle.dump(encoder_gdr, f)
        df_train = info_base.sample(frac=0.8, replace=False)
        df_val = info_base[~info_base.index.isin(df_train.index)]
        df_train_x = df_train[feature_columns]
        df_train_y = df_train[label_column]
        df_val_x = df_val[feature_columns]
        df_val_y = df_val[label_column]
        return df_train_x, df_train_y, df_val_x, df_val_y
    else:
        info_base = pd.read_csv('/data/testa_base.csv')
        info_base = info_base.sort_values(by='cust_wid')
        # info_base = info_base.dropna()
        encoder_gdr = pickle.load(open('/work/baseline_reg/encoder_gdr_cd.pkl','rb'))
        info_base['gdr_cd'] = encoder_gdr.transform(info_base['gdr_cd'])
        encoder_cty = pickle.load(open('/work/baseline_reg/encoder_cty_cd.pkl', 'rb'))
        # test中存在部分cty没有在train中出现过
        info_base['cty_cd'] = info_base['cty_cd'].map(lambda s:-1 if s not in encoder_cty.classes_ else s)
        encoder_cty.classes_ = np.append(encoder_cty.classes_, -1)
        info_base['cty_cd'] = encoder_cty.transform(info_base['cty_cd'])
        return info_base[feature_columns], info_base[['cust_wid']]
