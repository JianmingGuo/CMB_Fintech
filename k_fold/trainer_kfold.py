import xgboost as xgb
from sklearn.metrics import accuracy_score, auc, roc_curve, precision_score, recall_score
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, KFold
from utils import f2_scorer, f2_eval


def feature_columns():
    # names = ['cty_cd_' + str(i) for i in range(32)]
    columns_base = ['age', 'gdr_1', 'gdr_2']
    # names = ['amt_choice_' + str(i) for i in range(20)]
    # names1 = ['trx_cd_' + str(i) for i in range(42)]
    columns_trx = ['trx_len', 'amt_max', 'amt_mean', 'amt_sum', 'amt_min'] 
    # names = ['page_id_' + str(i) for i in range(64)]
    names = ['pages_id_' + str(i) for i in range(2327)]
    columns_view = ['view_len'] + names
    time = ['time_min','time_max','time_mean','time_median' ,'time_range','frequence']
    return columns_base+columns_trx+columns_view + time

def get_data(cv=False):
    feature = feature_columns()
    print('--------')
    print("length of feature", len(feature))
    print('--------')
    infoDf = pd.read_csv('/tasks/10632/process_0505/train_0505_onehot.csv')
    # 含有thresh个非0的值
    # infoDf = infoDf.dropna(axis=0, thresh=120)
    if not cv:
        infoDf['label'] = infoDf['label'].map(lambda s:0 if s==0 else 1)
        return infoDf
    df_train = infoDf.sample(frac=0.8, replace=False)
    df_val = infoDf[~infoDf.index.isin(df_train.index)]
    # df_train_x = df_train.iloc[:,2:]
    df_train_x = df_train[feature]
    df_train_y = df_train['label']
    # df_val_x = df_val.iloc[:, 2:]
    df_val_x = df_val[feature]
    df_val_y = df_val['label']
    dtrain = xgb.DMatrix(df_train_x, label = df_train_y)
    dval = xgb.DMatrix(df_val_x, label = df_val_y)
    return dtrain, dval, infoDf[feature], infoDf['label']

def ParamSearch():
    _, _, train_x, train_y = get_data(cv=True)
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',  # 多分类的问题          
        'gamma': 0.2,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 5,               # 构建树的深度，越大越容易过拟合
        'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.7,              # 随机采样训练样本
        'colsample_bytree': 0.7,       # 生成树时进行的列采样
        'min_child_weight': 3,
        'eta': 0.03,                  # 如同学习率
        'seed': 3407,
        'nthread':16,                  # cpu 线程数
        'eval_metric':'logloss',
    }

    hyparameter_grid = {
        'n_estimators': [300, 500, 1000, 2000, 3000],
        'eta': [0.03, 0.05, 0.1, 0.15],
        'min_child_weight': [3, 5, 7],
        'max_depth': [4,5,6,7],
        'gamma': [0.1, 0.15, 0.2, 0.25]

    }

    model = xgb.XGBRegressor(**params)
    random_cv = RandomizedSearchCV(estimator=model,
                    param_distributions=hyparameter_grid,
                    n_jobs=-1, n_iter=5, scoring=f2_scorer,
                    cv=5, verbose=3, random_state=42, return_train_score=True)
    random_cv.fit(train_x, train_y)
    print(random_cv.best_estimator_)

def trainer(dtrain, dval, saved_path, useTrainCV=False):
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',  # 多分类的问题          
        'gamma': 0.2,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 7,               # 构建树的深度，越大越容易过拟合
        'lambda': 3,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.7,              # 随机采样训练样本
        'colsample_bytree': 0.7,       # 生成树时进行的列采样
        'min_child_weight': 3,
        'silent': 0,                  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.15,                  # 如同学习率
        'seed': 3407,
        'nthread':16,                  # cpu 线程数
        'eval_metric':'logloss',
    }

    if useTrainCV:
        xgb_param = params.copy()
        print('---start CV----')
        cvresult = xgb.cv( xgb_param, dtrain,num_boost_round=5000, nfold=5, metrics=['auc'], early_stopping_rounds=50, stratified=True )
        params['n_estimators'] = int(cvresult.shape[0])
        print('Best number of trees = {}'.format(cvresult.shape[0]))

    # saved_path = '/work/reg_stage1_0505/reg_0505_onehot.json'

    watch_list = [(dtrain, 'train'), (dval, 'val')]
    num_round = 1000
    clf = xgb.train(params, dtrain, num_round, watch_list, feval=f2_eval, verbose_eval=100, early_stopping_rounds=50)
    # clf = xgb.train(params, dtrain, watch_list, verbose_eval=100)
    clf.save_model(saved_path)

    ans = clf.predict(dval, ntree_limit=clf.best_ntree_limit)
    ans = (ans >= 0.5)*1
    fpr, tpr, thersholds = roc_curve(dval.get_label(),ans,pos_label=1)
    roc_auc = auc(fpr,tpr)
    acc = accuracy_score(dval.get_label(), ans,normalize=True)
    print("auc",roc_auc)
    print("acc",acc)
    p = precision_score(dval.get_label(), ans, average='micro')
    r = recall_score(dval.get_label(), ans,average='micro')
    f2 = (5*p*r)/(4*p+r)
    print('f2', f2)
    # return clf

def train_kfold(infoDf):
    kfold = KFold(n_splits=5, shuffle=True, random_state=3407)
    feature = feature_columns()
    cnt = 0
    for train_idx, val_idx in kfold.split(infoDf['label'], infoDf['age']):
        print(len(train_idx), len(val_idx))
        df_train = infoDf.iloc[train_idx]
        df_val = infoDf.iloc[val_idx]
        df_train_x = df_train[feature]
        df_train_y = df_train['label']
        df_val_x = df_val[feature]
        df_val_y = df_val['label']
        dtrain = xgb.DMatrix(df_train_x, label = df_train_y)
        dval = xgb.DMatrix(df_val_x, label = df_val_y)
        saved_path = '/work/reg_stage1_kfold_0506/reg_part_' + str(cnt) + '.json'
        cnt += 1
        trainer(dtrain, dval, saved_path=saved_path)

if __name__ == '__main__':
    infoDf = get_data()
    train_kfold(infoDf)


