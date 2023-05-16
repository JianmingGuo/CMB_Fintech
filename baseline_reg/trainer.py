import xgboost as xgb
from data import GetDataBase
from sklearn.metrics import accuracy_score, auc, roc_curve, precision_score, recall_score
import numpy as np

def trainer(dtrain, dval):
    params = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',  # 多分类的问题          
        'gamma': 0.2,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 5,               # 构建树的深度，越大越容易过拟合
        'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.7,              # 随机采样训练样本
        'colsample_bytree': 0.7,       # 生成树时进行的列采样
        'min_child_weight': 3,
        'silent': 0,                  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.003,                  # 如同学习率
        'seed': 3407,
        'nthread':16,                  # cpu 线程数
        'eval_metric':'logloss',
    }
    saved_path = 'baseline.json'

    watch_list = [(dtrain, 'train'), (dval, 'val')]
    num_round = 30000
    clf = xgb.train(params, dtrain, num_round, watch_list, verbose_eval=100,early_stopping_rounds=20)
    clf.save_model(saved_path)
    return clf

if __name__ == '__main__':
    df_train_x, df_train_y, df_val_x, df_val_y = GetDataBase(train=True)
    print(len(df_val_x), len(df_val_y))
    print(len(df_train_x), len(df_train_y))
    # exit()
    dtrain = xgb.DMatrix(df_train_x, label = df_train_y)
    dval = xgb.DMatrix(df_val_x, label = df_val_y)

    clf = trainer(dtrain, dval)

    ans = clf.predict(dval, ntree_limit=clf.best_ntree_limit)
    ans = (ans >= 0.5)*1
    # np.savetxt('ans.txt',ans)
    fpr, tpr, thersholds = roc_curve(df_val_y.values,ans,pos_label=1)
    roc_auc = auc(fpr,tpr)
    acc = accuracy_score(df_val_y.values, ans,normalize=True)
    print("auc",roc_auc)
    print("acc",acc)
    p = precision_score(df_val_y.values, ans, average='micro')
    r = recall_score(df_val_y.values, ans,average='micro')
    f2 = (5*p*r)/(4*p+r)
    print('f2', f2)