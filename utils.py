import numpy as np
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, recall_score, precision_score
 
def f2_score(act, pred):
    pred = pred > 0.5
    p = precision_score(act, pred)
    r = recall_score(act, pred)
    f2 = (5 * p * r) / (4 * p + r)
    return f2

f2_scorer = make_scorer(f2_score, greater_is_better=True)

def f2_eval(pred, dtrain):
    act = dtrain.get_label()
    pred = (pred > 0) * 1
    p = precision_score(act, pred)
    r = recall_score(act, pred)
    # print(p.shape, p.dtype)
    # print(r.shape, r.dtype)
    f2 = (5 * p * r) / (4 * p + r)
    return "F2-score", 1-f2

def mape_score(ans, label):
    ans = np.argmax(ans, axis=1)+1
    label = label+1
    # index = np.where(label!=0)
    # res = abs(ans[index]-label[index])/(label[index])
    # res = np.sum(res)/len(index)
    res = abs(ans-label)/label
    res = np.sum(res)/len(label)
    return res

mape_scorer = make_scorer(mape_score, greater_is_better=False)

def mape_eval(ans, dtrain):
    label = dtrain.get_label()+1
    ans = np.argmax(ans, axis=1)+1
    # index = np.where(label!=0)
    # print(len(index))
    # print(ans.shape, label.shape)
    res = abs(ans-label)/(label)
    res = np.sum(res)/len(label)
    return "MAPE-score", res