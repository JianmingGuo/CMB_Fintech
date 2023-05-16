import xgboost as xgb
from sklearn.metrics import accuracy_score, auc, roc_curve, precision_score, recall_score
import numpy as np
import pandas as pd
import csv

def feature_columns():
    # names = ['cty_cd_' + str(i) for i in range(32)]
    columns_base = ['age', 'gdr_1', 'gdr_2']
    # names = ['amt_choice_' + str(i) for i in range(20)]
    # names1 = ['trx_cd_' + str(i) for i in range(42)]
    columns_trx = ['trx_len', 'amt_max', 'amt_mean', 'amt_sum', 'amt_min'] 
    names = ['page_id_' + str(i) for i in range(64)]
    columns_view = ['view_len'] + names
    time = ['time_min','time_max','time_mean','time_median','range']
    return columns_base+columns_trx+columns_view + time


def get_data():
    infoDf = pd.read_csv('tasks/8786/preprocess_infer/data_preprocess_infer_testb.csv')
    infoDf = infoDf.sort_values(by='cust_wid')
    wid = infoDf[['cust_wid']]
    # infoDf = infoDf.iloc[:, 1:]
    feature = feature_columns()
    infoDf = infoDf[feature]

    dtest = xgb.DMatrix(infoDf)
    return dtest, wid

def inference():
    dtest, wid = get_data()
    clf = xgb.Booster()
    # clf.load_model('/tasks/7424/reg_stage1_0503/reg.json')
    clf.load_model('/tasks/7424/reg_stage1_0503/reg.json')
    logits = clf.predict(dtest)
    ans = (logits >= 0.4)*1
    return ans

def main():
    ans = inference()
    _, wid = get_data()
    with open('/work/output.csv', "w") as outputFile:
        writer = csv.DictWriter(outputFile, fieldnames=['cust_wid', 'label'])
        writer.writeheader()
        for index, row in wid.iterrows():
            # print(index, row['cust_wid'])
            # if row['cust_wid'] == 'A00001':
            #     writer.writerow({'cust_wid': row['cust_wid'], 'label': 1})
            # else:
            # print(type(int(ans[index])))
            writer.writerow({'cust_wid': row['cust_wid'], 'label': int(ans[index])})
            

if __name__ == '__main__':
    main()