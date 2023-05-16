import xgboost as xgb
import pandas as pd
from sklearn.metrics import accuracy_score, auc, roc_curve, precision_score, recall_score
import numpy as np
import csv

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

def get_data():
    infoDf = pd.read_csv('/tasks/10679/process_0505/testb_0505_onehot.csv')
    # infoDf = infoDf.sort_values(by='cust_wid')
    wid = infoDf['cust_wid']
    # infoDf = infoDf.iloc[:, 1:]
    feature = feature_columns()
    infoDf = infoDf[feature]

    dtest = xgb.DMatrix(infoDf)
    return dtest, wid

def infer(json, dval):
    clf = xgb.Booster()
    clf.load_model(json)
    logit = clf.predict(dval, ntree_limit=clf.best_ntree_limit)
    # ans = (ans >= 0.5)*1
    return logit


def main():
    json_files = [
        '/work/reg_stage1_kfold_0503/reg_part_0.json',
        '/work/reg_stage1_kfold_0503/reg_part_1.json',
        '/work/reg_stage1_kfold_0503/reg_part_2.json',
        '/work/reg_stage1_kfold_0503/reg_part_3.json',
        '/work/reg_stage1_kfold_0503/reg_part_4.json',
    ]
    dtest, wid = get_data()
    logits_all = []
    for json in json_files:
        logit = infer(json, dtest)
        logits_all.append(logit)
    ans = np.max(logits_all, axis=0)
    np.save('logits_testb.csv', ans)
    ans = (ans >= 0.5)*1
    return ans


def save():
    ans = main()
    with open('/work/output.csv', "w") as outputFile, open('/data/testa_base.csv') as inputFile:
        writer,reader = csv.DictWriter(outputFile, fieldnames=['cust_wid', 'label']),  csv.DictReader(inputFile)
        writer.writeheader()
        for index, row in enumerate(reader):
            # print(index, row['cust_wid'])
            # if row['cust_wid'] == 'A00001':
            #     writer.writerow({'cust_wid': row['cust_wid'], 'label': 1})
            # else:
            # print(type(int(ans[index])))
            writer.writerow({'cust_wid': row['cust_wid'], 'label': int(ans[index])})


if __name__ == '__main__':
    save()