import xgboost as xgb
import pandas as pd
from data import GetDataBase
import random
import csv
from tqdm import tqdm

def infer():
    clf = xgb.Booster()
    clf.load_model('baseline.json')
    dtest = GetDataBase(train=False)

    with open('output.csv', 'w') as outputFile:
        writer = csv.DictWriter(outputFile, fieldnames=['cust_wid', 'label'])
        for index, row in tqdm(dtest.iterrows()):
            data = xgb.DMatrix(row.values[1:].reshape(1,-1))
            logit = clf.predict(data, validate_features=False)
            if logit < 0.5:
                logit = 0
            else:
                logit = random.randint(1,14)
            writer.writerow({'cust_wid': row['cust_wid'], 'label': logit})

def inference():
    clf = xgb.Booster()
    clf.load_model('/work/baseline_reg/baseline.json')
    dtest, wid = GetDataBase(train=False)

    dtest = xgb.DMatrix(dtest)
    logits = clf.predict(dtest)
    ans = (logits >= 0.5)*1

    for i in range(len(ans)):
        if ans[i] == 1:
            ans[i] = random.randint(1,14)
    label = pd.DataFrame(ans, columns=['label'])
    result = pd.concat([wid, label], axis=1, ignore_index=False)
    result.to_csv("/work/output.csv", index=False, encoding="latin_1")

if __name__ == '__main__':
    inference()