import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import csv
from datetime import datetime


target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
               'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
               'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
               'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
               'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
               'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
               'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
               'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
def readBuffer():
    print 'Reading train file'
    print '-'*30
    X = np.load('../data/input/train_lag.npy')
    y = np.load('../data/input/train_y.npy')
    print 'Reading test file'
    print '-'*30
    X_test = np.load('../data/input/test_lag.npy')
    with open('../data/input/test.csv', 'r') as testfile:
        ids = []
        for row in csv.DictReader(testfile):
            cust_id = row['ncodpers']
            ids.append(cust_id)
    return X, y, X_test, ids


def makeSubmition(filename, y_pred):
    print 'Writing results'
    print '-'*30
    y_pred = np.argsort(y_pred, axis=1)
    y_pred = np.fliplr(y_pred)[:, :8]
    with open(filename, 'w+') as f:
        f.write('added_products,ncodpers\n')
        for pred, idx in zip(y_pred, test_ids):
            line = " ".join(list(np.array(target_cols)[pred]))
            f.write(line)
            f.write(',')
            f.write(str(idx))
            f.write('\n')


X, y, X_test, test_ids = readBuffer()

# NBmodel = GaussianNB()
NBmodel = BernoulliNB(alpha=0.02)
NBmodel.partial_fit(X, y, classes=range(22))
y_pred = NBmodel.predict_proba(X_test)
outputpath = '../data/output/'
output = ('sub_nb_5all_{}.csv'
          .format(datetime.now().strftime("%Y-%m-%d-%H-%M")
                  )
          )
filename = outputpath + output
makeSubmition(filename, y_pred)
