import xgboost as xgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, log_loss
from datetime import datetime
from scipy import stats
import json


def loadData():
    inputpath = '../data/input/'
    datafile = 'train_lag.npy'
    targetfile = 'train_y.npy'
    print '*'*30
    print 'Reading train file'
    print '-'*30
    X = np.load(inputpath + datafile)
    y = np.load(inputpath + targetfile)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    return X_train, X_test, y_train, y_test


def RandomizedParamsTuning(X_train, X_test, y_train, y_test, num_param=30):
    xgbEst = xgb.XGBClassifier(n_estimators=2000)
    eval_set = [(X_train, y_train), (X_test, y_test)]
    params = {'min_child_weight': stats.randint(1, 20),
              'colsample_bytree': stats.uniform(0.5, 1),
              'max_depth': stats.randint(3, 21),
              'subsample': stats.uniform(0.5, 1),
              'gamma': stats.uniform(0, 10)
              }
    xgParams = {'eval_set': eval_set,
                'eval_metric': 'mlogloss',
                'early_stopping_rounds': 50,
                'verbose': False
                }
    rsModel = RandomizedSearchCV(estimator=xgbEst,
                                 n_iter=num_param,
                                 param_distributions=params,
                                 fit_params=xgParams, verbose=1,
                                 scoring=make_scorer(log_loss))
    rsModel.fit()


def write_log(model):
    logpath = '../log/'
    logfile = 'log_{}.json'.format(datetime.now().strftime("%Y-%m-%d-%H-%M"))
    with open(logpath+logfile, 'w+') as f:
        json.dump(model.cv_results_)
    with open('../log/log_max.txt', 'a+') as f:
        f.write('\n')
        f.write('*'*30 + '\n')
        f.write(datetime.now().strftime("%Y-%m-%d-%H-%M"))
        f.write('\n'+'-'*30+'\n')
        f.write('max_params:\n')
        res = model.best_params_
        for key in res:
            f.write('\t{}: {}\n'.format(key, res[key]))
        f.write('max_val: %f\n' % model.best_score_)
        f.write('-'*30 + '\n')


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = loadData()
    rsModel = RandomizedSearchCV(X_train, X_test, y_train, y_test, 30)
    write_log(rsModel)
