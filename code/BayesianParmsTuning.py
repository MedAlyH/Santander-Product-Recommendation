import xgboost as xgb
import numpy as np
from bayes_opt import BayesianOptimization
from datetime import datetime


def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma,
                 eta,
                 num_rounds):
    random_state = 123
    params = {'min_child_weight': int(min_child_weight),
              'cosample_bytree': max(min(colsample_bytree, 1), 0),
              'max_depth': int(max_depth),
              'subsample': max(min(subsample, 1), 0),
              'gamma': max(gamma, 0),
              'eta': 0.05,
              'silent': 1,
              'num_class': 22,
              'eval_metric': 'mlogloss',
              'verbose_eval': True,
              'seed': random_state
              }
    cv_result = xgb.cv(params, Xtrain, num_boost_round=int(num_rounds),
                       nfold=5, seed=random_state
                       )
    return -cv_result['test-mlogloss-mean'].values[-1]


def write_log(xgbBO):
    logfile = 'log_{}.csv'.format(datetime.now().strftime("%Y-%m-%d-%H-%M"))
    logpath = '../log/'
    xgbBO.points_to_csv(logpath+logfile)
    res = xgbBO.res['max']
    with open('../log/log_max.txt', 'a+') as f:
        f.write('\n')
        f.write('*'*30 + '\n')
        f.write(datetime.now().strftime("%Y-%m-%d-%H-%M"))
        f.write('\n'+'-'*30+'\n')
        f.write('max_params:\n')
        for key in res['max_params']:
            f.write('\t{}: {}\n'.format(key, res['max_params'][key]))
        f.write('max_val: %f\n' % res['max_val'])
        f.write('-'*30 + '\n')


if __name__ == '__main__':
    inputpath = '../data/input/'
    trainfile = 'train.csv'
    testfile = 'test.csv'
    print '*'*30
    print 'Reading train file'
    print '-'*30
    X = np.load('../data/input/train_lag.npy')
    y = np.load('../data/input/train_y.npy')
    Xtrain = xgb.DMatrix(X, label=y)
    num_iter = 30
    init_points = 5
    print 'Optimizing...'
    print '-'*30
    params = {'min_child_weight': (1, 20),
              'colsample_bytree': (0.5, 1),
              'max_depth': (3, 10),
              'subsample': (0.5, 1),
              'gamma': (0, 5),
              'eta': (0, 0.3),
              'num_rounds': (50, 500)
              }
    # xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 20),
    #                                             'colsample_bytree': (0.5, 1),
    #                                             'max_depth': (5, 15),
    #                                             'subsample': (0.5, 1),
    #                                             'gamma': (0, 10)
    #                                             })
    xgbBO = BayesianOptimization(xgb_evaluate, params)
    xgbBO.maximize(init_points=init_points, n_iter=num_iter)
    write_log(xgbBO)
