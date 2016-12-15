import xgboost as xgb
import numpy as np
import csv
from bayes_opt import BayesianOptimization
from datetime import datetime

features = ['ind_empleado', 'pais_residencia', 'sexo', 'age', 'fecha_alta',
            'ind_nuevo', 'antiguedad', 'indrel', 'ult_fec_cli_1t',
            'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext', 'conyuemp',
            'canal_entrada', 'indfall', 'tipodom', 'cod_prov',
            'ind_actividad_cliente', 'renta', 'segmento']

target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
               'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
               'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
               'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
               'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
               'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
               'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
               'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']


def getDate(traindate, lag):
    year, month, day = traindate
    if (month+lag) % 12 == 0:
        year += (month+lag)/12 - 1
        month = 12
    else:
        year += (month+lag) / 12
        month = (month+lag) % 12
    return (year, month, day)


def strDate(date):
    year, month, day = date
    if month < 10:
        return '{}-0{}-{}'.format(year, month, day)
    else:
        return '{}-{}-{}'.format(year, month, day)


def getAge(row):
    mean_age = 40.
    min_age = 20.
    max_age = 90.
    range_age = max_age - min_age
    age = int(row['age'])
    if age == -1:
        age = mean_age
    else:
        age = float(age)
        if age < min_age:
            age = min_age
        elif age > max_age:
            age = max_age
    return round((age - min_age) / range_age, 4)


def getCustSeniority(row):
    min_value = 0.
    max_value = 256.
    range_value = max_value - min_value
    missing_value = 0.
    cust_seniority = float(row['antiguedad'])
    if cust_seniority == -1:
        cust_seniority = missing_value
    else:
        if cust_seniority < min_value:
            cust_seniority = min_value
        elif cust_seniority > max_value:
            cust_seniority = max_value
    return round((cust_seniority-min_value) / range_value, 4)


def getRent(row):
    min_value = 0.
    max_value = 1500000.
    range_value = max_value - min_value
    missing_value = 101850.
    rent = float(row['renta'])
    if rent == -1:
        rent = missing_value
    else:
        rent = float(rent)
        if rent < min_value:
            rent = min_value
        elif rent > max_value:
            rent = max_value
    return round((rent-min_value) / range_value, 6)


def getColVal(row, col):
    if col == 'antiguedad':
        return getCustSeniority(row)
    elif col == 'renta':
        return getRent(row)
    elif col == 'age':
        return getAge(row)
    else:
        return int(row[col])


def creatTrainData(filename,
                   lag=-5,
                   traindate=(2015, 6, 28)
                   ):
    lagdate = strDate(getDate(traindate, lag))
    prevdate = strDate(getDate(traindate, -1))
    traindate = strDate(traindate)
    with open(filename, 'r') as trainfile:
        X = []
        y = []
        prev_dict = {}
        lag_dict = {}
        for row in csv.DictReader(trainfile):
            dt = row['fecha_dato']
            cust_id = row['ncodpers']
            if dt not in [traindate, lagdate, prevdate]:
                continue
            target = [getColVal(row, col) for col in target_cols]
            if dt == lagdate:
                lag_dict[cust_id] = target
            elif dt == prevdate:
                prev_dict[cust_id] = target
            elif dt == traindate:
                prev = prev_dict.get(cust_id, [0]*22)
                new_products = [max(x1-x2, 0) for (x1, x2)
                                in zip(target, prev)]
                if sum(new_products) > 0:
                    for ind, prod in enumerate(new_products):
                        if prod > 0:
                            lag = lag_dict.get(cust_id, [0]*22)
                            x_vars = [getColVal(row, col) for col in features]
                            X.append(x_vars+prev+lag)
                            y.append(ind)
    return np.array(X), np.array(y)


def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma):
    num_rounds = 200
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
    cv_result = xgb.cv(params, Xtrain, num_boost_round=num_rounds, nfold=5,
                       seed=random_state, early_stopping_rounds=25)
    return -cv_result['test-mlogloss-mean'].values[-1]


if __name__ == '__main__':
    inputpath = '../data/input/'
    trainfile = 'train.csv'
    testfile = 'test.csv'
    print '*'*30
    target_cols = target_cols[2:]
    print 'Reading train file'
    print '-'*30
    X, y = creatTrainData(inputpath+trainfile)
    Xtrain = xgb.DMatrix(X, label=y)
    num_iter = 100
    init_points = 10
    print 'Optimizing...'
    print '-'*30
    xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 20),
                                                'colsample_bytree': (0.5, 1),
                                                'max_depth': (5, 15),
                                                'subsample': (0.5, 1),
                                                'gamma': (0, 10)
                                                })
    xgbBO.maximize(init_points=init_points, n_iter=num_iter)
    logfile = 'log_{}.csv'.format(datetime.now().strftime("%Y-%m-%d-%H-%M"))
    logpath = '../log/'
    xgbBO.points_to_csv(logpath+logfile)
