import numpy as np
from datetime import datetime
import csv
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from modelparams import get_paramslist
import json

features = ['ind_empleado', 'pais_residencia', 'sexo', 'age', 'fecha_alta',
            'ind_nuevo', 'antiguedad', 'indrel', 'ult_fec_cli_1t',
            'indrel_1mes', 'tiprel_1mes', 'indresi', 'indext', 'conyuemp',
            'canal_entrada', 'indfall', 'tipodom', 'cod_prov',
            'ind_actividad_cliente', 'renta', 'segmento', 'est_month']

lag_fea = ['age', 'ind_nuevo', 'antiguedad', 'indrel', 'indrel_1mes',
           'tiprel_1mes', 'ind_actividad_cliente', 'segmento', 'fecha_alta']

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


def getMonthDiff(date1, date2):
    yr1, m1, _ = date1
    yr2, m2, _ = date2
    return (yr2 - yr1) * 12 + (m2 - m1)


def getAge(row):
    mean_age = 40
    age = int(row['age'])
    if age == -1:
        age = mean_age
    return age


def getCustSeniority(row):
    missing_value = 0.
    cust_seniority = float(row['antiguedad'])
    if cust_seniority == -1 or cust_seniority == -999999:
        cust_seniority = missing_value
    return cust_seniority


def getRent(row):
    renta_dict = {'-1': 134254, '1': 111098, '2':  83064, '3':  87357,
                  '4':  85400, '5':  76853, '6':  72179, '7': 171996,
                  '8': 164679, '9':  97881, '10':  75365, '11':  98639,
                  '12':  79182, '13':  69888, '14':  85610, '15': 112801,
                  '16':  69963, '17': 144280, '18':  96550, '19':  95555,
                  '20': 141895, '21':  76679, '22':  89228, '23':  77142,
                  '24':  93382, '25':  81211, '26':  99658, '27':  76686,
                  '28': 178898, '29': 121216, '30':  79084, '31': 102476,
                  '32':  83307, '33': 101403, '34':  92760, '35': 100331,
                  '36': 113531, '37': 105810, '38': 102752, '39': 121201,
                  '40':  98463, '41': 117427, '42':  88064, '43': 104581,
                  '44':  87655, '45':  80624, '46':  89799, '47': 101653,
                  '48': 110388, '49':  83322, '50': 110587, '51': 199074,
                  '52': 149910
                  }
    rent = float(row['renta'])
    if rent == -1:
        rent = float(renta_dict[row['cod_prov']])
    return np.log(rent)


def getColVal(row, col):
    if col == 'antiguedad':
        return getCustSeniority(row)
    elif col == 'renta':
        return getRent(row)
    elif col == 'age':
        return getAge(row)
    else:
        return int(row[col])


def read_train_file(filename):
    product_dict = {}
    features_dict = {}
    with open(filename, 'r') as trainfile:
        for row in csv.DictReader(trainfile):
            dt = row['fecha_dato']
            cust_id = row['ncodpers']
            if dt not in product_dict:
                product_dict[dt] = {}
            if dt not in features_dict:
                features_dict[dt] = {}
            target = [getColVal(row, col) for col in target_cols]
            fea = [getColVal(row, col) for col in features]
            product_dict[dt][cust_id] = target
            features_dict[dt][cust_id] = fea
    return product_dict, features_dict


def getMonthData(month, product_dict, features_dict, lag=5):
    lagdates = [strDate(getDate(month, -d)) for d in range(1, lag+1)]
    prevdate = strDate(getDate(month, -1))
    month = strDate(month)
    X, y = [], []
    empty_tar = [0 for i in range(N)]
    for cust_id in product_dict[month]:
        prev = product_dict[prevdate].get(cust_id, empty_tar)
        target = product_dict[month][cust_id]
        new_products = [max(x1-x2, 0) for (x1, x2) in zip(target, prev)]
        if sum(new_products) > 0:
            for ind, prod in enumerate(new_products):
                if prod > 0:
                    x_vars = features_dict[month][cust_id]
                    tars = []
                    for dt in lagdates:
                        tar_lag = product_dict[dt].get(cust_id, empty_tar)
                        tars += tar_lag
                    X.append(x_vars + tars)
                    y.append(ind)
    return X, y


def creatTrainData(filename, lag=5,
                   traindate=(2016, 5, 28),
                   startdate=(2015, 6, 28),
                   savefile=True, loadfile=False):
    print 'Reading train file'
    print '-'*30
    if loadfile:
        with open('../data/input/feaDict.json') as outfile:
            feaDict = json.load(outfile)
        with open('../data/input/prodDict.json') as outfile:
            prodDict = json.load(outfile)
    else:
        prodDict, feaDict = read_train_file(filename)
    month_diff = getMonthDiff(startdate, traindate)
    train_monthes = [getDate(traindate, -i) for i in range(month_diff+1)]
    X, y = [], []
    for month in train_monthes:
        print 'Adding', strDate(month)
        X_month, y_month = getMonthData(month, prodDict,
                                        feaDict, lag=lag)
        X += X_month
        y += y_month
    print '-'*30
    if savefile:
        print 'dumping json'
        print '-'*30
        with open('../data/input/feaDict.json', 'w') as outfile:
            json.dump(feaDict, outfile)
        with open('../data/input/prodDict.json', 'w') as outfile:
            json.dump(prodDict, outfile)
    del feaDict
    return np.array(X), np.array(y), prodDict


def creatTestData(filename, lag_dict,
                  testdate=(2016, 6, 28), lag=5):
    print 'Reading test file'
    print '-'*30
    lagdates = []
    empty_tar = [0 for i in range(N)]
    for day in range(1, lag+1):
        lagdates.append(strDate(getDate(testdate, -day)))
    with open(filename, 'r') as testfile:
        X = []
        ids = []
        for row in csv.DictReader(testfile):
            cust_id = row['ncodpers']
            x_vars = [getColVal(row, col) for col in features]
            for dt in lagdates:
                tar_lag = lag_dict[dt].get(cust_id, empty_tar)
                x_vars += tar_lag
            X.append(x_vars)
            ids.append(cust_id)
    return np.array(X), ids


def SaveBuffer(X_train, y_train, X_test):
    print 'Saving numpy array'
    print '-'*30
    np.save('../data/input/train_lag.npy', X)
    np.save('../data/input/train_y.npy', y)
    np.save('../data/input/test_lag.npy', X_test)


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


def runXGB(X_train, y_train, params, num_rounds):
    print 'Training'
    print '-'*30
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(params, xgtrain, num_rounds)
    print log_loss(y_train, model.predict(xgtrain))
    return model


def predictProduct(model, X_test):
    print 'Predicting'
    print '-'*30
    Xtest = xgb.DMatrix(X_test)
    y_pred = model.predict(Xtest)
    return y_pred


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


def one_run(X_train, y_train, X_test, params, num_rounds):
    model = runXGB(X_train, y_train, params, num_rounds)
    y_pred = predictProduct(model, X_test)
    outputpath = '../data/output/'
    output = ('sub_xgb_5all_{}.csv'
              .format(datetime.now().strftime("%Y-%m-%d-%H-%M"))
              )
    filename = outputpath + output
    makeSubmition(filename, y_pred)
    return model, y_pred


def cv_run(X_train, y_train, X_test, nfolds, params, num_rounds):
    kf = KFold(n_splits=nfolds)
    num = X_test.shape[0]
    y_preds = np.zeros((num, N))
    models = []
    for i, index in enumerate(kf.split(X_train)):
        print '{} / {} folds:'.format(i+1, nfolds)
        print '-'*30
        idx = index[0]
        model = runXGB(X_train[idx], y_train[idx], params, num_rounds)
        y_pred = predictProduct(model, X_test)
        y_preds += y_pred
        models.append(model)
    y_preds /= nfolds
    outputpath = '../data/output/'
    output = ('sub_xgb_{}fold_5all_{}.csv'
              .format(nfolds,
                      datetime.now().strftime("%Y-%m-%d-%H-%M")
                      )
              )
    filename = outputpath + output
    makeSubmition(filename, y_preds)
    return models, y_preds


def add_knn_feature(X_train, y_train, X_test):
    print 'Getting knn features'
    print '-'*30
    knn = KNeighborsClassifier(n_jobs=-1)
    knn.fit(X_train, y)
    print 'for train'
    knn_train = knn.predict_proba(X_train)
    print 'for test'
    knn_test = knn.predict_proba(X_test)
    X_train = np.hstack((X_train, knn_train))
    X_test = np.hstack((X_test, knn_test))
    return X_train, X_test


def model_ensemble(X_train, y_train, X_test):
    params_list, num_rounds_list = get_paramslist(N)
    num = X_test.shape[0]
    y_preds = np.zeros((num, N))
    nfolds = len(params_list)
    for params, num_rounds in zip(params_list, num_rounds_list):
        print '-'*30
        print params
        model = runXGB(X_train, y_train, params, num_rounds)
        y_pred = predictProduct(model, X_test)
        y_preds += y_pred
    y_preds /= nfolds
    outputpath = '../data/output/'
    output = ('sub_xgb_{}ensembles_5all_{}.csv'
              .format(nfolds,
                      datetime.now().strftime("%Y-%m-%d-%H-%M")
                      )
              )
    filename = outputpath + output
    makeSubmition(filename, y_preds)
    return y_preds


if __name__ == '__main__':
    inputpath = '../data/input/'
    trainfile = 'train.csv'
    testfile = 'test.csv'
    print '*'*30
    target_cols = target_cols[2:]
    N = len(target_cols)
    M = len(lag_fea)
    X, y, product_dict = creatTrainData(inputpath+trainfile,
                                        traindate=(2015, 12, 28))
    X_test, test_ids = creatTestData(inputpath+testfile, product_dict)
    del product_dict
    # X, X_test = add_knn_feature(X, y, X_test)
    SaveBuffer(X, y, X_test)
    # X, y, X_test, test_ids = readBuffer()
    print X.shape, y.shape, X_test.shape
    params = {'objective': 'multi:softprob',
              'num_class': N,
              'colsample_bytree': 0.810195135669,
              'gamma': 1.6446531418,
              'max_depth': 4,
              'min_child_weight': 2,
              'subsample': 1.0,
              'seed': 123,
              'eta': 0.205950347095,
              'silent': 1,
              'eval_metric': "mlogloss",
              }
    num_rounds = 373
    model, y_pred = one_run(X, y, X_test, params, num_rounds)
    # nfolds = 5
    # models, y_preds = cv_run(X, y, X_test, nfolds, params, num_rounds)
    y_preds = model_ensemble(X, y, X_test)
    print 'Done!'
