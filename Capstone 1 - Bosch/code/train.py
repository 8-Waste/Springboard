import settings as s


def train_main(reference_model=True, regular_model=True, tune_model=True, final_train_model=True):
    from utils import pu
    pu('Starting training...', int_pad=0)
    if reference_model:
        lgbm_train_reference_main()
    if regular_model:
        scikit_lgbm_train_main()
    if tune_model:
        lgbm_hyperopt_main()
    if final_train_model:
        scikit_lgbm_final_train_main()
    pu('Ending training...', int_pad=1, blank_lines=1)


def lgbm_train_reference_main():
    from utils import pu
    import getdata
    pu('Running reference lgbm model...', int_pad=1)
    pu('Loading training data...', int_pad=2)
    X = getattr(getdata, 'tr_al_o')()  # load all original data
    y = getattr(getdata, 'load_target')()
    pu('Initializing training...', int_pad=2, blank_lines=0)
    best_mcc = lgbm_train_reference(X, y)
    pu('Reference MCC: ' + str(round(best_mcc*100,2)) + '%', int_pad=1)
    pu('*'*60, int_pad=1, blank_lines=1)
    return


def scikit_lgbm_train_main():
    from utils import pu
    import pandas as pd
    import os
    import getdata
    pu('Running lgbm model...', int_pad=1)
    f_list = ['master_final_feats_subset']
    for f_name in f_list:
        if not os.path.isfile(s.DATA_DIR + f_name + '.pkl'):
            raise ValueError(s.DATA_DIR + f_name + '.pkl not found')
        pu('Training on ' + f_name + '...', int_pad=2, blank_lines=0)
        pu('Loading data...', int_pad=3)
        X = getattr(getdata, f_name)()
        y = getattr(getdata, 'load_target')()
        y = y['Response']
        pu('Initializing training ' + str(X.shape) + '...', int_pad=2)
        best_mcc = scikit_lgbm_train(X, y)
        pu('LGBM MCC: ' + str(round(best_mcc*100,2)) + '%', int_pad=1)
        pu('*'*60, int_pad=1, blank_lines=1)
    return


def lgbm_hyperopt_main():  # tuning
    from utils import pu
    import getdata
    pu('Tuning paramaters...', int_pad=1)
    pu('Loading data...', int_pad=2)
    f_name ='master_final_feats_subset'
    X = getattr(getdata, f_name)()
    y = getattr(getdata, 'load_target')()
    y = y['Response']
    pu('Initializing training...', int_pad=2)
    _ = lgbm_hyperopt(X, y)  # saved best to params.pkl
    return


def scikit_lgbm_final_train_main():
    from utils import pu
    import getdata
    pu('Running lgbm final model...', int_pad=1)
    pu('Loading data...', int_pad=2)
    f_name ='master_final_feats_subset'
    X = getattr(getdata, f_name)()
    y = getattr(getdata, 'load_target')()
    y = y['Response']
    pu('Initializing training...', int_pad=2)
    best_mcc = scikit_lgbm_final_train(X, y)
    # pu('LGBM Final MCC: ' + str(round(best_mcc*100,2)) + '%', int_pad=1)
    # pu('*'*60, int_pad=1, blank_lines=1)
    return best_mcc


def lgbm_train_reference(X, y):  # reference model with original data and generic parameters
    import lightgbm as lgb
    import numpy as np
    from utils import mcc_eval, pu, eval_mcc
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    import warnings
    warnings.filterwarnings('ignore', message='Using categorical_feature in Dataset.')
    warnings.filterwarnings('ignore', message='categorical_feature in param dict is overridden.')
    pu('Splitting data:  X_shape:' + str(X.shape) + '   y_shape:' + str(y.shape), int_pad=2)
    # Separate out test data
    X_leftover, X_test, y_leftover, y_test = \
        train_test_split(X, y, test_size=0.1, random_state=50, stratify=y)
    # Separate remainder into training and validation data for optimization
    X_train, X_valid, y_train, y_valid = \
        train_test_split(X_leftover, y_leftover, test_size=0.2, random_state=75, stratify=y_leftover)
    params = {'objective': 'binary',
              'metric': 'None',
              'verbose': -1,
              'boosting': 'gbdt'}
    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    valid_data = lgb.Dataset(X_valid, label=y_valid, free_raw_data=False)
    pu('Start training...', int_pad=2, blank_lines=1)
    evaluation_results = {}
    fit_model = lgb.train(
        params,
        train_set=train_data,
        valid_sets=valid_data,
        evals_result=evaluation_results,
        feval=mcc_eval,
        verbose_eval=-1,
        early_stopping_rounds=50)
    y_prob = fit_model.predict(X_valid, num_iteration=fit_model.best_iteration)
    best_proba, best_mcc, y_pred = eval_mcc(np.array(y_valid), y_prob, show=True)
    cm = confusion_matrix(y_valid, y_pred).ravel()
    msg = f'Reference Results: TN:{cm[0]:,}  FP:{cm[1]:,}  FN:{cm[2]:,}  TP:{cm[3]:,}  Total:{cm[1] + cm[2] + cm[3] + cm[0]:,}'
    pu('*'*60, int_pad=1, blank_lines_before=1)
    pu(msg, int_pad=1, blank_lines_before=0)
    return best_mcc


def scikit_lgbm_train(X, y):
    import numpy as np
    import pandas as pd
    from utils import mcc_eval2, eval_mcc, pu, mcc_scorer2
    from lightgbm import LGBMModel
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
    from sklearn.metrics import confusion_matrix, make_scorer, matthews_corrcoef
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore', message='Using categorical_feature in Dataset.')
    warnings.filterwarnings('ignore', message='categorical_feature in param dict is overridden.')
    warnings.filterwarnings('ignore', message='Data with input dtype int64, float64 were all converted to float64 by MinMaxScaler.')

    pu('Splitting data:  X_shape:' + str(X.shape) + '   y_shape:' + str(y.shape), int_pad=3)
    X_leftover, X_test, y_leftover, y_test = train_test_split(X, y, test_size=0.1, random_state=50, stratify=y)
    # Separate remainder into training and validation data for optimization
    X_train, X_valid, y_train, y_valid = \
        train_test_split(X_leftover, y_leftover, test_size=0.2, random_state=75, stratify=y_leftover)
    pu('Scaling data...', int_pad=3)
    scaler = MinMaxScaler()      # Scale data to 0-1
    cols_to_scale = list(X.select_dtypes(exclude=['category']).columns)
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_valid[cols_to_scale] = scaler.transform(X_valid[cols_to_scale])
    clf = LGBMModel(n_estimators=200,
                    objective='binary',
                    boost_from_average=False,
                    boosting_type='gbdt',
                    learning_rate=.001,    # small for better accuracy
                    num_leaves=31,         # (def=31), large for better accuracy but more overfitting
                    reg_alpha=0.2,         # lambda L1 regularization
                    reg_lambda=0.2,        # lambda L2 regularization
                    max_depth=-1,          # max depth, shallower trees reduce overfitting.
                    max_bin=500,           # maximum number of bins to bucket feature values in
                    colsample_bytree=.90,  # same as feature fraction
                    subsample=0.8,         # from observations
                    subsample_freq=15,
                    random_state=94,
                    metrics='None',
                    verbose=-1,
                    n_jobs=-1)
    pu('Start training...', int_pad=3, blank_lines=1)
    clf.fit(X_train, y_train, categorical_feature='auto',
            eval_set=[(X_valid, y_valid)], eval_metric=mcc_eval2,
            early_stopping_rounds=50, verbose=-1)
    y_prob = clf.predict(X_valid, num_iteration=clf.best_iteration_)
    best_proba, best_mcc, y_pred = eval_mcc(np.array(y_valid), y_prob, show=True)
    cm = confusion_matrix(y_valid, y_pred).ravel()
    msg = f'LGBM Results: TN:{cm[0]:,}  FP:{cm[1]:,}  FN:{cm[2]:,}  TP:{cm[3]:,}  Total:{cm[1] + cm[2] + cm[3] + cm[0]:,}'
    mcc_a = matthews_corrcoef(y_valid, y_pred)
    pu('*'*60, int_pad=1, blank_lines_before=1)
    pu(msg, int_pad=1, blank_lines_before=0)
    feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, X.columns)), columns=['Value', 'Feature'])
    feature_imp.to_csv(s.DATA_DIR + 'feature_importance.csv')
    feature_imp_plot = feature_imp.nlargest(40, 'Value')
    plt.figure(figsize=(20, 10))
    sns.barplot(x="Value", y="Feature", data=feature_imp_plot.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Feature Importance)')
    plt.tight_layout()
    plt.show()
    return best_mcc


def lgbm_hyperopt(X, y):
    import numpy as np
    import warnings
    from hyperopt import hp, tpe, Trials
    from hyperopt.fmin import fmin
    from utils import mcc_scorer2, pu, save_obj
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    from sklearn.metrics import make_scorer
    from sklearn.preprocessing import MinMaxScaler
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pu('Splitting data:  X_shape:' + str(X.shape) + '   y_shape:' + str(y.shape), int_pad=3)
    X_leftover, X_test, y_leftover, y_test = train_test_split(X, y, test_size=0.1, random_state=50, stratify=y)
    # Separate remainder into training and validation data for optimization
    X_train, X_valid, y_train, y_valid = \
        train_test_split(X_leftover, y_leftover, test_size=0.2, random_state=75, stratify=y_leftover)
    pu('Scaling data...', int_pad=3)
    scaler = MinMaxScaler()      # Scale data to 0-1
    cols_to_scale = list(X.select_dtypes(exclude=['category']).columns)
    X_train[cols_to_scale]= scaler.fit_transform(X_train[cols_to_scale])
    # X_valid_scaled = scaler.transform(X_valid)
    pu('Executing hyperopt...', int_pad=3, blank_lines=1)
    my_mcc_scorer = make_scorer(mcc_scorer2, greater_is_better=True)  # if used with LBGMClassificaiton needs_proba=True
    Y_train = np.array(y_train)
    Y_train = Y_train.ravel()
    def objective(params):
        params = {
            'n_estimators': int(params['n_estimators']),
            'learning_rate': params['learning_rate'],
            'num_leaves': int(params['num_leaves']),
            'subsample': round(params['subsample'], 2),
            'colsample_bytree': round(params['colsample_bytree'], 2),
            'reg_alpha': round(params['reg_alpha'], 2),
            'reg_lambda': round(params['reg_lambda'], 2)
            }
        clf = lgb.LGBMModel(
            objective='binary',
            boost_from_average=False,
            boosting_type='gbdt',  # dart for better accuracy (slow) 'gbdt'
            # learning_rate=.001,  # small for better accuracy
            # num_leaves=1000,  # (def=31), large for better accuracy but more overfitting
            # reg_alpha=0,  # lambda L1 regularization
            # reg_lambda=0,  # lambda L2 regularization
            max_depth=-1,  # max depth, shallower trees reduce overfitting.
            max_bin=500,  # maximum number of bins to bucket feature values in
            # subsample=0.65,  # from observations
            subsample_freq=15,
            silent=False,
            metrics='None',
            verbose=-1,
            n_jobs=-1,
            **params)
        score = cross_val_score(clf, X_train, Y_train, scoring=my_mcc_scorer,
                                cv=StratifiedKFold(n_splits=3)).mean()
        print("MCC {:.3f} params {}".format(score, params))
        return 1 - score  # return 1 - MCC need score minimized, corrected in results database before graphing

    space = {
        'n_estimators': hp.quniform('n_estimators', 150, 600, 10),
        'learning_rate': hp.quniform('learning_rate', 0.0001, 0.005, 0.0001),
        'num_leaves': hp.quniform('num_leaves', 5, 600, 5),
        'subsample': hp.quniform('subsample', .7, 1, .05),  # fraction of training rows used in each tree
        'colsample_bytree': hp.quniform('colsample_bytree', .7, 1, .05),
        'reg_alpha': hp.quniform('reg_alpha', 0, .2, .05),
        'reg_lambda': hp.quniform('reg_lambda', 0, .2, .05)
        }
    trials = Trials()  # for logging
    algo = tpe.suggest  # Tree of Parzen Estimators (a "Bayesian" method)
    best = fmin(fn=objective,
                space=space,
                algo=algo,  # optimization algorithm
                trials=trials,  # logging
                max_evals=200)
    save_obj(best, 'params')  # params used in final model
    msg = f'Hyperopt estimated optimum {best}'
    pu('*'*60, int_pad=1, blank_lines_before=1)
    pu(msg, int_pad=1, blank_lines_before=0)
    pu('*' * 60, int_pad=1, blank_lines=1)
    return best


def scikit_lgbm_final_train(X, y):
    import numpy as np
    import sys
    import pandas as pd
    from utils import mcc_eval2, eval_mcc, pu, mcc_scorer2, load_obj
    from lightgbm import LGBMModel
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, cross_val_predict
    from sklearn.metrics import confusion_matrix, make_scorer
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore', message='Using categorical_feature in Dataset.')
    warnings.filterwarnings('ignore', message='categorical_feature in param dict is overridden.')
    warnings.filterwarnings('ignore', message='Data with input dtype int64, float64 were all converted to float64 by MinMaxScaler.')

    pu('Splitting data:  X_shape:' + str(X.shape) + '   y_shape:' + str(y.shape), int_pad=3)
    X_leftover, X_test, y_leftover, y_test = train_test_split(X, y, test_size=0.1, random_state=50, stratify=y)
    # Separate remainder into training and validation data for optimization
    X_train, X_valid, y_train, y_valid = \
        train_test_split(X_leftover, y_leftover, test_size=0.2, random_state=75, stratify=y_leftover)
    my_mcc_scorer = make_scorer(mcc_scorer2, greater_is_better=True)  # if used with LBGMClassificaiton needs_proba=True
    scaler = MinMaxScaler()      # Scale data to 0-1
    cols_to_scale = list(X.select_dtypes(exclude=['category']).columns)
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_valid[cols_to_scale] = scaler.transform(X_valid[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    params = load_obj('params')
    print('')
    print(params)
    print('')
    h = int(params['num_leaves'])  # adjust because of how hyperopt saves the optimization informaiton
    params['num_leaves'] = h
    h = int(params['n_estimators'])
    params['n_estimators'] = h
    clf = LGBMModel(
                    # n_estimators=200,     # populated through **params from hyperopt
                    objective='binary',
                    boost_from_average=False,
                    boosting_type='gbdt',
                    # learning_rate=.001,    # populated through **params from hyperopt
                    # num_leaves=1000,       # populated through **params from hyperopt
                    # reg_alpha=0,           # populated through **params from hyperopt
                    # reg_lambda=0,          # populated through **params from hyperopt
                    max_depth=-1,
                    max_bin=500,
                    # colsample_bytree=.90,  # populated through **params from hyperopt
                    # subsample=0.8,         # populated through **params from hyperopt
                    subsample_freq=15,
                    random_state=94,
                    metrics='None',
                    verbose=-1,
                    n_jobs=-1,
                    **params)
    pu('Start training...', int_pad=3, blank_lines=1)
    clf.fit(X_train, y_train, categorical_feature='auto',
            eval_set=[(X_valid, y_valid)], eval_metric=mcc_eval2,
            early_stopping_rounds=50, verbose=-1)

    y_prob = clf.predict(X_valid, num_iteration=clf.best_iteration_)
    best_proba, best_mcc, y_pred = eval_mcc(np.array(y_valid), y_prob, show=True)
    cm = confusion_matrix(y_valid, y_pred).ravel()
    msg = f'LGBM Final Results: TN:{cm[0]:,}  FP:{cm[1]:,}  FN:{cm[2]:,}  TP:{cm[3]:,}  Total:{cm[1] + cm[2] + cm[3] + cm[0]:,}'
    pu('*' * 60, int_pad=1, blank_lines_before=1)
    pu('LGBM Results below are from validation data', int_pad=1)
    pu('*'*60, int_pad=1)
    pu(msg, int_pad=1, blank_lines_before=0)
    pu('LGBM Final MCC: ' + str(round(best_mcc*100,2)) + '%', int_pad=1)
    pu('*'*60, int_pad=1, blank_lines=1)

    y_prob = clf.predict(X_test, num_iteration=clf.best_iteration_)
    best_proba, best_mcc, y_pred = eval_mcc(np.array(y_test), y_prob, show=True)
    cm = confusion_matrix(y_test, y_pred).ravel()
    msg = f'LGBM Final Results: TN:{cm[0]:,}  FP:{cm[1]:,}  FN:{cm[2]:,}  TP:{cm[3]:,}  Total:{cm[1] + cm[2] + cm[3] + cm[0]:,}'
    pu('*' * 60, int_pad=1, blank_lines_before=1)
    pu('LGBM Results below are from hold-out test data', int_pad=1)
    pu('*'*60, int_pad=1)
    pu(msg, int_pad=1, blank_lines_before=0)
    pu('LGBM Final MCC: ' + str(round(best_mcc*100,2)) + '%', int_pad=1)
    pu('*'*60, int_pad=1, blank_lines=1)
    return best_mcc
