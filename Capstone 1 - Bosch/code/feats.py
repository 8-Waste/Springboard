import settings as s

'''
Puppose of this file:

There are distinct but related part paths (with some possible duplicaiton): 
    The original data
    Data by Line
    Data by Station
    Data by Pod

All engineered features will be merged appropriaetly (e.g. line features with line...)
Due to the volume of features, we will utilize model_selection to retain the predictive features from each category.  
Those predective features will be combined and analyzed with LGBMmodel's feature importance.

feature selection for original, line, station, pod
combine surviving features
feature selection for surviving features as a group to obtain the "final features"
'''


def feats_selection_main():
    import os
    import getdata
    import pandas as pd
    from utils import pu
    pu('Starting feature selection...', int_pad=0, blank_lines=0)
    f_list = ['o_final_feats_all', 'l_final_feats_all', 's_final_feats_all', 'p_final_feats_all']
    for f_name in f_list:
        if not os.path.isfile(s.DATA_DIR + f_name[:-3] + 'subset.pkl'):
            feats_selection(f_name)
        else:
            pu('Located ' + f_name[:-3] + 'subset.pkl', int_pad=1)

    df_1 = pd.DataFrame()  # empty dataframe to make the first concat easy
    f_name = 'master_final_feats_all'
    if not os.path.isfile(s.DATA_DIR + f_name + '.pkl'):
        pu('Creating ' + f_name + '.pkl', int_pad=2)
        for f_name in f_list:
            pu('Loading and combining ' + f_name + '.pkl', int_pad=3)
            df = getattr(getdata, f_name[:-3] + 'subset')()
            cols_to_use = df.columns.difference(df_1.columns)  # keeps out duplicate columns
            df_1 = pd.concat([df[cols_to_use], df_1], axis=1, sort=True)
        pu('Saving master_final_feats_all.pkl', int_pad=2)
        df_1.to_pickle(s.DATA_DIR + 'master_final_feats_all.pkl')
    else:
        pu('Located ' + f_name + '.pkl', int_pad=1)
    f_name = 'master_final_feats_all'
    if not os.path.isfile(s.DATA_DIR + f_name[:-3] + 'subset.pkl'):
        final_master_feat_selection(f_name)
    else:
        pu('Located ' + f_name[:-3] + 'subset.pkl', int_pad=1)
    pu('Completed feature selection...', int_pad=0, blank_lines=1)


def feats_selection(df_name):
    import getdata
    from utils import pu
    from lightgbm import LGBMModel
    from sklearn.feature_selection import SelectFromModel
    pu('Performing feature reduction on ' + df_name + ' ...', int_pad=1)
    X = getattr(getdata, df_name)()
    X = X.fillna(0)
    y = getattr(getdata, 'load_target')()
    pu('Checking ' + str(len(X.columns)) + ' features ' + str(X.shape) + '...', int_pad=2)
    clf = LGBMModel(n_estimators=400,
                    objective='binary',
                    boost_from_average=False,
                    boosting_type='gbdt',
                    learning_rate=.001,  # small for better accuracy
                    num_leaves=400,  # (def=31), large for better accuracy but more overfitting
                    reg_alpha=0,  # lambda L1 regularization
                    reg_lambda=0,  # lambda L2 regularization
                    max_depth=-1,  # max depth, shallower trees reduce overfitting.
                    max_bin=1000,  # maximum number of bins to bucket feature values in
                    colsample_bytree=.75,
                    subsample=0.75,  # from observations
                    subsample_freq=15,
                    silent=True,
                    metrics='None',
                    n_jobs=-1)
    select_model = SelectFromModel(clf)
    # fit on train set
    fit = select_model.fit(X, y)
    feats_to_keep = X.columns[select_model.get_support()].tolist()
    pu('Keeping ' + str(len(feats_to_keep)) + ' features from ' + df_name + '...', int_pad=2)
    X = X[feats_to_keep]
    X.to_pickle(s.DATA_DIR + df_name[:-3] + 'subset.pkl')
    pu('Completed feature reduction on ' + df_name + ' ...', int_pad=1, blank_lines=1)


def final_master_feat_selection(df_name):
    import pandas as pd
    import getdata
    import os
    from utils import mcc_eval2, pu, update_categorical
    from lightgbm import LGBMModel
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    warnings.filterwarnings('ignore', message='Using categorical_feature in Dataset.')
    warnings.filterwarnings('ignore', message='categorical_feature in param dict is overridden.')
    warnings.filterwarnings('ignore', message='Data with input dtype int64, float64 were all converted to float64 by MinMaxScaler.')

    pu('Beginning final feature selection...', int_pad=1)
    if not os.path.isfile(s.DATA_DIR + df_name + '.pkl'):
        raise ValueError(s.DATA_DIR + df_name + '.pkl not found')
    pu('Performing feature reduction on ' + df_name + '...', int_pad=2, blank_lines=0)
    pu('Loading data...', int_pad=3)
    X = getattr(getdata, df_name)()
    y = getattr(getdata, 'load_target')()
    y = y['Response']
    X = update_categorical(X)  # mark categorical features

    pu('Starting selection process...', int_pad=2)
    for i in range(20):
        pu('Splitting data:  X_shape:' + str(X.shape) + '   y_shape:' + str(y.shape), int_pad=3)
        X_leftover, X_test, y_leftover, y_test = train_test_split(X, y, test_size=0.1, random_state=50, stratify=y)
        # Separate remainder into training and validation data for optimization
        X_train, X_valid, y_train, y_valid = \
            train_test_split(X_leftover, y_leftover, test_size=0.2, random_state=75, stratify=y_leftover)
        pu('Modeling data...', int_pad=3, blank_lines=1)
        scaler = MinMaxScaler()      # Scale data to 0-1
        cols_to_scale = list(X.select_dtypes(exclude=['category']).columns)  # scale all but categorical
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
        clf.fit(X_train, y_train, categorical_feature='auto',
                eval_set=[(X_valid, y_valid)], eval_metric=mcc_eval2,
                early_stopping_rounds=50, verbose=-1)
        feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_, X.columns)), columns=['Value', 'Feature'])
        keep = feature_imp[feature_imp['Value'] > 0]
        keep = keep['Feature'].tolist()
        X = X[keep]
        pu('Keeping ' + str(len(keep)) + ' features ', int_pad=3, blank_lines_before=1)
        # plotted for entertainment value... while running
        feature_imp_plot = feature_imp.nlargest(40, 'Value')
        plt.figure(figsize=(20, 10))
        sns.barplot(x="Value", y="Feature", data=feature_imp_plot.sort_values(by="Value", ascending=False))
        plt.title('LightGBM Feature Importance)')
        plt.tight_layout()
        plt.show()

    keep = feature_imp.nlargest(18, 'Value')  # keep the 18 most predictive features (determined by trial and error)
    keep = keep['Feature'].tolist()
    X = X[keep]
    X.to_pickle(s.DATA_DIR + df_name[:-3] + 'subset.pkl')
    pu('Completed feature reduction on ' + df_name + 'with ' +
       str(X.shape[0]) + ' features..', int_pad=1, blank_lines=1)
    return
