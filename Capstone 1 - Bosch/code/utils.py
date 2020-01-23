import settings as s
from numba import jit
import numpy as np


def train_test_separation(df, check_rows=True):  # requires Id as index
    import getdata
    train_ids = getattr(getdata, 'load_target')()
    train_ids_list = train_ids.index.to_list()
    df_1 = df[~df.index.isin(train_ids_list)]
    df = df[df.index.isin(train_ids_list)]
    if df.shape[0] != 1183747 and check_rows:
        raise ValueError('Training rows not equal to 1183747 but is equal to ' + str(df.shape[0]))
    if df_1.shape[0] != 1183748 and check_rows:
        raise ValueError('Testing rows not eqaul to 1183748 but is equal to ' + str(df.shape[0]))
    return df, df_1


@jit
def mcc(tp, tn, fp, fn):
    sup = tp * tn - fp * fn
    inf = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if inf==0:
        return 0
    else:
        return sup / np.sqrt(inf)


@jit
def eval_mcc(y_true, y_prob, show=False):
    # import numpy as np
    # from sklearn.metrics import matthews_corrcoef
    # import matplotlib.pyplot as plt
    y_true = np.array(y_true)
    idx = np.argsort(y_prob)
    y_true_sort = y_true[idx]
    n = y_true.shape[0]
    nump = 1.0 * np.sum(y_true)  # number of positive
    numn = n - nump  # number of negative
    tp = nump
    tn = 0.0
    fp = numn
    fn = 0.0
    best_mcc = 0.0
    prev_proba = -1
    best_proba = -1
    new_mcc = -1
    mccs = np.zeros(n)
    for i in range(n):
        # all items with idx < i are predicted negative while others are predicted positive
        # only evaluate mcc when probability changes
        proba = y_prob[idx[i]]
        if proba != prev_proba:
            prev_proba = proba
            new_mcc = mcc(tp, tn, fp, fn)
            if new_mcc >= best_mcc:
                best_mcc = new_mcc
                best_proba = proba
        mccs[i] = new_mcc
        if y_true_sort[i] == 1:
            tp -= 1.0
            fn += 1.0
        else:
            fp -= 1.0
            tn += 1.0
    if show:
        y_pred = (y_prob >= best_proba).astype(int)
        # score = matthews_corrcoef(y_true, y_pred)  # for confirming caluclations
        # print(score, best_mcc)
        # plt.plot(mccs)
        # plt.show()
        return best_proba, best_mcc, y_pred
    else:
        return best_mcc


def mcc_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    best_mcc = eval_mcc(y_true, y_pred)
    return 'MCC', best_mcc, True  # True means Greater is good, False means Greater is bad


def mcc_eval2(y_true, y_pred):
    best_mcc = eval_mcc(y_true, y_pred)
    return 'MCC', best_mcc, True  # True means Greater is good, False means Greater is bad


def mcc_scorer2(y_true, y_pred):
    mcc_score = eval_mcc(y_true, y_pred)
    return mcc_score


def get_bar_plot(data, title, suptitle, x_column, x_label, y_column, y_label, y_tick_labels=1,
                 hue=None, palette=s.MY_COLORS, figure_size=(15, 5)):
    import pandas as pd
    from matplotlib.pyplot import figure
    import matplotlib.pyplot as plt
    import seaborn as sns
    df_data = pd.read_hdf(data, mode='r')
    figure(figsize=figure_size)
    title_addition = '  **' + data + '**'
    ax = sns.barplot(x=x_column, y=y_column, data=df_data, hue=hue, palette=palette)
    vals = ax.get_yticks()
    plt.suptitle(suptitle)
    plt.title(title + title_addition)
    ax.set_xlabel(x_label, fontsize='large')
    ax.set_ylabel(y_label, fontsize='large')
    if y_tick_labels == 1:
        ax.set_yticklabels(['{:,.0f}'.format(x) for x in vals])
    else:
        ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])
    plt.show()


def pu(msg, int_pad=0, blank_lines=0, blank_lines_before=0):
    from datetime import datetime as dt
    cpu_stats = ''
    if s.COMPUTER_STATS:
        import psutil
        cpu_stats = ' (CPU' + str(psutil.cpu_percent()).rjust(5) + '%)(RAM' + str(
            dict(psutil.virtual_memory()._asdict())['percent']).rjust(5) + '%) '
    msg = cpu_stats + dt.now().strftime('%H:%M:%S') + ' ' + ' ' * int_pad * 2 + msg
    if blank_lines_before > 0:
        for l in range(blank_lines_before):
            print()
    print(msg)
    with open(s.LOG_FILE, 'a') as logfile:
        logfile.write(msg + '\n')
    if blank_lines > 0:
        for l in range(blank_lines):
            print()
    return


def reset_luigi_status():
    import os
    import glob
    files = glob.glob('luigi_status/*')
    for f in files:
        os.remove(f)


def save_obj(obj, name):
    import pickle
    with open(s.DATA_DIR + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name ):
    import pickle
    with open(s.DATA_DIR + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def update_categorical(df):
    cat_col_list = ['p_first_location', 'w_starting_batch', 'x_route', '_month', '_wday', '_fnsta']

    for cat_col in cat_col_list:
        chg_to_cat = [col for col in df.columns.values.tolist() if col.endswith(cat_col)]
        if len(chg_to_cat) > 0:
            df[chg_to_cat] = df[chg_to_cat].astype('category')
    return df


