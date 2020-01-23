import numpy as np
from numba import njit
import settings as s


'''
general file name convention:
  xx_yy_z
    xx = tr, te, al   (train, test, both/all)
    yy = dt, nu, al  (date, numeric, both/all)
    z = o, l, s, p   (original data, line data, station data, pod data)
    
    tr_nu_p   train data for numeric features for pods 
'''


def eda():  # all data prep and feature engineering starts and ends here
    from utils import pu

    pu('Creating features...', int_pad=0)

    pu('Creating train numeric features...', int_pad=1)
    nu_eda('tr_nu_o')

    pu('Creating test numeric features...', int_pad=1)
    nu_eda('te_nu_o')

    pu('Creating train date features...', int_pad=1)
    dt_eda('tr_dt_o')

    pu('Creating test date features...', int_pad=1)
    dt_eda('te_dt_o')

    pu('Combining files...', int_pad=1)
    comb_files(['tr_dt_l', 'te_dt_l'], 'al_dt_l', caxis=0)
    comb_files(['tr_dt_s', 'te_dt_s'], 'al_dt_s', caxis=0)
    comb_files(['tr_dt_p', 'te_dt_p'], 'al_dt_p', caxis=0)

    pu('Creating part distribution...', int_pad=1)
    part_dist()

    pu('Creating part routes...', int_pad=1)
    create_part_routes()

    pu('Creating que and wip...', int_pad=1)
    que_and_wip('EF__L_', 'Line', 'al_dt_l')
    que_and_wip('EF__S_', 'Station', 'al_dt_s')
    que_and_wip('EF__P_', 'Pod', 'al_dt_p')

    pu('Populating que and wip...', int_pad=1)
    populate_wip('Line', 'tr_dt_l', 'l_wip')
    populate_wip('Station', 'tr_dt_s', 's_wip')
    populate_wip('Pod', 'tr_dt_p', 'p_wip')

    pu('Creating training source files...', int_pad=1)
    comb_files(['tr_al_o', 'tr_dt_rec_ord'], 'o_final_feats_all', caxis=1)
    comb_files(['tr_dt_l', 'tr_nu_l', 'tr_dt_rec_ord', 'tr_route_l', 'l_wip_by_id'], 'l_final_feats_all', caxis=1)
    comb_files(['tr_al_s', 'tr_dt_rec_ord', 'tr_route_s', 's_wip_by_id'], 's_final_feats_all', caxis=1)
    comb_files(['tr_dt_p', 'tr_nu_p', 'tr_dt_rec_ord', 'tr_route_p', 'p_wip_by_id'], 'p_final_feats_all', caxis=1)

    pu('Feature creation complete', blank_lines=1)


def nu_eda(f_name):  # numeric data prep and feature engineering
    import os
    from utils import pu
    import getdata

    lst = ['Line', 'Station', 'Pod']
    loc_name_list = []
    for loc_name in lst:
        if os.path.isfile(s.DATA_DIR + f_name[:-1] + loc_name[0].lower() + '.pkl'):
            pu('Located ' + f_name[:-1] + loc_name[0].lower(), int_pad=2)
        else:
            loc_name_list.append(loc_name)
    if len(loc_name_list) != 0:
        pu('Loading ' + f_name + '...', int_pad=2)
        df = getattr(getdata, f_name)()
        src = 'num'
        pu('Processing ' + f_name + '...', int_pad=2)
        split_l_s_p(df, src, f_name, loc_name_list)


def dt_eda(f_name):  # date prep and feature engineering
    import os
    from utils import pu
    import getdata

    lst = ['Line', 'Station', 'Pod']
    loc_name_list = []
    for loc_name in lst:
        if os.path.isfile(s.DATA_DIR + f_name[:-1] + loc_name[0].lower() + '.pkl'):
            pu('Located ' + f_name[:-1] + loc_name[0].lower(), int_pad=2)
        else:
            loc_name_list.append(loc_name)
    if len(loc_name_list) != 0:
        pu('Loading ' + f_name + '...', int_pad=2)
        df = getattr(getdata, f_name)()
        src = 'date'
        pu('Processing ' + f_name + '...', int_pad=2)
        split_l_s_p(df, src, f_name, loc_name_list)
    return


def split_l_s_p(df, src, f_name, loc_name_list):  # create and populate line, station and pod columns
    import os
    import pandas as pd
    from utils import pu
    import warnings
    warnings.filterwarnings('ignore', message='All-NaN slice encountered')
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice')
    df_cols = pd.DataFrame(df.columns)
    df_cols.columns = ['L_S_D']
    df_cols['Line'], df_cols['Station'], df_cols['Feature'] = df_cols['L_S_D'].str[1:12].str.split('_\w').str
    df_cols[['Line_int', 'Station_int', 'Feature_int']] = df_cols[['Line', 'Station', 'Feature']].astype('int')
    df_cols.reset_index(inplace=True)
    station_to_pod = {
        0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4, 8: 5, 9: 6, 10: 6, 11: 6, 12: 0, 13: 1, 14: 2,
        15: 2, 16: 3, 17: 3, 18: 4, 19: 4, 20: 5, 21: 6, 22: 6, 23: 6, 24: 7, 25: 7, 26: 8, 27: 8,
        28: 8, 29: 9, 30: 10, 31: 11, 32: 12, 33: 13, 34: 14, 35: 14, 36: 15, 37: 16, 38: 17, 39: 18,
        40: 19, 41: 20, 42: 98, 43: 21, 44: 21, 45: 22, 46: 98, 47: 23, 48: 24, 49: 25, 50: 25, 51: 26
        }
    df_cols['Pod'] = df_cols['Station_int'].map(station_to_pod).astype(np.int32)
    # loc_name_list = ['Line', 'Station', 'Pod']
    for loc_name in loc_name_list:
        if os.path.isfile(s.DATA_DIR + f_name[:-1] + loc_name[0].lower()+'.pkl'):
            pu('Located ' + f_name[:-1] + loc_name[0].lower(), int_pad=3)
            continue
        pu('Processing ' + loc_name.lower() + '...', int_pad=3)
        loc_prefix = loc_name[0] + '_'
        unique_loc_nbr = df_cols[loc_name].unique()
        for cntr, loc_nbr in enumerate(unique_loc_nbr):  # line, station or pod id (e.g. Line 0,1,2,3)
            # need to enumerate above and update message below so don't have 52 of 50 due to empty stations
            pu(str(int(cntr)+1).zfill(2) + ' of ' + str(len(unique_loc_nbr)).zfill(2), int_pad=5)
            col_idx_array = df_cols['index'][df_cols[loc_name] == loc_nbr].to_numpy()  # index is the col pos in df
            new_col = 'EF__' + loc_prefix + str(int(loc_nbr)+1).zfill(2)  # need to start at one becuase of fillna(0)
            if src == 'num':
                df[new_col+'num_min'] = np.nanmin(df.iloc[:, col_idx_array].values, axis=1)  # speed improvement
                df[new_col + 'num_max'] = np.nanmax(df.iloc[:, col_idx_array].values, axis=1)
                df[new_col+'num_avg'] = np.nanmean(df.iloc[:, col_idx_array].values, axis=1)
                df[new_col + 'num_median'] = np.nanmedian(df.iloc[:, col_idx_array].values, axis=1)
                df[new_col+'num_std'] = np.nanstd(df.iloc[:, col_idx_array].values, axis=1)
            elif src == 'date':
                # df[new_col+'c_stime'] = df.iloc[:,col_idx_array].min(axis=1, skipna=True)  # replaced w/ faster
                df[new_col+'c_stime'] = np.nanmin(df.iloc[:, col_idx_array].values, axis=1)
                # df[new_col+'d_etime'] = df.iloc[:,col_idx_array].max(axis=1, skipna=True)  # replaced w/ faster
                df[new_col+'d_etime'] = np.nanmax(df.iloc[:, col_idx_array].values, axis=1)
            else:
                raise ValueError('Invalid scr')
        df_t = df[[col for col in df.columns if col.startswith('EF__' + loc_name[0].upper())]]

        if src == 'date':
            pu('Populating additional information...', int_pad=4)
            df_t = populate_line_station_pod(df_t, loc_name[0])

            pu('Creating timestamps...', int_pad=4)
            cols = [col for col in df_t.columns if col.endswith('stime') or col.endswith('etime')]
            df_t[cols].to_pickle(s.DATA_DIR + f_name[:-1] + loc_name[0].lower()+'_ctime.pkl')
            df_t = add_timestamp_cols(df_t, cols)
            df_t = df_t.select_dtypes(exclude=['object', 'datetime64[ns]'])
        pu('Saving ' + loc_name + ': ' + f_name[:-1] + loc_name[0].lower(), int_pad=3)
        df_t.to_pickle(s.DATA_DIR + f_name[:-1] + loc_name[0].lower()+'.pkl')
    return


def populate_line_station_pod(data, loc_type):  # populate line, station and pod
    import pandas as pd
    import warnings
    from utils import pu
    from pandas.core.common import SettingWithCopyWarning
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    pu('Processing ' + loc_type.lower() + ' ...', int_pad=5)
    # sys.stdout.flush()
    pre = 'EF__' + loc_type + '_'  # L_ for line, P_ for Pod, S_ for Station
    start_time_cols = sorted([col for col in data.columns if col.startswith(pre) if col.endswith('stime')])
    end_time_cols = sorted([col for col in data.columns if col.startswith(pre) if col.endswith('etime')])
    start_and_end_time_cols = sorted([col for col in data.columns if col.startswith(pre) if col.endswith('time')])
    df = data[start_and_end_time_cols]
    for i, col in enumerate(start_time_cols):  # iterate through all starting times
        col_root = col[0:8]  # grab root / first 4 chars of column (eg. L_00)
        # print(col_root)
        df[col_root + '__'] = np.where(df[col].isnull(), np.nan,
                                       i + 1)  # populated with column number if part when through this location
        df[col_root + 'e_ttime'] = np.where(df[col].isnull(),
                                            np.nan, df[col_root + 'd_etime'] - df[
                                                col_root + 'c_stime'])  # populated w/ col nbr if part used this loc
        df[col_root + 'a_time_frm_psta'] = np.NaN  # create columns for future use
        df[col_root + 'b_psta'] = np.NaN
        df[col_root + 'f_nsta'] = np.NaN
        df[col_root + 'g_time_to_nsta'] = np.NaN
    used_loc = sorted([col for col in df.columns if col.startswith(pre) if col.endswith('__')])
    df[pre + '99p_first_location'] = df[used_loc].min(axis=1)
    df[pre + '99q_last_location'] = df[used_loc].max(axis=1)
    df[pre + '99r_prod_start_date'] = df[start_time_cols].min(axis=1)
    df[pre + '99s_prod_end_date'] = df[end_time_cols].max(axis=1)
    df[pre + '99t_total_prod_time'] = df[pre + '99s_prod_end_date'] - df[pre + '99r_prod_start_date']
    df.sort_index(axis=1, inplace=True)
    for i, col in enumerate(used_loc):
        previous_cols = [x for x in used_loc if x < col]
        df[col[:-2] + 'b_psta'] = np.where(df[col].isnull(), np.NaN,
                                           df[previous_cols].max(axis=1))  # -1 to remove second _ from __
    for i, col in enumerate(used_loc):
        next_cols = [x for x in used_loc if x > col]
        df[col[:-2] + 'f_nsta'] = np.where(df[col].isnull(), np.NaN,
                                           df[next_cols].min(axis=1))  # -1 to remove second _ from __
    for i, col in enumerate(used_loc):
        next_cols = [x for x in used_loc if x > col]
        next_cols = [x[0:8] + 'c_stime' for x in next_cols]
        df[col[:-2] + 'g_time_to_nsta'] = np.where(df[col].isnull(), np.NaN,
                                                   df[next_cols].min(axis=1))  # -1 to remove second _ from __
        df[col[:-2] + 'g_time_to_nsta'] = df[col[:-2] + 'g_time_to_nsta'] - df[col[:-2] + 'd_etime']
    for i, col in enumerate(used_loc):
        next_cols = [x for x in used_loc if x < col]
        next_cols = [x[0:8] + 'd_etime' for x in next_cols]
        df[col[:-2] + 'a_time_frm_psta'] = np.where(df[col].isnull(), np.NaN,
                                                    df[next_cols].max(axis=1))  # -1 to remove second _ from __
        df[col[:-2] + 'a_time_frm_psta'] = df[col[:-2] + 'c_stime'] - df[col[:-2] + 'a_time_frm_psta']
    wait_cols = [col for col in df if col.endswith('g_time_to_nsta') if col.startswith(pre)]
    df[pre + '99u_total_wait_time'] = df[wait_cols].sum(axis=1)
    work_cols = [col for col in df if col.endswith('e_ttime') if col.startswith(pre)]
    df[pre + '99v_total_work_time'] = df[work_cols].sum(axis=1)
    start_times = df[pre + '99r_prod_start_date'].unique()
    x = pd.Series(list(range(len(start_times))), index=start_times).to_dict()
    df[pre + '99w_starting_batch'] = df[pre + '99r_prod_start_date'].map(x)
    if loc_type == 'L':
        df = add_timestamp_cols(df, [pre + '99r_prod_start_date'])
        df = add_timestamp_cols(df, [pre + '99s_prod_end_date'])
    return df


def create_part_routes():  # create unique routes for parts through line, stations and pods
    import os
    import pandas as pd
    import getdata
    from utils import train_test_separation, pu

    for each in ['Line', 'Station', 'Pod']:
        if each == 'Line':
            if os.path.isfile(s.DATA_DIR + 'tr_routes_'+each[0:1].lower() + '.pkl') and \
                    os.path.isfile(s.DATA_DIR + 'te_routes_'+each[0:1].lower() + '.pkl'):
                pu('Located tr_routes_'+each[0:1].lower() + '.pkl', int_pad=2)
                pu('Located te_routes_'+each[0:1].lower() + '.pkl', int_pad=2)
                continue
            else:
                pu('Processing ' + each + '...', int_pad=2)
                data = getattr(getdata, 'al_dt_l_ctime')()
        elif each == 'Station':
            if os.path.isfile(s.DATA_DIR + 'tr_routes_'+each[0:1].lower() + '.pkl') and \
                    os.path.isfile(s.DATA_DIR + 'te_routes_'+each[0:1].lower() + '.pkl'):
                pu('Located tr_routes_'+each[0:1].lower() + '.pkl', int_pad=2)
                pu('Located te_routes_'+each[0:1].lower() + '.pkl', int_pad=2)
                continue
            else:
                pu('Processing ' + each + '...', int_pad=2)
                data = getattr(getdata, 'al_dt_s_ctime')()
        elif each == 'Pod':
            if os.path.isfile(s.DATA_DIR + 'tr_routes_'+each[0:1].lower() + '.pkl') and \
                    os.path.isfile(s.DATA_DIR + 'te_routes_'+each[0:1].lower() + '.pkl'):
                pu('Located tr_routes_'+each[0:1].lower() + '.pkl', int_pad=2)
                pu('Located te_routes_'+each[0:1].lower() + '.pkl', int_pad=2)
                continue
            else:
                pu('Processing ' + each + '...', int_pad=2)
                data = getattr(getdata, 'al_dt_p_ctime')()
        else:
            raise ValueError('Bad filename in create part routes')
        cols = [col for col in data.columns if col.endswith('_stime')]
        df = data[cols].copy()
        df.fillna(0, inplace=True)  # required because NaN evaluates to True for bool on next line
        df = df.astype(bool).astype(int)
        shopfloor(df, each)
        df2 = df.drop_duplicates(keep='first').copy()
        df2.reset_index(inplace=True, drop=True)
        df2['EF__'+each[0:1] + '_99x_route'] = df2.index
        df['Id'] = df.index
        df = pd.merge(df, df2, left_on=cols, right_on=cols)
        df.drop(cols, inplace=True, axis=1)
        df.set_index('Id', drop=True, inplace=True, verify_integrity=True)
        df.sort_index(axis=0, inplace=True)
        cols = [col for col in df.columns if col.endswith('_route')]
        df, df_1 = train_test_separation(df[cols], True)  # need to update this
        pu('Saving ' + each.lower() + ' routes...', int_pad=3)
        df.to_pickle(s.DATA_DIR + 'tr_routes_'+each[0:1].lower() + '.pkl')
        df_1.to_pickle(s.DATA_DIR + 'te_routes_'+each[0:1].lower() + '.pkl')
    return


def add_timestamp_cols(data, cols):  # add timestamp columns and temporal encode appropriate columns
    import pandas as pd
    import datetime as dt
    suf = '_ts'
    for col in cols:
        data[col+suf] = [np.nan if x == 0 else (x * 100 * 6 / 60 / 24) for x in data[col]]
        # for reference, add 41640 for ordinal date
        data[col + suf] = pd.TimedeltaIndex(data[col+suf], unit='d') + dt.datetime(2014, 1, 1)
        data[col + suf] = data[col+suf].dt.round('min')
        data[col + 'date_only'+suf] = data[col+suf].dt.date
        data[col + '_month'] = data[col+suf].dt.month    # np.sin((df.month - 1) * (2. * np.pi / 12))
        data[col + '_month_sin'] = np.sin((data[col+suf].dt.month - 1) * (2. * np.pi / 12))
        data[col + '_month_cos'] = np.cos((data[col+suf].dt.month - 1) * (2. * np.pi / 12))
        data[col + '_wday'] = data[col + suf].dt.weekday
        data[col + '_wday_sin'] = np.sin(data[col+suf].dt.weekday) * (2. * np.pi / 7)
        data[col + '_wday_cos'] = np.cos(data[col+suf].dt.weekday) * (2. * np.pi / 7)
    return data


def shopfloor(data, cur_col):  # create shopfloor diagram
    df = data.copy()
    shopfloor_plot(df, sort_by=list(range(0, len(df.columns))), f_sta=0, l_sta=len(df.columns), cur_col=cur_col)


def shopfloor_plot(df, sort_by, f_sta, l_sta, cur_col):  # create shopfloor diagram
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    df.columns = range(df.shape[1])
    df = df.sort_values(by=sort_by, ascending=True)  # parts processed though stations will have 1's, others 0's.
    id_matrix = np.array(df[df.columns[f_sta:l_sta]].values, dtype=float)  # transpose so station on y-axis
    fig, ax = plt.subplots(figsize=(20, 20))
    aspect_ratio = id_matrix.shape[1] / id_matrix.shape[0] * .5
    ax.imshow(id_matrix, cmap='binary', interpolation='nearest', aspect=aspect_ratio, alpha=1)
    plt.xticks(range(id_matrix.shape[1]), df.columns[f_sta:l_sta], rotation=30)
    plt.gca().invert_yaxis()
    plt.tick_params(labelsize=20)
    plt.xlabel(cur_col, fontsize=20)
    plt.ylabel('Production Volume', fontsize=20)
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    plt.xticks(range(id_matrix.shape[1]), df.columns[f_sta:l_sta], rotation=45)
    plt.savefig(s.REPORTS_DIR + 'shopfloor' + cur_col + '.png')
    plt.show()


def part_dist():  # create part order (e.g. which part come before/after current part)
    import os
    import pandas as pd
    from utils import pu, train_test_separation
    import getdata
    if os.path.isfile(s.DATA_DIR + 'tr_dt_rec_ord.pkl') and os.path.isfile(s.DATA_DIR + 'te_dt_rec_ord.pkl'):
        pu('Located tr_dt_rec_ord.pkl', int_pad=2)
        pu('Located te_dt_rec_ord.pkl', int_pad=2)
    else:
        pu('Loading data...', int_pad=2)
        df = getattr(getdata, 'al_dt_o')()
        df = pd.DataFrame(df.min(axis=1))
        df.columns = ['stime']
        df.reset_index(inplace=True)
        pu('Calculating distance...', int_pad=2)
        for i in range(1, 10):
            prefix = 'EF__ord_'
            suffix = str(i)
            df[prefix + 'prev_id_a' + suffix] = df['Id'].shift(i)
            df[prefix + 'next_id_a' + suffix] = df['Id'].shift(-i)
            df[prefix + 'dist_frm_last_a' + suffix] = df['Id'] - df[prefix + 'prev_id_a' + suffix]
            df[prefix + 'dist_to_next_a' + suffix] = df['Id'] - df[prefix + 'next_id_a' + suffix]
        df = df.sort_values(['stime', 'Id'], ascending=True)
        for i in range(1, 10):
            prefix = 'EF__ord_'
            suffix = str(i)
            df[prefix + 'prev_id_b' + suffix] = df['Id'].shift(i)
            df[prefix + 'next_id_b' + suffix] = df['Id'].shift(-i)
            df[prefix + 'dist_frm_last_b' + suffix] = df['Id'] - df[prefix + 'prev_id_b' + suffix]
            df[prefix + 'dist_to_next_b' + suffix] = df['Id'] - df[prefix + 'next_id_b' + suffix]
        df.drop(['stime'], inplace=True, axis=1)
        df.set_index('Id', drop=True, inplace=True)
        df.sort_index(inplace=True)
        df_train, df_test = train_test_separation(df, check_rows=True)

        pu('Saving pickle...', int_pad=2)
        df_train.to_pickle(s.DATA_DIR + 'tr_dt_rec_ord.pkl')
        df_test.to_pickle(s.DATA_DIR + 'te_dt_rec_ord.pkl')
    return


def que_and_wip(prefix, loc_name, source_file):  # create master que and wip file
    import pandas as pd
    import gc
    import os
    from utils import pu
    import getdata
    pd.options.mode.chained_assignment = None  # default='warn'

    if os.path.isfile(s.DATA_DIR + loc_name[0:1].lower() + '_wip' + '.pkl'):
        pu('Located '+loc_name[0:1].lower() + '_wip' + '.pkl', int_pad=2)
    else:
        pu('Creating ' + loc_name.lower(), int_pad=2)
        pu('Loading Data', int_pad=3)
        df = getattr(getdata, source_file)()
        x = [col for col in df.columns if col.endswith('stime') or
             col.endswith('__') or col.endswith('99p_first_location')]
        pu('Getting start and stop times', int_pad=3)
        df_stime = df[x]
        df_stime.reset_index(inplace=True)
        df_stime.columns = [*df_stime.columns[:-1], 'first_loc']  # rename ..99p_first_location
        loc_list = sorted(list(set([col[0:8] for col in df_stime.columns.values if col != 'Id' and
                                    col != 'first_loc'])))

        ''' 
        parts always follow the same pattern...  
            exit_que -> enter_wip -> exit_wip -> enter_que
                first "for loop"
                    exit_que is always the same # as enter wip.  
                        One adjustment is made for products when they start
                    enter_wip is always the current location
                second "for loop"
                    exit_wip is always the current location
                    enter_que is always the next station (_nsta)
                        if _nsta equals NaN means the product is finished 
                            which means we move it to 99_que which is effectively finished goods
                            the 99's are named to for coding ease, 99_wip is not used
        '''

        df1 = pd.DataFrame()
        for loc in loc_list:
            sel_cols = ['Id'] + [col for col in df_stime.columns if col.startswith(loc)] + ['first_loc']
            df_t = df_stime[sel_cols]
            df_t.columns = ['Id', 'enter_wip', 'time', 'first_loc']
            df_t['exit_que'] = df_t['enter_wip']
            df1 = pd.concat([df1, df_t])
        df1 = df1[['Id', 'first_loc', 'time', 'exit_que', 'enter_wip']]
        df1.dropna(subset=['time'], inplace=True)  # any observations without time are not valuable for prediction
        x = [col for col in df.columns if col.endswith('etime') or col.endswith('f_nsta')
             or col.endswith('__') or col.endswith('99p_first_location')]
        df_etime = df[x]
        df_etime.reset_index(inplace=True)
        df_etime.columns = [*df_etime.columns[:-1], 'first_loc']
        location_list = sorted(list(set([col[0:8] for col in df_etime.columns.values if col != 'Id'
                                         and col != 'first_loc'])))

        '''
        second "for loop" as described in last comment
        -> exit_wip -> enter_que  
        '''

        df2 = pd.DataFrame()
        for loc in location_list:
            sel_cols = ['Id'] + [col for col in df_etime.columns if col.startswith(loc)] + ['first_loc']
            df_t = df_etime[sel_cols]
            df_t.columns = ['Id', 'exit_wip', 'time', 'enter_que', 'first_loc']
            df2 = pd.concat([df2, df_t])
        df2 = df2[['Id', 'first_loc', 'time', 'exit_wip', 'enter_que']]
        df2.dropna(subset=['time'], inplace=True)  # Nan times not useful
        df2['enter_que'] = df2['enter_que'].fillna(99)  # if enter_que (next_que) is blank, fill 99 (finished goods)
        df = pd.concat([df1, df2], sort=True)
        # memory tight when calculating stations
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        df_stime = pd.DataFrame()
        df_etime = pd.DataFrame()
        df_t = pd.DataFrame()
        gc.collect()
        pu('Sorting...', int_pad=3)
        df['sort_v'] = 0
        df['sort_v'] = df[[col for col in df.columns if col != 'Id' if col != 'time' if col != 'first_loc']].sum(
            axis=1)  # axis=1 causes Nan ignore
        df = df[['Id', 'sort_v', 'first_loc', 'time', 'exit_que', 'enter_wip', 'exit_wip', 'enter_que']]
        # df.sort_values(by=['Id', 'time', 'sort_v'], ascending=True, inplace=True)
        df.sort_values(by=['time', 'Id', 'sort_v'], ascending=True, inplace=True)
        l_list_s = sorted(list(pd.unique(df[['exit_que', 'enter_wip', 'exit_wip', 'enter_que']].values.ravel('K'))))
        l_list_s = [str(int(x)).zfill(2) for x in l_list_s if str(x) != 'nan']

        pu('Populating que and wip columns', int_pad=3)
        pu('Section 1', int_pad=4)
        for ss in l_list_s:
            df[prefix + ss + '_que'] = 0
            df[prefix + ss + '_wip'] = 0
        pu('Section 2', int_pad=4)
        for ss in l_list_s:  # for example, for Line ss '01' '02' '03' '04'
            i = int(ss)
            df[prefix + ss + '_que'] = np.where(df['exit_que'] == i, -1, df[prefix + ss + '_que'])
            df[prefix + ss + '_wip'] = np.where(df['enter_wip'] == i, 1, df[prefix + ss + '_wip'])
            df[prefix + ss + '_wip'] = np.where(df['exit_wip'] == i, -1, df[prefix + ss + '_wip'])
            df[prefix + ss + '_que'] = np.where(df['enter_que'] == i, 1, df[prefix + ss + '_que'])
        pu('Section 3', int_pad=4)
        for ss in l_list_s:
            df[prefix + ss + '_que'] = np.where(df['exit_que'] == df['first_loc'], 0, df[prefix + ss + '_que'])
        pu('Removing unneeded columns', int_pad=3)
        df.reset_index(drop=True, inplace=True)    # set_index('time', inplace=True)
        sel_cols = sorted([col for col in df.columns if col.endswith('_wip') or col.endswith('_que')  if not col.startswith('enter')  if not col.startswith('exit')])
        sel_cols = ['time'] + sel_cols
        n = 250000  # chunk row size, only way I could get to execute in available memory (probably a better way)
        list_df = [df[i:i + n] for i in range(0, df.shape[0], n)]  # break up DataFrame into 250k row chunks
        df = pd.DataFrame()  # assign this again shortly, trying to squeeze a little more memory
        gc.collect()
        for i in range(0, len(list_df)):
            list_df[i] = list_df[i][sel_cols]  # keep the columns we want
        df = pd.concat(list_df)  # pull all the chunks back together
        list_df = []  # free up memory
        gc.collect()
        df.set_index('time', inplace=True)
        pu('Calculating running balance', int_pad=3)
        for i, col in enumerate(df.columns):
            df[col] = numba_cumsum(df[col].to_numpy())  # extremely fast
        pu('Deleting duplicates', int_pad=3)
        # df.drop_duplicates(subset=['time'], keep='last', inplace=True)  # slow and fails - replaced with next line
        df = df[~df.index.duplicated(keep='last')]  # the index is time
        pu('Saving pickle', int_pad=3)
        df.to_pickle(s.DATA_DIR + loc_name[0:1].lower() + '_wip' + '.pkl')


def populate_wip(loc_name, source_file, wip_file):  # create que and wip by id based on time at line/station/pod
    import pandas as pd
    import os
    from utils import pu
    import getdata
    pd.options.mode.chained_assignment = None  # default='warn'

    if os.path.isfile(s.DATA_DIR + loc_name[0:1].lower() + '_wip_by_id.pkl'):
        pu('Located ' + loc_name[0:1].lower() + '_wip_by_id.pkl', int_pad=2)
    else:
        pu('Populating ' + loc_name.lower() + ' wip', int_pad=2)
        pu('Loading Data', int_pad=3)
        df = getattr(getdata, source_file)()
        x = [col for col in df.columns if col.endswith('stime')]
        pu('Getting start and stop times', int_pad=3)
        df_stime = df[x]
        df = getattr(getdata, wip_file)()
        cols = df_stime.columns.tolist()
        for i, col in enumerate(cols):
            df_stime = pd.merge(df_stime, df[[col[0:8] + '_que', col[0:8] + '_wip']], left_on=col, right_index=True,
                                how='left', copy=False)
        x = [col for col in df_stime.columns if col.endswith('_wip') or col.endswith('_que')]
        df_stime[x].to_pickle(s.DATA_DIR + loc_name[0:1].lower() + '_wip_by_id.pkl')


@njit
def numba_cumsum(df):  # helps speed up cumsum calculation
    x = np.cumsum(df)
    return x


def comb_files(source_files, dest_file, caxis):  # helper for combining files
    import os
    from utils import pu
    import getdata
    import pandas as pd
    if os.path.isfile(s.DATA_DIR + dest_file + '.pkl'):
        pu('Located ' + dest_file + '.pkl', int_pad=2)
    else:
        if len(source_files) == 2:
            pu('Creating ' + dest_file + '.pkl', int_pad=2)
            df = pd.concat([getattr(getdata, source_files[0])(),
                            getattr(getdata, source_files[1])()], axis=caxis, sort=True)
            df.to_pickle(s.DATA_DIR + dest_file + '.pkl')
        elif len(source_files) == 3:
            pu('Creating ' + dest_file + '.pkl', int_pad=2)
            df = pd.concat([getattr(getdata, source_files[0])(),
                            getattr(getdata, source_files[1])(),
                            getattr(getdata, source_files[2])()], axis=caxis, sort=True)
            df.to_pickle(s.DATA_DIR + dest_file + '.pkl')
        elif len(source_files) == 4:
            pu('Creating ' + dest_file + '.pkl', int_pad=2)
            df = pd.concat([getattr(getdata, source_files[0])(),
                            getattr(getdata, source_files[1])(),
                            getattr(getdata, source_files[2])(),
                            getattr(getdata, source_files[3])()], axis=caxis, sort=True)
            df.to_pickle(s.DATA_DIR + dest_file + '.pkl')
        elif len(source_files) == 5:
            pu('Creating ' + dest_file + '.pkl', int_pad=2)
            df = pd.concat([getattr(getdata, source_files[0])(),
                            getattr(getdata, source_files[1])(),
                            getattr(getdata, source_files[2])(),
                            getattr(getdata, source_files[3])(),
                            getattr(getdata, source_files[4])()], axis=caxis, sort=True)

            df.to_pickle(s.DATA_DIR + dest_file + '.pkl')
        else:
            raise ValueError('Invalid number of files in source files')
