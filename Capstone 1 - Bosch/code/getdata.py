import settings as s


def load_target(rows=s.ALL, subset=False):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import pandas as pd
    if subset:
        return pd.read_csv(s.DATA_DIR + 'train_numeric.csv', nrows=100000, index_col='Id', usecols=['Id', 'Response'])
    else:
        return pd.read_csv(s.DATA_DIR + 'train_numeric.csv', nrows=rows, index_col='Id', usecols=['Id', 'Response'])


def tr_nu_o():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'tr_nu_o'+'.pkl')


def te_nu_o():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'te_nu_o'+'.pkl')


def tr_dt_o():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'tr_dt_o'+'.pkl')


def te_dt_o():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'te_dt_o'+'.pkl')


def al_dt_o():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'al_dt_o'+'.pkl')


def tr_al_o():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'tr_al_o'+'.pkl')


def tr_al_l():
    import pandas as pd
    df = pd.read_pickle(s.DATA_DIR + 'tr_nu_l.pkl')
    df_1 = pd.read_pickle(s.DATA_DIR + 'tr_dt_l.pkl')
    return pd.concat([df, df_1], axis=1)


def tr_al_s():
    import pandas as pd
    df = pd.read_pickle(s.DATA_DIR + 'tr_nu_s.pkl')
    df_1 = pd.read_pickle(s.DATA_DIR + 'tr_dt_s.pkl')
    return pd.concat([df, df_1], axis=1)


def tr_al_p():
    import pandas as pd
    df = pd.read_pickle(s.DATA_DIR + 'tr_nu_p.pkl')
    df_1 = pd.read_pickle(s.DATA_DIR + 'tr_dt_p.pkl')
    return pd.concat([df, df_1], axis=1)


def tr_dt_rec_ord():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'tr_dt_rec_ord'+'.pkl')


def te_dt_rec_ord():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'te_dt_rec_ord'+'.pkl')


def tr_dt_l():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'tr_dt_l.pkl')


def te_dt_l():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'te_dt_l.pkl')


def al_dt_l():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'al_dt_l.pkl')


def tr_dt_s():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'tr_dt_s.pkl')


def te_dt_s():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'te_dt_s.pkl')


def al_dt_s():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'al_dt_s.pkl')


def tr_dt_p():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'tr_dt_p.pkl')


def tr_nu_l():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'tr_nu_l.pkl')


def tr_nu_s():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'tr_nu_s.pkl')


def tr_nu_p():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'tr_nu_p.pkl')


def te_dt_p():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'te_dt_p.pkl')


def al_dt_p():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'al_dt_p.pkl')


def l_wip():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'l_wip.pkl')


def s_wip():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 's_wip.pkl')


def p_wip():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'p_wip.pkl')


def tr_route_l():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'tr_routes_l.pkl')


def tr_route_s():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'tr_routes_s.pkl')


def tr_route_p():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'tr_routes_p.pkl')


def l_wip_by_id():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'l_wip_by_id.pkl')


def s_wip_by_id():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 's_wip_by_id.pkl')


def p_wip_by_id():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'p_wip_by_id.pkl')


def l_final_feats_all():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'l_final_feats_all.pkl')


def s_final_feats_all():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 's_final_feats_all.pkl')


def p_final_feats_all():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'p_final_feats_all.pkl')


def o_final_feats_all():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'o_final_feats_all.pkl')


def s_final_feats_subset():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 's_final_feats_subset.pkl')


def p_final_feats_subset():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'p_final_feats_subset.pkl')


def o_final_feats_subset():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'o_final_feats_subset.pkl')


def l_final_feats_subset():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'l_final_feats_subset.pkl')


def master_final_feats_all():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'master_final_feats_all.pkl')


def master_final_feats_subset():
    import pandas as pd
    return pd.read_pickle(s.DATA_DIR + 'master_final_feats_subset.pkl')


def al_dt_l_ctime():
    import pandas as pd
    df = pd.read_pickle(s.DATA_DIR + 'tr_dt_l_ctime' + '.pkl')
    df_1 = pd.read_pickle(s.DATA_DIR + 'te_dt_l_ctime' + '.pkl')
    df = pd.concat([df, df_1], axis=0)
    return df


def al_dt_s_ctime():
    import pandas as pd
    df = pd.read_pickle(s.DATA_DIR + 'tr_dt_s_ctime' + '.pkl')
    df_1 = pd.read_pickle(s.DATA_DIR + 'te_dt_s_ctime' + '.pkl')
    df = pd.concat([df, df_1], axis=0)
    return df


def al_dt_p_ctime():
    import pandas as pd
    df = pd.read_pickle(s.DATA_DIR + 'tr_dt_p_ctime' + '.pkl')
    df_1 = pd.read_pickle(s.DATA_DIR + 'te_dt_p_ctime' + '.pkl')
    df = pd.concat([df, df_1], axis=0)
    return df


def check_for_pickle(f_name, source_files):
    import os
    import pandas as pd
    from utils import pu

    if not os.path.isfile(s.DATA_DIR + f_name + '.pkl'):
        pu('Creating ' + f_name + '.pkl', int_pad=1)
        if len(source_files) == 1:
            df = pd.read_csv(s.DATA_DIR + source_files[0] + '.csv', index_col='Id')
            if 'Response' in df.columns:
                df.drop(['Response'], inplace=True, axis=1)
        elif len(source_files) == 2:
            if os.path.isfile(s.DATA_DIR + source_files[0] + '.pkl'):
                df = pd.read_pickle(s.DATA_DIR + source_files[0] + '.pkl')
            else:
                df = pd.read_csv(s.DATA_DIR + source_files[0] + '.csv', index_col='Id')
            if os.path.isfile(s.DATA_DIR + source_files[1] + '.pkl'):
                df_1 = pd.read_pickle(s.DATA_DIR + source_files[1] + '.pkl')
            else:
                df_1 = pd.read_csv(s.DATA_DIR + source_files[1] + '.csv', index_col='Id')
            if len(df) == len(df_1):
                df = pd.concat([df, df_1], sort=True, axis=1)
            else:
                df = pd.concat([df, df_1], sort=True, axis=0)
        else:
            print('problem')
        df.to_pickle(s.DATA_DIR + f_name + '.pkl')
    else:
        pu('Located ' + f_name + '.pkl', int_pad=1)


def make_pickles():
    from utils import pu
    pu('Verifying files...', int_pad=0)
    for pkl, pkl_src_files in s.SRC_FILES.items():
        check_for_pickle(pkl, pkl_src_files)
    pu('Files verified...', int_pad=0, blank_lines=1)


def luigi_make_source_pickles():
    import luigi
    import logging
    import sys
    from utils import pu

    logging.disable(100)  # disable Luigi screen logging
    sys.path.append('luigi_code/')  # luigi in path
    pu('Checking for files...', int_pad=0)
    # check for files, create if necessary
    from test_all import TestAll
    luigi.build([TestAll()], workers=2, local_scheduler=True)
    pu('File check complete...', int_pad=0, blank_lines=1)
