ALL = 10000000  # used for testing, allows limiting records to reduce execution time
DATA_DIR = '../data/'  # data directory
REPORTS_DIR = '../reports/'  # reports directory
LOG_FILE = '../data/log.txt'  # file to log messages, resets every time main.py is executed
COMPUTER_STATS = True  # set to false to remove CPU and RAM from messasges

SRC_FILES = {'tr_nu_o': ['train_numeric'],
             'te_nu_o': ['test_numeric'],
             'tr_dt_o': ['train_date'],
             'te_dt_o': ['test_date'],
             'al_nu_o': ['tr_nu_o', 'te_nu_o'],
             'al_dt_o': ['tr_dt_o', 'te_dt_o'],
             'tr_al_o': ['tr_nu_o', 'tr_dt_o'],
             'te_al_o': ['te_nu_o', 'te_dt_o']}

MY_COLORS = ["#0000FF",  # blue
             "#FF0000",  # red
             "#00FF00",  # green
             "#FFFF00",  # yellow
             "#00FFFF",  # aqua
             "#FF00FF"]  # fuchsia
MY_BLUE = ["#0000FF"]
MY_RED = ["#FF0000"]
