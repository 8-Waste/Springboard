MY_COLORS = ["#0000FF",  # blue
             "#FF0000",  # red
             "#00FF00",  # green
             "#FFFF00",  # yellow
             "#00FFFF",  # aqua
             "#FF00FF"   # fuchsia
             ]
MY_BLUE = ["#0000FF"]
MY_RED = ["#FF0000"]


def data_overview():
    from utils import sum1forline
    from getdata import get_data

    training_rows = sum1forline('../data/train_numeric.csv')
    testing_rows = sum1forline('../data/test_numeric.csv')
    total_rows = training_rows + testing_rows
    numeric_features = get_data('train_numeric', 1)
    date_features = get_data('train_date', 1)
    categorical_features = get_data('train_categorical', 1)
    total_features = numeric_features.shape[1] + date_features.shape[1] + categorical_features.shape[1]
    failed_products = sum(get_data('target')['Response'])

    print(' ')
    print('Product Data (rows)')
    print(f"   {'Training rows:':<30} {training_rows:>15,}")
    print(f"   {'Testing rows:':<30} {testing_rows:>15,}")
    print(f"   {'Total rows:':<50} {total_rows:>15,}")
    print('\n')
    print('Features (columns')
    print(f"   {'Numeric Features:':<30} {numeric_features.shape[1]:>15,}")
    print(f"   {'Date Features:':<30} {date_features.shape[1]:>15,}")
    print(f"   {'Categorical Features:':<30} {categorical_features.shape[1]:>15,}")
    print(f"   {'Total Features:':<50} {total_features:>15,}")
    print('\n')
    print('Various Statistics')
    print(f"   {'Failed Products (rows):':<30} {failed_products:>15,}")
    print(f"   {'Total Products (rows):':<30} {training_rows:>15,}")
    print(f"   {'Failure Rate:':<60} 1/{round(1 / (failed_products / training_rows), 0):.0f}")
    print(f"   {'Failure Rate Percent:':<50} {failed_products / training_rows:>15.2%}")
    print('\n')
    print(f"   {'~Potential Data Elements (features x rows):':<50} {total_features * total_rows:>15,}")
    print('\n')


def bar_plot(code):
    from utils import get_bar_plot
    if code == 'count_id_per_line':
        get_bar_plot(data='../data/special/data_source_100.h5',
                     title='Training Data vs Test Data',
                     suptitle="Count of Product Id's through Line",
                     x_column='Line',
                     x_label='Line',
                     y_column='Total',
                     y_label='Total',
                     hue='Source',
                     figure_size=(8, 5))
    if code == 'count_id_per_station':
        get_bar_plot(data='../data/special/data_source_102.h5',
                     title='Training Data vs Test Data ',
                     suptitle="Count of Product Id's through Stations",
                     x_column='Station',
                     x_label='Line',
                     y_column='Total',
                     y_label='Total',
                     hue='Source')
    if code == 'error_rate_per_line':
        get_bar_plot(data='../data/special/data_source_104.h5',
                     title='(Total Line Errors) / (Volume through Line)',
                     suptitle="Error Rate per Line",
                     x_column='Line',
                     x_label='Line',
                     y_column='Error_Pct',
                     y_label='Line Error Percent',
                     y_tick_labels=2,
                     hue=None,
                     palette=MY_BLUE,
                     figure_size=(8, 5))
    if code == 'total_errors_per_line':
        get_bar_plot(data='../data/special/data_source_104.h5',
                     title='(Total Line Errors) / (All Errors)',
                     suptitle="Percent of Total Error by Line",
                     x_column='Line',
                     x_label='Line',
                     y_column='Error_Pct_Total',
                     y_label='Percent of Total Errors',
                     y_tick_labels=2,
                     hue=None,
                     palette=MY_BLUE,
                     figure_size=(8, 5))
    if code == 'pct_total_error_per_station':
        get_bar_plot(data='../data/special/data_source_106.h5',
                     title='(Total Station Errors) / (All Errors)',
                     suptitle="Percent of Total Error by Station",
                     x_column='Station',
                     x_label='Station',
                     y_column='Error_Pct_Total',
                     y_label='Percent of Total Errors',
                     y_tick_labels=2,
                     hue=None,
                     palette=MY_BLUE)
    if code == 'pct_error_per_station':
        get_bar_plot(data='../data/special/data_source_106.h5',
                     title='(Total Line Errors) / (All Errors)',
                     suptitle="Percent Error by Station",
                     x_column='Station',
                     x_label='Station',
                     y_column='Error_Pct',
                     y_label='Station Error Percent',
                     y_tick_labels=2,
                     hue=None,
                     palette=MY_RED)


if __name__ == '__main__':
    data_overview()
