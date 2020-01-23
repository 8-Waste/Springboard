import settings as s


def main():
    import os
    from utils import pu, reset_luigi_status
    from getdata import make_pickles, luigi_make_source_pickles
    from eda import eda
    from feats import feats_selection_main
    from train import train_main

    pu('Housekeeping...')
    os.remove(s.LOG_FILE) if os.path.isfile(s.LOG_FILE) else None  # reset the log on each run
    reset_luigi_status()  # ensure luigi checks all files

    luigi_make_source_pickles()      # make source pickles for faster access

    make_pickles()  # make source pickles for faster access

    eda()      # prep data

    feats_selection_main()      # select features from main sources and combine

    # train reference model, first model, tune model, final model
    train_main(reference_model=False, regular_model=False, tune_model=False, final_train_model=True)

    pu('ALL COMPLETE', blank_lines=1)


if __name__ == '__main__':
    main()
