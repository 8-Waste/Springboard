import luigi
from test_numeric import TestNumeric

class TrainNumeric(luigi.Task):

    def requires(self):
        return [TestNumeric()]

    def output(self):
        return luigi.LocalTarget('luigi_status/tr_nu_o.txt')

    def run(self):
        import settings as s
        from getdata import check_for_pickle
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)

        pkl = 'tr_nu_o'
        pkl_src_files = s.SRC_FILES[pkl]
        check_for_pickle(pkl, pkl_src_files)

        with self.output().open('w') as outfile:
            for item in pkl_src_files:
                outfile.write("%s\n" % item)


