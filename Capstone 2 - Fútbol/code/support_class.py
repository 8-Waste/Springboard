import settings as s

import concurrent.futures
import cv2
from cv2 import Canny
from functools import partial
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from matplotlib import transforms
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np
from pathlib import Path
import pandas as pd
import pickle
import rawpy
import seaborn as sns
from skimage.filters import laplace, sobel, roberts
from sklearn.preprocessing import binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef
import time


# decorator to help with timing functions
def time_it(func):
    """Decorator function to time functions in Jupyter Notebook"""

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f'{time.strftime("%H:%M:%S", time.gmtime(int(time.time() - start)))}')
        return result

    return wrapper


# class for holding functions for project
class Fball:

    # print blank lines
    @staticmethod
    def bl(qty=1):
        for ln in range(qty):
            print()

    # save model
    @staticmethod
    def save_model(model_obj, name, ir=True):
        if ir:
            with open(s.MODELS_jn / str(name + '.pkl'), 'wb') as f:
                pickle.dump(model_obj, f, pickle.HIGHEST_PROTOCOL)
        else:
            with open(s.MODELS / str(name + '.pkl'), 'wb') as f:
                pickle.dump(model_obj, f, pickle.HIGHEST_PROTOCOL)


    # load model
    @staticmethod
    def load_model(name, ir=True):
        if ir:
            with open(s.MODELS_jn / str(name + '.pkl'), 'rb') as f:
                return pickle.load(f)
        else:
            with open(s.MODELS / str(name + '.pkl'), 'rb') as f:
                return pickle.load(f)


    @staticmethod
    def checkfile(file, ir=True):
        num = 0
        if ir:
            file = s.REPORT_IMAGES_jn / file
        else:
            file = s.REPORT_IMAGES / file
        file_stem = file.stem
        while file.exists() or num == 0:
            num += 1
            file = file.parent / Path(file_stem + '_' + str(num).zfill(2) + file.suffix)
        return file


    @staticmethod
    def count_summary(item_list, item_text_list):
        assert len(item_list) + 1 == len(item_text_list), "Invalid paramaters passed to count summary"
        total = sum([len(item) for item in item_list])
        width = len(f'{total:,.0f}')
        for i, item in enumerate(item_list):
            print(f'{len(item):>{width},.0f} {item_text_list[i]}')
        print(f'{total:>{width},.0f} {item_text_list[i + 1]}')

    def get_face_embedding(image, embedder):
        faceBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()
        return vec.flatten()

    def get_crop(img, pct):
        x = int(img.shape[0] * ((1 - pct) / 2))
        h = int(img.shape[0] * pct)
        y = int(img.shape[1] * ((1 - pct) / 2))
        w = int(img.shape[1] * pct)
        img = img[x:x + h, y:y + w]
        return img

    def skin_tone(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        skin_tone_pct = np.sum(Fball.vectorized_form(img)) / (img.shape[0] * img.shape[1])
        return skin_tone_pct

    # used with skin tone
    def vectorized_form(img):
        R, G, B = [img[:, :, x] for x in range(3)]
        delta15 = np.abs(R.astype(np.int8) - G.astype(
            np.int8)) > 15  # watch out for np.abs(R-G): because of the UNsigned numbers, they could get clipped!
        more_R_than_B = (R > B)
        is_skin_coloured_during_daytime = ((R > 95) & (G > 40) & (B > 20) &
                                           (img.ptp(axis=-1) > 15) & delta15 & (R > G) & more_R_than_B)
        is_skin_coloured_under_flashlight = ((R > 220) & (G > 210) & (B > 170) &
                                             ~delta15 & more_R_than_B & (G > B))
        return np.logical_or(is_skin_coloured_during_daytime, is_skin_coloured_under_flashlight)

    # used in focus vs unfocused model creation
    @staticmethod
    def get_data(images, pct=1):
        func = partial(Fball.process_img, pct)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            features = executor.map(func, images)
        df = pd.DataFrame(features)
        df.drop(0, axis=1, inplace=True)
        return df

    @staticmethod
    def process_img(pct, img):
        feature = []
        image_gray = cv2.imread(str(img), 0)
        image_gray = Fball.get_crop(image_gray, pct)
        lap_feat = laplace(image_gray)
        sob_feat = sobel(image_gray)
        rob_feat = roberts(image_gray)
        can_feat = Canny(image_gray, 100, 200, 3, L2gradient=True)
        feature.extend([img, lap_feat.mean(), lap_feat.var(), np.amax(lap_feat),
                        sob_feat.mean(), sob_feat.var(), np.max(sob_feat),
                        rob_feat.mean(), rob_feat.var(), np.max(rob_feat),
                        can_feat.mean(), can_feat.var(), np.max(can_feat)])
        return feature

    @staticmethod
    def get_sharp_data(image_gray):
        feature = []
        lap_feat = laplace(image_gray)
        sob_feat = sobel(image_gray)
        rob_feat = roberts(image_gray)
        can_feat = Canny(image_gray, 100, 200, 3, L2gradient=True)
        feature.extend([lap_feat.mean(), lap_feat.var(), np.amax(lap_feat),
                        sob_feat.mean(), sob_feat.var(), np.max(sob_feat),
                        rob_feat.mean(), rob_feat.var(), np.max(rob_feat),
                        can_feat.mean(), can_feat.var(), np.max(can_feat)])
        return feature

    # plot histogram of predicted probability
    @staticmethod
    def pred_prob_hist(y_pred_prob, bins=8, xlabel='positive', cutoff=0.5):
        plt.hist(y_pred_prob, bins=bins)
        plt.xlim(0, 1)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.axvline(x=cutoff, ymin=0, ymax=1, color='r', linewidth=3, linestyle='--')
        plt.title("Histogram of predicted probabilities")
        plt.xlabel("Predicted probability of " + xlabel)
        plt.ylabel('Frequency')
        colors = ['darkred']
        lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
        labels = ['Cutoff']
        plt.legend(lines, labels)
        plt.show()
        Fball.bl()
        return

    # plot feature importances
    @staticmethod
    def feat_imp_plot(df, model, n_features, max_features=30):
        n_features = min(n_features, max_features)
        d = dict(zip(df.columns, model.feature_importances_))
        ss = sorted(d, key=d.get, reverse=True)
        top_names = ss[0:n_features]
        plt.figure(figsize=(15, 5))
        plt.title("Feature importances")
        plt.bar(range(n_features), [d[i] for i in top_names], color="r", align="center")
        plt.xlim(-1, n_features)
        plt.xticks(range(n_features), top_names, rotation="vertical")


    # plot confusion matrix plus
    @staticmethod
    def plot_cm_plus(tn, fp, fn, tp):
        def range_brace(x_min, x_max, mid=0.75, beta1=50.0, beta2=100.0, height=1, initial_divisions=11,
                        resolution_factor=1.5):
            # determine x0 adaptively values using second derivitive
            # could be replaced with less snazzy:
            #   x0 = NP.arange(0, 0.5, .001)
            x0 = np.array(())
            tmpx = np.linspace(0, 0.5, initial_divisions)
            tmp = beta1 ** 2 * (np.exp(beta1 * tmpx)) * (1 - np.exp(beta1 * tmpx)) / np.power(
                (1 + np.exp(beta1 * tmpx)), 3)
            tmp += beta2 ** 2 * (np.exp(beta2 * (tmpx - 0.5))) * (1 - np.exp(beta2 * (tmpx - 0.5))) / np.power(
                (1 + np.exp(beta2 * (tmpx - 0.5))), 3)
            for i in range(0, len(tmpx) - 1):
                t = int(np.ceil(resolution_factor * max(np.abs(tmp[i:i + 2])) / float(initial_divisions)))
                x0 = np.append(x0, np.linspace(tmpx[i], tmpx[i + 1], t))
            x0 = np.sort(np.unique(x0))  # sort and remove dups
            # half brace using sum of two logistic functions
            y0 = mid * 2 * ((1 / (1. + np.exp(-1 * beta1 * x0))) - 0.5)
            y0 += (1 - mid) * 2 * (1 / (1. + np.exp(-1 * beta2 * (x0 - 0.5))))
            # concat and scale x
            x = np.concatenate((x0, 1 - x0[::-1])) * float((x_max - x_min)) + x_min
            y = np.concatenate((y0, y0[::-1])) * float(height)
            return (x, y)

        fig = plt.figure(figsize=(18, 9))
        grid_shape = (39, 37)

        axes_list = ['ax_n1', 'ax_n2', 'ax_predicted_negative', 'ax_predicted_positive', 'ax_actual_negative',
                     'ax_actual_positive',
                     'ax_true_negative', 'ax_false_positive', 'ax_false_negative', 'ax_true_positive',
                     'ax_total_predicted_negative', 'ax_total_predicted_positive', 'ax_total_actual_negative',
                     'ax_total_actual_positive', 'ax_precision', 'ax_negative_predictive_value', 'ax_accuracy', 'ax_f1',
                     'ax_specificity', 'ax_sensitivity', 'ax_predicted_class', 'ax_actual_class', 'ax_top_brace',
                     'ax_left_brace']
        typex = ['coordinates', 'spans', 'facecolor', 'spines']
        coordinates = [[(6, 3), (2, 4), 'w', False], [(24, 23), (4, 5), 'lavender', True],
                       [(6, 7), (2, 8), 'pink', True],
                       [(6, 15), (2, 8), 'pink', True], [(8, 3), (8, 4), 'pink', True], [(16, 3), (8, 4), 'pink', True],
                       [(8, 7), (8, 8), 'w', True], [(8, 15), (8, 8), 'w', True], [(16, 7), (8, 8), 'w', True],
                       [(16, 15), (8, 8), 'w', True], [(24, 7), (4, 8), 'lavender', True],
                       [(24, 15), (4, 8), 'lavender', True], [(8, 23), (8, 5), 'lavender', True],
                       [(16, 23), (8, 5), 'lavender', True], [(30, 15), (9, 8), 'antiquewhite', True],
                       [(30, 7), (9, 8), 'antiquewhite', True], [(30, 25), (9, 6), 'antiquewhite', True],
                       [(30, 31), (9, 6), 'antiquewhite', True], [(8, 29), (8, 8), 'antiquewhite', True],
                       [(16, 29), (8, 8), 'antiquewhite', True], [(0, 7), (2, 16), 'w', False],
                       [(8, 0), (16, 1), 'w', False], [(2, 7), (3, 16), 'w', False], [(8, 1), (16, 1), 'w', False]]
        d = {}
        for ax, coordinates in zip(axes_list, coordinates):
            d[ax] = dict(zip(typex, coordinates))
        axes = {}
        for ax, v in d.items():
            loc = d[ax]['coordinates']
            rowspan, colspan = d[ax]['spans']
            facecolor = d[ax]['facecolor']
            spines = d[ax]['spines']
            axes[ax] = plt.subplot2grid(shape=grid_shape, loc=loc, rowspan=rowspan, colspan=colspan,
                                        facecolor=facecolor,
                                        xticks=[], yticks=[])
            for sp in axes[ax].spines.values():
                sp.set_visible(spines)
        # draw the top brace
        x, y = range_brace(0, 1)
        axes['ax_top_brace'].set_xlim(0, 1)
        axes['ax_top_brace'].set_ylim(0, 1)
        axes['ax_top_brace'].plot(x, y, '-', clip_on=False)
        # draw the left brace
        x, y = range_brace(0, 1)
        axes['ax_left_brace'].set_xlim(0, 1)
        axes['ax_left_brace'].set_ylim(0, 1)
        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg_around(.5, .5, 90)
        axes['ax_left_brace'].plot(x, y, '-', clip_on=False, transform=rot + base)
        typexx = ['ax', 'coord', 'text', 'align', 'fontsize', 'weight', 'color', 'path_effectsx', 'rotationx']
        text_list = [
            ['ax_n2', (.5, .7), 'Total', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_n2', (.5, .3), f'{(tn + tp + fn + fp):,.0f}', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_predicted_negative', (.5, .5), 'Negative', ('center', 'center'), 15, 'bold', 'black', False, 0],
            ['ax_predicted_positive', (.5, .5), 'Positive', ('center', 'center'), 15, 'bold', 'black', False, 0],
            ['ax_actual_negative', (.5, .5), 'Negative', ('center', 'center'), 15, 'bold', 'black', False, 0],
            ['ax_actual_positive', (.5, .5), 'Positive', ('center', 'center'), 15, 'bold', 'black', False, 0],
            ['ax_true_negative', (.5, .7), 'True Negative (TN)', ('center', 'center'), 18, 'normal', 'green', True, 0],
            ['ax_true_negative', (.5, .3), f'{tn:,.0f}', ('center', 'center'), 18, 'normal', 'green', True, 0],
            ['ax_false_positive', (.5, .8), 'False Positive (FP))', ('center', 'center'), 18, 'normal', 'red', True, 0],
            ['ax_false_positive', (.5, .55), '(Type I Error)', ('center', 'center'), 16, 'normal', 'darkred', False, 0],
            ['ax_false_positive', (.5, .3), f'{fp:,.0f}', ('center', 'center'), 20, 'normal', 'red', False, 0],
            ['ax_false_negative', (.5, .8), 'False Negative (FN)', ('center', 'center'), 18, 'normal', 'red', True, 0],
            ['ax_false_negative', (.5, .55), '(Type II Error)', ('center', 'center'), 16, 'normal', 'darkred', False, 0],
            ['ax_false_negative', (.5, .3), f'{fn:,.0f}', ('center', 'center'), 20, 'normal', 'red', False, 0],
            ['ax_true_positive', (.5, .7), 'True Positive (TP)', ('center', 'center'), 18, 'normal', 'green', True, 0],
            ['ax_true_positive', (.5, .3), f'{tp:,.0f}', ('center', 'center'), 20, 'normal', 'green', False, 0],
            ['ax_total_predicted_negative', (.5, .7), 'Total Predicted Negative', ('center', 'center'), 14, 'normal', 'black', False, 0],
            ['ax_total_predicted_negative', (.5, .3), f'{(tn + fn):,.0f}', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_total_predicted_positive', (.5, .7), 'Total Predicted Positive', ('center', 'center'), 14, 'normal', 'black', False, 0],
            ['ax_total_predicted_positive', (.5, .3), f'{(tp + fp):,.0f}', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_total_actual_negative', (.5, .7), 'Total\nActual Negative', ('center', 'center'), 14, 'normal', 'black', False, 0],
            ['ax_total_actual_negative', (.5, .3), f'{(tn + fp):,.0f}', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_total_actual_positive', (.5, .7), 'Total\nActual Positive', ('center', 'center'), 14, 'normal', 'black', False, 0],
            ['ax_total_actual_positive', (.5, .3), f'{(tp + fn):,.0f}', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_precision', (.5, .8), 'Precision\nPositive Predictive Value', ('center', 'center'), 15, 'normal', 'black', False, 0],
            ['ax_precision', (.5, .45), r'$\frac{TP}{TP+FP}$', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_precision', (.5, .1), f'{tp / (tp + fp):.2%}', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_negative_predictive_value', (.5, .8), 'Negative Predictive Value', ('center', 'center'), 15, 'normal', 'black', False, 0],
            ['ax_negative_predictive_value', (.5, .45), r'$\frac{TN}{TN+FN}$', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_negative_predictive_value', (.5, .1), f'{tn / (tn + fn):.2%}', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_accuracy', (.5, .8), 'Accuracy', ('center', 'center'), 15, 'normal', 'black', False, 0],
            ['ax_accuracy', (.5, .45), r'$\frac{TP+TN}{TP+TN+FP+FN}$', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_accuracy', (.5, .1), f'{(tn + tp) / (tn + tp + fn + fp):.2%}', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_f1', (.5, .8), 'F1 Score', ('center', 'center'), 15, 'normal', 'black', False, 0],
            ['ax_f1', (.5, .45), r'$2\left(\frac{(Precision)(Recall)}{Precision + Recall}\right)$', ('center', 'center'), 20, 'normal',
             'black', False, 0],
            ['ax_f1', (.5, .1), f'{2 * (((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn)))):.2%}',
             ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_specificity', (.5, .8), 'True Negative Rate (TNR)\nSpecificity', ('center', 'center'), 15, 'normal', 'black', False, 0],
            ['ax_specificity', (.5, .425), r'$\frac{TN}{TN+FP}$', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_specificity', (.5, .1), f'{tn / (tn + fp):.2%}', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_sensitivity', (.5, .8), 'True Positive Rate (TPR)\nSensitivity or Recall', ('center', 'center'), 15, 'normal', 'black',
             False, 0],
            ['ax_sensitivity', (.5, .425), r'$\frac{TP}{TP+FN}$', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_sensitivity', (.5, .1), f'{tp / (tp + fn):.2%}', ('center', 'center'), 20, 'normal', 'black', False, 0],
            ['ax_predicted_class', (.5, .5), 'Predicted Class', ('center', 'center'), 15, 'bold', 'black', False, 0],
            ['ax_actual_class', (.5, .5), 'Actual Class', ('center', 'center'), 15, 'bold', 'black', False, 90]]
        d2 = {}
        for ax, text_list in zip(range(len(text_list)), text_list):
            d2[ax] = dict(zip(typexx, text_list))
        for key, value in d2.items():
            x, y = value['coord']
            txt = value['text']
            va, ha = value['align']
            fontsize = value['fontsize']
            weight = value['weight']
            color = value['color']
            path_effectsx = value['path_effectsx']
            if path_effectsx:
                path_effects_var = [path_effects.withSimplePatchShadow()]
            else:
                path_effects_var = False
            rotationx = value['rotationx']
            axes[d2[key]['ax']].text(x=x, y=y, s=txt, va=va, ha=ha, fontsize=fontsize, weight=weight, color=color,
                                     path_effects=path_effects_var, rotation=rotationx)
        plt.suptitle('Confusion Matrix Plus', fontsize=30)
        plt.subplots_adjust(wspace=0, hspace=0)
        Fball.bl()
        plt.show()
        return fig

    @staticmethod
    def plot_model_comparison(classification_algos_name, accuracy_score_list, y_limit):
        df_models = pd.DataFrame({"algorithms": classification_algos_name,
                                  "accuracy": accuracy_score_list})
        df_models.set_index(keys='algorithms', drop=True, inplace=True)
        df_models.sort_values(by='accuracy', ascending=True, inplace=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(df_models.index, df_models['accuracy'], palette='Blues', ax=ax)  # revoved xyz
        plt.xticks(rotation=90)
        plt.ylim(y_limit)
        plt.ylabel('Accuracy Score')
        plt.grid(True)
        plt.show()
        model_best_roc = df_models.index[-1]
        print(f'Best classifier: {model_best_roc}')
        Fball.bl()
        return model_best_roc

    @staticmethod
    def get_model_info(classification_algos, model_best_roc, classification_algos_name, X_train, y_train,
                       show_feat_imp=True):
        for idx, mod in enumerate(classification_algos):
            if model_best_roc == classification_algos_name[idx]:
                model = mod
                print(f'Selecting {model_best_roc}')
                model.fit(X_train, y_train)
        if hasattr(model, 'feature_importances_'):
            has_feature_importances_ = True
            print(f'{"":<3}{model_best_roc} has features_importances_')
        else:
            has_feature_importances_ = False
            print(f'{"":<3}{model_best_roc} does NOT has features_importances_')
        if hasattr(model, 'coef_'):
            has_coef_ = True
            print(f'{"":<3}{model_best_roc} has coef_')
        else:
            has_coef_ = False
            print(f'{"":<3}{model_best_roc} does NOT has coef_')
        Fball.bl()
        print(model)
        Fball.bl()
        if has_feature_importances_ and show_feat_imp:
            Fball.feat_imp_plot(X_train, model, X_train.shape[1], max_features=30)
            plt.show()
            Fball.bl()
        # print(f'{time.strftime("%H:%M:%S", time.gmtime(int(time.time() - start)))}')
        return model

    @staticmethod
    def eval_model_valid(model, X_train, y_train, X_valid, y_valid, print_kappa=False, print_mcc=False, save_fig=None):
        start = time.time()
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, y_pred_prob)
        print(f'The baseline score (roc_auc) on the validation set is {auc:.2%}')
        y_pred_prob = y_pred_prob.reshape(-1, 1)
        y_pred_class = binarize(y_pred_prob, 0.5)
        tn, fp, fn, tp = confusion_matrix(y_valid, y_pred_class).ravel()
        # print('\n', classification_report(y_valid, y_pred_class, digits=4), '\n');
        fig = Fball.plot_cm_plus(tn=tn, fp=fp, fn=fn, tp=tp)
        if save_fig != None:
            fig.savefig(save_fig)
        Fball.bl()
        Fball.addl_measures(y_pred_prob, y_valid, print_kappa=print_kappa, print_mcc=print_mcc)
        print(f'{time.strftime("%H:%M:%S", time.gmtime(int(time.time() - start)))}')
        return y_pred_prob

    @staticmethod
    def addl_measures(pred, actual, print_kappa=False, print_mcc=False):
        if print_kappa:
            kappa_score = cohen_kappa_score(binarize(pred, .5), actual)
            print(f'The kappa score is {kappa_score:.2%}')
            Fball.bl()
        if print_mcc:
            mcc_score = matthews_corrcoef(actual, binarize(pred, .5))
            print(f'The MCC score is {mcc_score:.2%}')
            Fball.bl()
        return


    @staticmethod
    def train_valid_test(X, y, hyperopt_train_set=False):
        X_train_and_valid, X_test, y_train_and_valid, y_test = train_test_split(X, y, test_size=0.1, random_state=50, stratify=y)
        # Separate remainder into training and validation data for optimization
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_and_valid, y_train_and_valid, test_size=0.2, random_state=50,
                                                              stratify=y_train_and_valid, )
        Fball.bl()
        print(f'{"Total Data:":<28}{X.shape[0]:7,.0f}{1:6.0%}  {"X"}')
        Fball.bl()
        print(f'{"Test (holdout) data:":<28}{X_test.shape[0]:7,.0f}{.1:6.0%}  {"X_test"}')
        print(f'{"Train and Validation data:":<28}{X_train_and_valid.shape[0]:7,.0f}{.9:6.0%}  {"X_train_and_valid"}')
        Fball.bl()
        print(f'{"Train data:":<28}{X_train.shape[0]:7,.0f}{.72:6.0%}  {"X_train"}')
        print(f'{"Validation data:":<28}{X_valid.shape[0]:7,.0f}{.18:6.0%}  {"X_valid"}')
        Fball.bl()
        if hyperopt_train_set:
            Fball.bl()
            train_set = lgb.Dataset(X_train, label=y_train)  # used in lgb cross validation (hyperopt)
            print(f'Note: Created train_set for Hyperopt')
            return X_train, y_train, X_valid, y_valid, X_test, y_test, X_train_and_valid, y_train_and_valid, train_set
        else:
            return X_train, y_train, X_valid, y_valid, X_test, y_test, X_train_and_valid, y_train_and_valid

    @staticmethod
    def graph_cutoff(y_pred_prob, y_valid):
        columns = ['cutoff', 'True Negative', 'False Positive', 'False Negative', 'True Positive']
        df_cutoff = pd.DataFrame(columns=columns)
        entry = {}
        for i in range(100):
            cutoff = i / 100
            y_pred_class = binarize(y_pred_prob.reshape(-1, 1), cutoff)
            tn, fp, fn, tp = confusion_matrix(y_valid, y_pred_class).ravel()
            entry['cutoff'] = cutoff * 100
            entry['True Negative'] = tn
            entry['False Positive'] = fp
            entry['False Negative'] = fn
            entry['True Positive'] = tp
            df_cutoff = df_cutoff.append(entry, ignore_index=True)
        df_cutoff = df_cutoff.set_index('cutoff', drop=True)
        fig, ax = plt.subplots(figsize=(10, 6),
                               subplot_kw={'ylim': (0, 125), 'xlim': (0, 100), 'title': 'Confusion Matrix',
                                           'xlabel': 'Cutoff', 'ylabel': 'Frequency / Count'})
        df_cutoff[['True Positive', 'False Negative', 'True Negative', 'False Positive']].plot(ax=ax, grid=True,
                                                                                               color=['g', 'r', 'lightcoral', 'palegreen'])
        ax.xaxis.set_major_formatter(PercentFormatter())
        plt.show()
        return df_cutoff

    @staticmethod
    def exploded_cm(df_cutoff, _cutoff, y_limits):
        _tn, _fp, _fn, _tp = df_cutoff.iloc[_cutoff]
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={'ylim': (0, 181), 'ylabel': 'Frequency / Count'})
        ax = axs.flatten()
        colors = ['g', 'r', 'lightcoral', 'palegreen']
        result = [('True Positive', _tp), ('False Negative', _fn), ('True Negative', _tn), ('False Positive', _fp)]
        for i, c in enumerate(result):
            df_cutoff[[c[0]]].plot(ax=ax[i], color=colors[i], grid=True, title=str(c[0]), legend=False)
            yminz = y_limits[i][0]
            ymaxz = y_limits[i][1]
            ax[i].set_ylim(yminz, ymaxz)
            ax[i].xaxis.set_major_formatter(PercentFormatter())
            ax[i].axvline(x=_cutoff, ymin=0, ymax=((c[1] - yminz) / (ymaxz - yminz)), color='r', linestyle='--')
            ax[i].axhline(y=c[1], xmin=0, xmax=_cutoff / 100, color='r', linestyle='--')
        plt.tight_layout();

    @staticmethod
    def cm_after_cutoff(_cutoff, y_pred_prob, X_valid, y_valid, print_kappa=False, print_mcc=False, save_fig=None):
        cutoff = _cutoff / 100
        y_pred_class = binarize(y_pred_prob.reshape(-1, 1), cutoff)
        tn, fp, fn, tp = confusion_matrix(y_valid, y_pred_class).ravel()
        Fball.pred_prob_hist(y_pred_prob, bins=8, xlabel='X_label', cutoff=cutoff)
        fig = Fball.plot_cm_plus(tn=tn, fp=fp, fn=fn, tp=tp)
        if save_fig != None:
            fig.savefig(save_fig)
        if print_kappa:
            kappa_score = cohen_kappa_score(binarize(y_pred_prob, .5), y_valid)
            print(f'The kappa score is {kappa_score:.2%}')
        if print_mcc:
            mcc_score = matthews_corrcoef(y_valid, binarize(y_pred_prob, .5))
            print(f'The MCC score is {mcc_score:.2%}')
        Fball.bl()
        return cutoff

    @staticmethod
    def eval_hyperopt_on_valid(best_bayes_model, X_valid, y_valid, results, cutoff, print_kappa=False, print_mcc=False, save_fig=None):
        Fball.bl()
        preds = best_bayes_model.predict_proba(X_valid)[:, 1]
        print('The best model from Bayes optimization scores {:.2%} AUC ROC on the validation set.'.format(roc_auc_score(y_valid, preds)))
        print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))
        preds = preds.reshape(-1, 1)
        y_pred_class = binarize(preds, cutoff)
        tn, fp, fn, tp = confusion_matrix(y_valid, y_pred_class).ravel()
        fig = Fball.plot_cm_plus(tn=tn, fp=fp, fn=fn, tp=tp)
        if save_fig != None:
            fig.savefig(save_fig)
        Fball.bl()
        Fball.addl_measures(y_pred_class, y_valid, print_kappa=print_kappa, print_mcc=print_mcc)
        return

    @staticmethod
    def eval_hyperopt_on_test(best_bayes_model, X_test, y_test, results, cutoff, print_kappa=False, print_mcc=False, save_fig=None):
        Fball.bl()
        preds = best_bayes_model.predict_proba(X_test)[:, 1]
        print('The best model from Bayes optimization scores {:.2%} AUC ROC on the test set.'.format(roc_auc_score(y_test, preds)))
        print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))
        preds = preds.reshape(-1, 1)
        y_pred_class = binarize(preds, cutoff)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
        fig = Fball.plot_cm_plus(tn=tn, fp=fp, fn=fn, tp=tp)
        if save_fig != None:
            fig.savefig(save_fig)
        Fball.bl()
        Fball.addl_measures(y_pred_class, y_test, print_kappa=print_kappa, print_mcc=print_mcc)

    @staticmethod
    def balance_classes(X, y, msg="", hyperopt_train_set=False):
        if msg != "":
            print(msg)
            Fball.bl()
        print("Before OverSampling, counts of label '1': {}".format(sum(y == 1)))
        print("Before OverSampling, counts of label '0': {} \n".format(sum(y == 0)))
        sm = SMOTE(random_state=2)
        X_res, y_res = sm.fit_sample(X, y.ravel())
        print('After OverSampling, the shape of train_X: {}'.format(X_res.shape))
        print('After OverSampling, the shape of train_y: {} \n'.format(y_res.shape))
        print("After OverSampling, counts of label '1': {}".format(sum(y_res == 1)))
        print("After OverSampling, counts of label '0': {}".format(sum(y_res == 0)))
        if hyperopt_train_set:
            Fball.bl()
            train_set = lgb.Dataset(X, label=y)  # used in lgb cross validation (hyperopt)
            print(f'Note: Created train_set for Hyperopt')
            return X, y, train_set
        else:
            return X, y


# No longer used but left in for "02 - Data Wrangling" and "03 - Data Storytelling" support
    @staticmethod
    def show_images(path, feat, pcts):
        # import cv2
        # import matplotlib.pyplot as plt
        # from skimage.filters import laplace, sobel, roberts
        def get_crop(img, pct):
            x = int(img.shape[0] * ((1 - pct) / 2))
            h = int(img.shape[0] * pct)
            y = int(img.shape[1] * ((1 - pct) / 2))
            w = int(img.shape[1] * pct)
            img = img[x:x + h, y:y + w]
            return img

        for pct in pcts:
            plt.figure(figsize=(10, 10))
            x_label = ['Sharp', 'Defocused Blur', 'Motion Blur']
            for i in range(len(path)):
                if feat == None:
                    img = cv2.imread(str(path[i]))  # read image in BGR
                    img = get_crop(img, pct)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img = cv2.imread(str(path[i]), 0)  # read gray scale image
                    img = get_crop(img, pct)

                if feat == 'laplace':
                    img = laplace(img)
                elif feat == 'sobel':
                    img = sobel(img)
                elif feat == 'roberts':
                    img = roberts(img)
                elif feat == 'canny':
                    img = cv2.Canny(img, 100, 200, 3, L2gradient=True)
                elif feat == None:
                    pass
                else:
                    raise ValueError('Feature must be None, laplace, sobel or roberts')

                plt.subplot(1, 3, i + 1)
                plt.imshow(img, cmap='gray')
                plt.xticks([])
                plt.yticks([])
                xlabel_temp = f'{x_label[i]} ({pct:.0%} crop)'
                # plt.xlabel(x_label[i]+' '+str(pct))
                plt.xlabel(xlabel_temp)

            plt.tight_layout()
            plt.show()
        # print(img.shape)

    @staticmethod
    def show_5x5_images(rows, columns, title, style='default', plot='rgb', linef=True):
        if linef:
            print('')
        plt.style.use(style)
        color_dict = Fball.get_plot_data()
        fig, axs = plt.subplots(rows, columns, figsize=(18, 6 * rows))  # figsize=(w,h)

        for i, ax in enumerate(axs.flatten()):
            Fball.plot_heatmap(color_dict, plot[i], ax, title[i])
        plt.tight_layout()
        fig.subplots_adjust(wspace=.15)
        return

    @staticmethod
    def get_plot_data():
        np.random.seed(1)
        r = np.random.choice((0, 1), size=(5, 5))  # 1
        g = np.random.choice((0, 2), size=(5, 5))  # 2
        b = np.random.choice((0, 4), size=(5, 5))  # 4
        y = np.add(r, g)  # 3
        m = np.add(r, b)  # 5
        c = np.add(g, b)  # 6
        w = np.add(y, b)  # 7
        color_dict = {'r': r, 'g': g, 'b': b, 'y': y, 'm': m, 'c': c, 'w': w}
        return color_dict

    @staticmethod
    def plot_heatmap(color_dict, clr, ax, title):
        ax = ax
        data = color_dict[clr]
        title_font_size = 20
        ticklabel_font_size = 12

        ax.set_title(title, fontdict={'fontsize': title_font_size})

        label_text = {0: {'lbl': f'R G B\n0,0,0', 'cmap': 'black'},  # black
                      1: {'lbl': f'R G B\n255,0,0', 'cmap': 'red'},  # red
                      2: {'lbl': f'R G B\n0,255,0', 'cmap': 'green'},  # green
                      3: {'lbl': f'R G B\n255,255,0', 'cmap': 'yellow'},  # yellow
                      4: {'lbl': f'R G B\n0,0,255', 'cmap': 'blue'},  # blue
                      5: {'lbl': f'R G B\n255,0,255', 'cmap': 'magenta'},  # magenta
                      6: {'lbl': f'R G B\n0,255,255', 'cmap': 'cyan'},  # cyan
                      7: {'lbl': f'R G B\n255,255,255', 'cmap': 'white'}}  # white

        cmap = sorted(list(set([i for i in sorted(data.flatten())])))  # gives us a list  of all unique values for colors
        cmap = [label_text[x]['cmap'] for x in cmap]  # create an list containing colors for color map
        labels = np.array([label_text[i]['lbl'] for i in data.flatten()]).reshape(5,
                                                                                  5)  # create array with labels (RGB value) for each datapoint
        bins = sorted(list(set([i for i in sorted(data.flatten())])))  # create bins so matplot lib uses the correct colors
        data = np.digitize(data.flatten(), bins, right=True).reshape(5,
                                                                     5)  # rename the data to start with 0 and end with the number of unique elements
        sns.heatmap(data, ax=ax, annot=labels, linewidths=.5, square=True, cbar=False,
                    annot_kws={"fontsize": 10, "color": 'White', 'weight': 'bold'}, fmt='', cmap=ListedColormap(cmap));
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)  # fixes 1/2 squares at top and bottom, seaborn - matplotlib bug
        for ticklabel in (ax.get_xticklabels()):  # correct font for x labels
            ticklabel.set_fontsize(ticklabel_font_size)
        for ticklabel in (ax.get_yticklabels()):  # correct font for y labels
            ticklabel.set_fontsize(ticklabel_font_size)
        for i, val in enumerate(color_dict[clr].flatten()):  # change white text for certain color backgrounds
            if val in [3, 5, 6, 7, ]:  # yellow, magenta, cyan and white
                ax.texts[i].set_color('Black')
        return ax

    def ds_read_image(file, resize_img=True):
        import PIL
        if file.suffix.lower() == '.cr2':
            with rawpy.imread(str(file)) as raw:
                image = raw.postprocess()
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            file = Path(file.with_suffix('.tiff'))
        elif file.suffix.lower() in ['.jpg', 'jpeg']:
            image = cv2.imread(str(file))
        elif file.suffix.lower() in ['.tiff', '.tif']:
            image = cv2.imread(str(file))

        else:
            image = None
            print('file type not found: ' + str(file.suffix))
            return False, image, 0
        pct_resize = 1
        if resize_img:
            image, pct_resize = Fball.ds_image_resize(image, height=1350)
        return True, image, pct_resize

    def ds_image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image and the ratio used
        return resized, r


# face aligner
class FaceAligner:
    def __init__(self, desired_left_eye=(0.35, 0.35),
                 desired_face_width=256, desired_face_height=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        # self.predictor = predictor
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height
        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desired_face_height is None:
            self.desired_face_height = self.desired_face_width

    def align(self, image, rect, left_eye_center, right_eye_center):
        # compute the angle between the eye centroids
        d_y = right_eye_center[1] - left_eye_center[1]
        d_x = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(d_y, d_x))  # - 180
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desired_right_eye_x = 1.0 - self.desired_left_eye[0]
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((d_x ** 2) + (d_y ** 2))
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0])
        desired_dist *= self.desired_face_width
        scale = desired_dist / dist
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2, (left_eye_center[1] + right_eye_center[1]) // 2)
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
        # update the translation component of the matrix
        t_x = self.desired_face_width * 0.5
        t_y = self.desired_face_height * self.desired_left_eye[1]
        M[0, 2] += (t_x - eyes_center[0])
        M[1, 2] += (t_y - eyes_center[1])
        # apply the affine transformation
        (w, h) = (self.desired_face_width, self.desired_face_height)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
        # return the aligned face
        return output

