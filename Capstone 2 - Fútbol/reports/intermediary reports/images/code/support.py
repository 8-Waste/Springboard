import settings as s
import cv2
import numpy as np
import pickle
import time
import lightgbm as lgb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv
from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter, PercentFormatter, StrMethodFormatter
from sklearn.preprocessing import  binarize, scale
from sklearn.model_selection import train_test_split


from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    SCORERS,  # # sorted(SCORERS.keys())
    confusion_matrix,
    classification_report,
    plot_confusion_matrix,
    roc_curve
)


# utility functions

# save pickle
def save_obj(objs):
    for obj in objs:
        with open(s.DATA_DIR / '10_other/objects' / str(obj[1] + '.pkl'), 'wb') as f:
            pickle.dump(obj[0], f, pickle.HIGHEST_PROTOCOL)


# load pickle
def load_obj(names):
    object_return_list = []
    for name in names:
        with open(s.DATA_DIR / '10_other/objects' / str(name + '.pkl'), 'rb') as f:
            object_return_list.append(pickle.load(f))
    return object_return_list

#utility functions

# # save pickle
# def save_obj(obj, name):
#     import pickle
#     with open(DATA_DIR / '10_other/objects' / str(name + '.pkl'), 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#
# # load pickle
# def load_obj(name ):
#     import pickle
#     with open(DATA_DIR / '10_other/objects' / str(name + '.pkl'), 'rb') as f:
#         return pickle.load(f)



# print blank lines
def bl(qty=1):
    for l in range(qty):
        print()


def get_face_embedding(image):  # need to move embedder
    embedder = cv2.dnn.readNetFromTorch('../../data/10_other/models/openface_nn4.small2.v1.t7')
    faceBlob = cv2.dnn.blobFromImage(image, 1.0 / 255,
                                     (96, 96), (0, 0, 0), swapRB=True, crop=False)
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

# decorator to help with timing functions
def time_it(func):
    """Decorator function to time functions in Jupyter Notebook"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{time.strftime("%H:%M:%S", time.gmtime(int(time.time() - start)))}')
        return result
    return wrapper


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


def skin_tone(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    skin_tone_pct = np.sum(vectorized_form(img)) / (img.shape[0] * img.shape[1])
    return  skin_tone_pct

# face aligner
class FaceAligner:
    def __init__(self, desiredLeftEye=(0.35, 0.35),
        desiredFaceWidth=256, desiredFaceHeight=None):
        # store the facial landmark predictor, desired output left
        # eye position, and desired output face width + height
        # self.predictor = predictor
        self.desiredLeftEye = desiredLeftEye
        self.desiredFaceWidth = desiredFaceWidth
        self.desiredFaceHeight = desiredFaceHeight

        # if the desired face height is None, set it to be the
        # desired face width (normal behavior)
        if self.desiredFaceHeight is None:
            self.desiredFaceHeight = self.desiredFaceWidth

    def align(self, image, rect, leftEyeCenter, rightEyeCenter):
        # compute the angle between the eye centroids
        dY = rightEyeCenter[1] - leftEyeCenter[1]
        dX = rightEyeCenter[0] - leftEyeCenter[0]
        angle = np.degrees(np.arctan2(dY, dX)) #- 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= self.desiredFaceWidth
        scale = desiredDist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
            (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

        # update the translation component of the matrix
        tX = self.desiredFaceWidth * 0.5
        tY = self.desiredFaceHeight * self.desiredLeftEye[1]
        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])

        # apply the affine transformation
        (w, h) = (self.desiredFaceWidth, self.desiredFaceHeight)
        output = cv2.warpAffine(image, M, (w, h),
            flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output

# used in Blury_Faces model creation
def get_data(images, pct):
    import os
    import concurrent.futures
    from functools import partial
    import pandas as pd
    # images = [os.path.join(path, image) for image in images]
    func = partial(process_img, pct)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        features = executor.map(func, images)
    df = pd.DataFrame(features)
    df.drop(0, axis=1, inplace=True)
    return df


def process_img(pct, img):
    import cv2
    import numpy as np
    from cv2 import Canny
    from skimage.filters import laplace, sobel, roberts
    feature = []
    image_gray = cv2.imread(str(img), 0)
    image_gray = get_crop(image_gray, pct)
    lap_feat = laplace(image_gray)
    sob_feat = sobel(image_gray)
    rob_feat = roberts(image_gray)
    can_feat = Canny(image_gray, 100, 200, 3, L2gradient=True)
    feature.extend([img, lap_feat.mean(), lap_feat.var(), np.amax(lap_feat),
                    sob_feat.mean(), sob_feat.var(), np.max(sob_feat),
                    rob_feat.mean(), rob_feat.var(), np.max(rob_feat),
                    can_feat.mean(), can_feat.var(), np.max(can_feat)])
    return feature


def get_sharp_data(image_gray):
    import cv2
    import numpy as np
    from cv2 import Canny
    from skimage.filters import laplace, sobel, roberts
    feature = []
    # image_gray = cv2.imread(str(img), 0)
    # image_gray = get_crop(image_gray, pct)
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
def pred_prob_hist(y_pred_prob, bins=8, xlabel='positive', cutoff=0.5):
    plt.hist(y_pred_prob, bins=bins)
    plt.xlim(0, 1)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.axvline(x=cutoff, ymin=0, ymax=1, color='r', linewidth=3, linestyle='--')
    plt.title("Histogram of predicted probabilities")
    plt.xlabel("Predicted probability of " + xlabel)
    plt.ylabel('Frequency');
    colors = ['darkred']
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='--') for c in colors]
    labels = ['Cutoff']
    plt.legend(lines, labels)
    plt.show();
    bl()


# plot feature importances
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


# draw diag line
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


# plot roc curve
def plot_roc_curve(y_valid, y_pred_prob):
    fpr, tpr, thresholds = roc_curve(y_valid, y_pred_prob.ravel())
    optimal_cutoff = cutoff_youdens_j(fpr, tpr, thresholds)
    y_pred_class = binarize(y_pred_prob.reshape(1, -1), optimal_cutoff)
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred_class.ravel()).ravel()
    false_positive_rate = fp / (fp + tn)
    true_positive_rate = tp / (tp + fn)

    fig, ax = plt.subplots(figsize=(5, 5))
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.gca().set_aspect('equal', adjustable='box')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    plt.title('ROC curve\n', fontsize=16)
    plt.xlabel('False Positive Rate\nFP / (FP + TN)', fontsize=14)  # : (1 - Specificity)
    plt.ylabel('True Positive Rate (Recall)\nTP / (TP + FN)', fontsize=14)  # : (Recall)
    plt.grid(True)
    abline(slope=1, intercept=0)
    ax.axvline(x=false_positive_rate, ymin=0, ymax=true_positive_rate, color='r', linestyle='--')
    ax.axhline(y=true_positive_rate, xmin=0, xmax=false_positive_rate, color='r', linestyle='--')
    plt.tight_layout()
    fig.savefig('images/ROC_Curve')
    return optimal_cutoff


# plot confusion matrix plus
def plot_cm_plus(tn, fp, fn, tp):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    from matplotlib import transforms
    import numpy as np

    def range_brace(x_min, x_max, mid=0.75,
                    beta1=50.0, beta2=100.0, height=1,
                    initial_divisions=11, resolution_factor=1.5):
        # determine x0 adaptively values using second derivitive
        # could be replaced with less snazzy:
        #   x0 = NP.arange(0, 0.5, .001)
        x0 = np.array(())
        tmpx = np.linspace(0, 0.5, initial_divisions)
        tmp = beta1 ** 2 * (np.exp(beta1 * tmpx)) * (1 - np.exp(beta1 * tmpx)) / np.power((1 + np.exp(beta1 * tmpx)), 3)
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
    coordinates = [[(6, 3), (2, 4), 'w', False], [(24, 23), (4, 5), 'lavender', True], [(6, 7), (2, 8), 'pink', True],
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
        axes[ax] = plt.subplot2grid(shape=grid_shape, loc=loc, rowspan=rowspan, colspan=colspan, facecolor=facecolor,
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
        ['ax_total_predicted_negative', (.5, .7), 'Total Predicted Negative', ('center', 'center'), 14, 'normal',
         'black', False, 0],
        ['ax_total_predicted_negative', (.5, .3), f'{(tn + fn):,.0f}', ('center', 'center'), 20, 'normal', 'black',
         False, 0],
        ['ax_total_predicted_positive', (.5, .7), 'Total Predicted Positive', ('center', 'center'), 14, 'normal',
         'black', False, 0],
        ['ax_total_predicted_positive', (.5, .3), f'{(tp + fp):,.0f}', ('center', 'center'), 20, 'normal', 'black',
         False, 0],
        ['ax_total_actual_negative', (.5, .7), 'Total\nActual Negative', ('center', 'center'), 14, 'normal', 'black',
         False, 0],
        ['ax_total_actual_negative', (.5, .3), f'{(tn + fp):,.0f}', ('center', 'center'), 20, 'normal', 'black', False,
         0],
        ['ax_total_actual_positive', (.5, .7), 'Total\nActual Positive', ('center', 'center'), 14, 'normal', 'black',
         False, 0],
        ['ax_total_actual_positive', (.5, .3), f'{(tp + fn):,.0f}', ('center', 'center'), 20, 'normal', 'black', False,
         0],
        ['ax_precision', (.5, .8), 'Precision\nPositive Predictive Value', ('center', 'center'), 15, 'normal', 'black',
         False, 0],
        ['ax_precision', (.5, .45), r'$\frac{TP}{TP+FP}$', ('center', 'center'), 20, 'normal', 'black', False, 0],
        ['ax_precision', (.5, .1), f'{tp / (tp + fp):.2%}', ('center', 'center'), 20, 'normal', 'black', False, 0],
        ['ax_negative_predictive_value', (.5, .8), 'Negative Predictive Value', ('center', 'center'), 15, 'normal',
         'black', False, 0],
        ['ax_negative_predictive_value', (.5, .45), r'$\frac{TN}{TN+FN}$', ('center', 'center'), 20, 'normal', 'black',
         False, 0],
        ['ax_negative_predictive_value', (.5, .1), f'{tn / (tn + fn):.2%}', ('center', 'center'), 20, 'normal', 'black',
         False, 0],
        ['ax_accuracy', (.5, .8), 'Accuracy', ('center', 'center'), 15, 'normal', 'black', False, 0],
        ['ax_accuracy', (.5, .45), r'$\frac{TP+TN}{TP+TN+FP+FN}$', ('center', 'center'), 20, 'normal', 'black', False,
         0],
        ['ax_accuracy', (.5, .1), f'{(tn + tp) / (tn + tp + fn + fp):.2%}', ('center', 'center'), 20, 'normal', 'black',
         False, 0],
        ['ax_f1', (.5, .8), 'F1 Score', ('center', 'center'), 15, 'normal', 'black', False, 0],
        ['ax_f1', (.5, .45), r'$2\left(\frac{(Precision)(Recall)}{Precision + Recall}\right)$', ('center', 'center'),
         20, 'normal', 'black', False, 0],
        ['ax_f1', (.5, .1),
         f'{2 * (((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn)))):.2%}',
         ('center', 'center'), 20, 'normal', 'black', False, 0],
        ['ax_specificity', (.5, .8), 'True Negative Rate (TNR)\nSpecificity', ('center', 'center'), 15, 'normal',
         'black', False, 0],
        ['ax_specificity', (.5, .425), r'$\frac{TN}{TN+FP}$', ('center', 'center'), 20, 'normal', 'black', False, 0],
        ['ax_specificity', (.5, .1), f'{(tn) / (tn + fp):.2%}', ('center', 'center'), 20, 'normal', 'black', False, 0],
        ['ax_sensitivity', (.5, .8), 'True Positive Rate (TPR)\nSensitivity or Recall', ('center', 'center'), 15,
         'normal', 'black', False, 0],
        ['ax_sensitivity', (.5, .425), r'$\frac{TP}{TP+FN}$', ('center', 'center'), 20, 'normal', 'black', False, 0],
        ['ax_sensitivity', (.5, .1), f'{(tp) / (tp + fn):.2%}', ('center', 'center'), 20, 'normal', 'black', False, 0],
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
        if path_effectsx == True:
            path_effects_var = [path_effects.withSimplePatchShadow()]
        else:
            path_effects_var = False
        rotationx = value['rotationx']

        axes[d2[key]['ax']].text(x=x, y=y, s=txt, va=va, ha=ha, fontsize=fontsize, weight=weight, color=color,
                                 path_effects=path_effects_var, rotation=rotationx)

    plt.suptitle('Confusion Matrix Plus', fontsize=30)
    plt.subplots_adjust(wspace=0, hspace=0)
    print('')
    plt.show()
    return fig


def plot_model_comparison(classification_algos_name, accuracy_score_list, y_limit):
    start = time.time()
    df_models = pd.DataFrame({"algorithms": classification_algos_name,
                              "accuracy": accuracy_score_list})

    df_models.set_index(keys='algorithms', drop=True, inplace=True)
    df_models.sort_values(by='accuracy', ascending=True, inplace=True)
    fig, ax = plt.subplots(figsize=(10,5))
    xyz = sns.barplot(df_models.index, df_models['accuracy'], palette='Blues', ax=ax)
    plt.xticks(rotation=90)
    plt.ylim(y_limit)
    plt.ylabel('Accuracy Score')
    plt.grid(True)
    plt.show()
    model_best_roc = df_models.index[-1]
    print(f'Best classifier: {model_best_roc}')
    bl()
    return model_best_roc

def get_model_info(classification_algos, model_best_roc, classification_algos_name, X_train, y_train, show_feat_imp=True):
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
    bl()
    print(model)
    bl()
    if has_feature_importances_ and show_feat_imp:
        feat_imp_plot(X_train, model, X_train.shape[1], max_features=30)
        plt.show()
        bl()
    # print(f'{time.strftime("%H:%M:%S", time.gmtime(int(time.time() - start)))}')
    return model


def eval_model_valid(model, X_train, y_train, X_valid, y_valid):
    start = time.time()
    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, y_pred_prob)

    print(f'The baseline score (roc_auc) on the validation set is {auc:.2%}')

    y_pred_prob = y_pred_prob.reshape(-1,1)
    y_pred_class = binarize(y_pred_prob, 0.5)
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred_class).ravel()
    # print('\n', classification_report(y_valid, y_pred_class, digits=4), '\n');
    _ = plot_cm_plus(tn=tn, fp=fp, fn=fn, tp=tp)
    print(f'{time.strftime("%H:%M:%S", time.gmtime(int(time.time() - start)))}')
    return y_pred_prob


def train_valid_test(X, y):

    X_train_and_valid, X_test, y_train_and_valid, y_test = train_test_split(
        X, y, test_size=0.1, random_state=50, stratify=y
    )
    # Separate remainder into training and validation data for optimization
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_and_valid,
        y_train_and_valid,
        test_size=0.2,
        random_state=50,
        stratify=y_train_and_valid,
    )

    print(f'{"Total Data:":<28}{X.shape[0]:7,.0f}{1:6.0%}  {"X"}')
    bl()
    print(f'{"Test (holdout) data:":<28}{X_test.shape[0]:7,.0f}{.1:6.0%}  {"X_test"}')
    print(
        f'{"Train and Validation data:":<28}{X_train_and_valid.shape[0]:7,.0f}{.9:6.0%}  {"X_train_and_valid"}'
    )
    bl()
    print(f'{"Train data:":<28}{X_train.shape[0]:7,.0f}{.72:6.0%}  {"X_train"}')
    print(f'{"Validation data:":<28}{X_valid.shape[0]:7,.0f}{.18:6.0%}  {"X_valid"}')
    bl()
    train_set = lgb.Dataset(X_train, label=y_train)  # used in lgb cross validation (hyperopt)
    print(f'Note: Created train_set for Hyperopt')
    return X_train, y_train, X_valid, y_valid, X_test, y_test, X_train_and_valid, y_train_and_valid, train_set


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
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'ylim': (0, 125), 'xlim': (0, 100), 'title': 'Confusion Matrix',
                                                        'xlabel': 'Cutoff', 'ylabel': 'Frequency / Count'})
    df_cutoff[['True Positive', 'False Negative', 'True Negative', 'False Positive']].plot(ax=ax, grid=True,
                                                                                      color=['g', 'r', 'lightcoral',
                                                                                             'palegreen'])
    ax.xaxis.set_major_formatter(PercentFormatter())
    plt.show()
    return df_cutoff

def exploded_cm(df_cutoff, _cutoff, y_limits):
    _tn, _fp, _fn, _tp = df_cutoff.iloc[_cutoff]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), subplot_kw={'ylim': (0, 181), 'ylabel': 'Frequency / Count'})
    ax = axs.flatten()
    colors = ['g', 'r', 'lightcoral', 'palegreen']
    # y_limits = [(75, 125), (0, 30), (200, 325), (0, 50)]
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

def cm_after_cutoff(_cutoff, y_pred_prob, X_valid, y_valid):
    cutoff = _cutoff / 100
    start = time.time()
    y_pred_class = binarize(y_pred_prob.reshape(-1, 1), cutoff)
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred_class).ravel()
    pred_prob_hist(y_pred_prob, bins=8, xlabel='X_label', cutoff=cutoff)
    _ = plot_cm_plus(tn=tn, fp=fp, fn=fn, tp=tp)
    return cutoff

def eval_hyperopt_on_valid(best_bayes_model, X_valid, y_valid, results, cutoff):
    bl()
    preds = best_bayes_model.predict_proba(X_valid)[:, 1]
    print('The best model from Bayes optimization scores {:.2%} AUC ROC on the validation set.'.format(
        roc_auc_score(y_valid, preds)))
    print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))
    preds = preds.reshape(-1, 1)
    y_pred_class = binarize(preds, cutoff)
    tn, fp, fn, tp = confusion_matrix(y_valid, y_pred_class).ravel()
    print('\n', classification_report(y_valid, y_pred_class, digits=4), '\n');
    _ = plot_cm_plus(tn=tn, fp=fp, fn=fn, tp=tp)
    return

def eval_hyperopt_on_test(best_bayes_model, X_test, y_test, results, cutoff):
    bl()
    preds = best_bayes_model.predict_proba(X_test)[:, 1]
    print('The best model from Bayes optimization scores {:.2%} AUC ROC on the test set.'.format(
        roc_auc_score(y_test, preds)))
    print('This was achieved after {} search iterations'.format(results.loc[0, 'iteration']))
    preds = preds.reshape(-1, 1)
    y_pred_class = binarize(preds, cutoff)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
    print('\n', classification_report(y_test, y_pred_class, digits=4), '\n');
    fig = plot_cm_plus(tn=tn, fp=fp, fn=fn, tp=tp)
    fig.savefig('images/Optimize_roc_auc')
