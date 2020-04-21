import settings as s
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from pathlib import Path
import pickle
import random
from utility import read_image


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


def show_5x5_images(rows, columns, title, style='default', plot='rgb', linef=True):
    if linef:
        print('')
    plt.style.use(style)
    color_dict = get_plot_data()
    fig, axs = plt.subplots(rows, columns, figsize=(18, 6 * rows))  # figsize=(w,h)

    for i, ax in enumerate(axs.flatten()):
        plot_heatmap(color_dict, plot[i], ax, title[i])
    plt.tight_layout()
    fig.subplots_adjust(wspace=.15)
    return


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


def dominate_colors(image_type='faces', k=5):
    # assert Path.cwd().name == 'code'
    random.seed(7)
    if image_type == 'faces':
        files = [p.resolve() for p in Path(s.output_jn).glob("**/*") if p.suffix.lower() in [".cr2", ".jpg"] if  #output_jn
                 p.parts[-2] == 'faces' if p.stem[-15:] != '_dominatecolors']
    elif image_type == 'images':
        files = [p.resolve() for p in Path(s.output_jn).glob("**/*") if p.suffix.lower() in [".cr2", ".jpg"] if  #output_jn
                 p.parts[-2] != 'faces' if p.stem[-16:] != '_dominatecolors']

    if k > 0: # if k is negative or 0, include all files by skipping line
        files = random.choices(population=files, k=k)

    for file in files:
        print(f'file: {file.name}')
        ret, img, pct_resize = read_image(file, False)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        skin_tone_pct = np.sum(vectorized_form(img)) / (img.shape[0] * img.shape[1])

        average = img.mean(axis=0).mean(axis=0)
        pixels = np.float32(img.reshape(-1, 3))

        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)

        dominant = palette[np.argmax(counts)]
        avg_patch = np.ones(shape=img.shape, dtype=np.uint8) * np.uint8(average)

        indices = np.argsort(counts)[::-1]
        freqs = np.cumsum(np.hstack([[0], counts[indices] / counts.sum()]))
        rows = np.int_(img.shape[0] * freqs)

        dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
        for i in range(len(rows) - 1):
            dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

        # # Setup figure and subplots
        fig = plt.figure()

        gs = fig.add_gridspec(1, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        ax1.imshow(avg_patch)
        ax1.set_title('Average color')
        ax1.axis('off')
        ax1.text(0.5, -0.1, f"RGB: ({average[0]:.0f}, {average[1]:.0f}, {average[2]:.0f})", size=8, ha="center",
                 transform=ax1.transAxes)
        ax2.imshow(dom_patch)
        ax2.set_title('Dominant colors')
        ax2.axis('off')
        ax2.text(0.5, -0.1, f"RGB: ({dominant[0]:.0f}, {dominant[1]:.0f}, {dominant[2]:.0f})", size=8, ha="center",
                 transform=ax2.transAxes)
        ax3.imshow(img)
        if image_type == 'faces':
            ax3.set_title('Face')
        else:
            ax3.set_title('Image')
        ax3.axis('off')
        ax3.text(0.5, -0.1, f"{img.shape}", size=8, ha="center", transform=ax3.transAxes)
        ax3.text(0.5, -0.2, f"{img.shape[0] * img.shape[1]:,} pixels", size=8, ha="center", transform=ax3.transAxes)
        ax3.text(0.5, -0.3, f"Skin tones: {skin_tone_pct:.2%}", size=8, ha="center", transform=ax3.transAxes)
        plt.tight_layout()
        plt.figtext(0.02, 0.02, str(file.name), horizontalalignment='left')

        plt.savefig(Path(file.parent, file.stem + '_dominatecolors' + '.jpg'), bbox_inches='tight')
        plt.show()

        entry = {'dominate_color': tuple(dominant), 'average_color': tuple(average), 'pct_skin_tone': skin_tone_pct}

        sidecar_file = file.with_suffix('.pkl')
        print(f'sidecar_file: {sidecar_file}')
        print('-'*50)

        if sidecar_file.is_file():
            with open(str(sidecar_file), 'rb') as handle:
                sidecar_dict = pickle.load(handle)
            if image_type == 'faces':
                sidecar_dict['face_colors'] = entry
            elif image_type == 'images':
                pass  # TODO need to update this
        else:
            pass # TODO this should never happen

        with open(str(sidecar_file), 'wb') as handle:
            pickle.dump(sidecar_dict, handle)


