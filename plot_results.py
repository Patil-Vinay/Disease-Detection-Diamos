# Makes the figures in images/ that are used in the README.
# Needs King_Saud (after split_data.py), the results/ files and the model saved by Siamese_AllClasses.ipynb.
#
#   python plot_results.py
import os
import re
import glob
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from predict import CLASSES, L1Dist, load_image, scores

DATA = 'King_Saud'
OUT = 'images'
COLORS = {'MobileNetV2': '#2a78d6', 'EfficientNetB0': '#eb6834'}
VARIANTS = ['shared', 'embedding', 'all_refs', 'top5', 'hard_neg', 'plain']


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=110, bbox_inches='tight')
    plt.close(fig)
    print('saved', name)


def show(ax, path, title):
    ax.imshow(load_image(path))
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def plot_classes():
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9))
    for ax, cls in zip(axes, CLASSES):
        path = sorted(glob.glob(os.path.join(DATA, 'Train', cls, 'anchor', '*.jpg')))[0]
        show(ax, path, cls)
    save(fig, 'classes.png')


def plot_pairs():
    # one anchor of spot with a positive (label 1) and a negative (label 0)
    folder = os.path.join(DATA, 'Train', 'spot')
    anchor = sorted(glob.glob(os.path.join(folder, 'anchor', '*.jpg')))[0]
    positive = sorted(glob.glob(os.path.join(folder, 'positive', '*.jpg')))[0]
    negative = sorted(glob.glob(os.path.join(folder, 'negative', '*.jpg')))[0]
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.9))
    show(axes[0], anchor, 'anchor (spot)')
    show(axes[1], positive, 'positive, label 1')
    show(axes[2], anchor, 'anchor (spot)')
    show(axes[3], negative, 'negative ({}), label 0'.format(os.path.basename(negative).split('_')[0]))
    save(fig, 'pairs.png')


def test_predictions():
    model = tf.keras.models.load_model('models/siamese_EfficientNetB0.h5',
                                       custom_objects={'L1Dist': L1Dist}, compile=False)
    w, b = model.get_layer('head').get_weights()
    refs = np.load('models/refs_EfficientNetB0.npz')

    files = {}
    for f in sorted(glob.glob(os.path.join(DATA, 'Test', '*', '*', '*.jpg'))):
        files.setdefault(os.path.basename(f).replace('(1)', ''), f)
    paths = [files[n] for n in sorted(files)]

    images = np.stack([load_image(p) for p in paths])
    emb = model.get_layer('embed').predict(images, batch_size=64, verbose=0)
    pred = np.argmax(scores(emb, refs['embeddings'], refs['labels'], w[:, 0], b[0]), axis=1)
    true = np.array([CLASSES.index(os.path.basename(p).split('_')[0]) for p in paths])
    print('accuracy: {:.4f}'.format(np.mean(pred == true)))
    return paths, true, pred


def plot_confusion(true, pred):
    cm = np.array([[np.sum((true == i) & (pred == j)) for j in range(len(CLASSES))] for i in range(len(CLASSES))])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap='Blues')
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title('Shared model, {} test images, accuracy {:.3f}'.format(len(true), np.mean(true == pred)), fontsize=10)
    save(fig, 'confusion_matrix.png')


def plot_examples(paths, true, pred, idx, name):
    rows = (len(idx) + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(10, 3.1 * rows))
    for ax in axes.flat[len(idx):]:
        ax.axis('off')
    for ax, i in zip(axes.flat, idx):
        show(ax, paths[i], 'true {}\npredicted {}'.format(CLASSES[true[i]], CLASSES[pred[i]]))
    save(fig, name)


def read_means(path):
    # mean lines of siamese_experiments.py: "  all_refs  test 0.874 +- 0.013   val 0.848"
    out = {}
    backbone = None
    for line in open(path):
        m = re.match(r'(\w+) \d+px, mean over', line)
        if m:
            backbone = m.group(1)
        m = re.match(r'\s+(\w+)\s+test ([\d.]+) \+- ([\d.]+)', line)
        if m and backbone:
            out[(backbone, m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out


def plot_step_by_step():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    x = np.arange(len(VARIANTS))
    for ax, size in zip(axes, [100, 224]):
        means = read_means('results/siamese_{}px.txt'.format(size))
        for k, (bb, color) in enumerate(COLORS.items()):
            m = [means[(bb, v)][0] for v in VARIANTS]
            s = [means[(bb, v)][1] for v in VARIANTS]
            ax.errorbar(x + (k - 0.5) * 0.25, m, yerr=s, fmt='o', color=color, markersize=7,
                        capsize=3, linewidth=1.5, label=bb)
        ax.set_xticks(x, VARIANTS, rotation=20)
        ax.set_title('{}x{} images'.format(size, size), fontsize=10)
        ax.grid(axis='y', color='#e0e0e0')
        ax.set_axisbelow(True)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel('4 class test accuracy\n(mean and std, 5 splits)')
    axes[0].legend(frameon=False, loc='lower right')
    save(fig, 'step_by_step.png')


def plot_fine_tune():
    runs = []
    for line in open('results/siamese_finetune.txt'):
        m = re.match(r'seed\s+(\d+): val per epoch (.*)', line)
        if m:
            runs.append((m.group(1), [float(v) for v in m.group(2).split()]))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for seed, vals in runs:
        ax.plot(range(len(vals)), vals, color='#9ec5f4', linewidth=1.5)
    mean = np.mean([vals for _, vals in runs], axis=0)
    ax.plot(range(len(mean)), mean, color='#2a78d6', linewidth=2.5, marker='o', markersize=5, label='mean of 5 splits')
    ax.plot([], [], color='#9ec5f4', linewidth=1.5, label='one split')
    ax.set_xlabel('fine tuning epoch (0 = frozen backbone)')
    ax.set_ylabel('validation accuracy')
    ax.set_title('EfficientNetB0 224x224, top 30 layers unfrozen', fontsize=10)
    ax.grid(axis='y', color='#e0e0e0')
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False)
    save(fig, 'fine_tune.png')


def main():
    os.makedirs(OUT, exist_ok=True)
    plot_classes()
    plot_pairs()
    plot_step_by_step()
    plot_fine_tune()

    paths, true, pred = test_predictions()
    plot_confusion(true, pred)

    # 2 random right ones per class and 8 random mistakes
    rng = random.Random(0)
    right = []
    for c in range(len(CLASSES)):
        right += rng.sample([i for i in range(len(paths)) if true[i] == c and pred[i] == c], 2)
    plot_examples(paths, true, pred, right, 'test_right.png')
    wrong = [i for i in range(len(paths)) if true[i] != pred[i]]
    plot_examples(paths, true, pred, rng.sample(wrong, 8), 'test_wrong.png')


if __name__ == '__main__':
    main()
