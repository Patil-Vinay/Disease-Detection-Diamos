# Repeats the experiment on 5 different leaf splits so the results don't depend on one split.
# For each split and backbone it trains:
#   siamese - one model per class like the notebooks, combined like predict.py
#   plain   - one softmax classifier on the same frozen backbone
# The backbone is frozen so its features are computed once and only the heads are trained.
#
#   python experiments.py               (same epochs as the notebooks)
#   python experiments.py --early-stop  (epochs picked on validation leaves)
import os
import glob
import math
import random
import argparse

import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.applications import MobileNetV2, EfficientNetB0

# pick_leaves is the same picking as split_data.py, so seed 42 is the test split of the notebooks
from split_data import CLASSES, leaf_id, pick_leaves

EPOCHS = {'curl': 20, 'healthy': 20, 'slug': 50, 'spot': 30}
PLAIN_EPOCHS = 30
MAX_EPOCHS = 100
SEEDS = [42, 1, 2, 3, 4]
BACKBONES = ['MobileNetV2', 'EfficientNetB0']


def list_images(data_dir):
    # folders as they were before split_data.py, Train and Test together
    roles = {}
    paths = {}
    for split in ['Train', 'Test']:
        for folder in CLASSES:
            for role in ['anchor', 'positive', 'negative']:
                for p in sorted(glob.glob(os.path.join(data_dir, split, folder, role, '*.jpg'))):
                    name = os.path.basename(p).replace('(1)', '')
                    roles.setdefault((folder, role), []).append(name)
                    paths.setdefault(name, p)
    return roles, paths


def load_images(names, paths, cache, size=100):
    if os.path.exists(cache):
        data = np.load(cache)
        if list(data['names']) == names:
            return data['images']
    images = []
    for n in names:
        img = tf.io.decode_jpeg(tf.io.read_file(paths[n]), channels=3)
        images.append(tf.image.resize(img, (size, size)).numpy())
    images = np.stack(images)
    np.savez(cache, names=np.array(names), images=images)
    return images


def backbone_features(backbone, images):
    size = images.shape[1]
    if backbone == 'MobileNetV2':
        base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(size, size, 3))
        images = images / 127.5 - 1
    else:
        base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(size, size, 3))
    return base.predict(images, batch_size=32, verbose=0)


class L1Dist(layers.Layer):
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def siamese_head(shape):
    a = layers.Input(shape=shape)
    b = layers.Input(shape=shape)
    x = layers.Flatten()(L1Dist()(a, b))
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model([a, b], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy')
    return model


def plain_head(shape):
    inp = layers.Input(shape=shape)
    out = layers.Dense(len(CLASSES), activation='softmax')(layers.Flatten()(inp))
    model = models.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy')
    return model


def train_head(make_model, x, y, epochs, fit=None, val=None):
    # with validation data: find the epoch with the lowest val loss first,
    # then train again on all of x for that many epochs
    if val is not None:
        model = make_model()
        stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        hist = model.fit(fit[0], fit[1], validation_data=val, batch_size=16,
                         epochs=MAX_EPOCHS, shuffle=True, callbacks=[stop], verbose=0)
        epochs = int(np.argmin(hist.history['val_loss'])) + 1
    model = make_model()
    model.fit(x, y, batch_size=16, epochs=epochs, shuffle=True, verbose=0)
    return model, epochs


def siamese_pairs(feats, index, roles, cls, keep, rng):
    anchor = [n for n in roles[(cls, 'anchor')] if keep(n)]
    positive = [n for n in roles[(cls, 'positive')] if keep(n)]
    negative = [n for n in roles[(cls, 'negative')] if keep(n)]
    rng.shuffle(anchor)
    rng.shuffle(positive)
    rng.shuffle(negative)

    # anchor + positive = 1, anchor + negative = 0 (zip cuts to the shorter list like in the notebooks)
    pairs = [(a, p, 1.0) for a, p in zip(anchor, positive)] + [(a, n, 0.0) for a, n in zip(anchor, negative)]
    xa = feats[[index[a] for a, _, _ in pairs]]
    xb = feats[[index[b] for _, b, _ in pairs]]
    y = np.array([l for _, _, l in pairs])
    return [xa, xb], y, anchor


def run_siamese(feats, index, roles, is_train, is_val, test_names, seed):
    rng = random.Random(seed)
    make_model = lambda: siamese_head(feats.shape[1:])
    scores = np.zeros((len(test_names), len(CLASSES)))
    test_f = feats[[index[n] for n in test_names]]
    used = {}

    for c, cls in enumerate(CLASSES):
        x, y, anchor = siamese_pairs(feats, index, roles, cls, is_train, rng)
        if is_val:
            fit = siamese_pairs(feats, index, roles, cls, lambda n: is_train(n) and not is_val(n), rng)[:2]
            val = siamese_pairs(feats, index, roles, cls, is_val, rng)[:2]
            model, used[cls] = train_head(make_model, x, y, None, fit, val)
        else:
            model, used[cls] = train_head(make_model, x, y, EPOCHS[cls])

        # compare with 10 train anchors of this class, like predict.py
        refs = sorted(set(anchor))
        random.Random(0).shuffle(refs)
        ref_f = feats[[index[r] for r in refs[:10]]]
        a = np.repeat(test_f, len(ref_f), axis=0)
        b = np.tile(ref_f, (len(test_f), 1, 1, 1))
        s = model.predict([a, b], batch_size=256, verbose=0)
        scores[:, c] = s.reshape(len(test_f), len(ref_f)).mean(axis=1)

    return np.argmax(scores, axis=1), used


def run_plain(feats, index, train_names, test_names, is_val):
    make_model = lambda: plain_head(feats.shape[1:])
    xy = lambda names: (feats[[index[n] for n in names]],
                        np.array([CLASSES.index(n.split('_')[0]) for n in names]))
    x, y = xy(train_names)
    if is_val:
        fit = xy([n for n in train_names if not is_val(n)])
        val = xy([n for n in train_names if is_val(n)])
        model, epochs = train_head(make_model, x, y, None, fit, val)
    else:
        model, epochs = train_head(make_model, x, y, PLAIN_EPOCHS)
    pred = np.argmax(model.predict(feats[[index[n] for n in test_names]], verbose=0), axis=1)
    return pred, epochs


def mcnemar(right_a, right_b):
    # exact two sided McNemar test on the images only one of the two got right
    b = int(np.sum(right_a & ~right_b))
    c = int(np.sum(~right_a & right_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(p, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='King_Saud')
    parser.add_argument('--cache', default='images_100.npz')
    parser.add_argument('--early-stop', action='store_true',
                        help='pick epochs with early stopping on 20%% of the train leaves')
    parser.add_argument('--out')
    args = parser.parse_args()
    if not args.out:
        args.out = 'results/experiments_early_stop.txt' if args.early_stop else 'results/experiments.txt'

    lines = []

    def log(s=''):
        print(s, flush=True)
        lines.append(s)

    roles, paths = list_images(args.data)
    names = sorted(paths)
    index = {n: i for i, n in enumerate(names)}
    images = load_images(names, paths, args.cache)
    feats = {b: backbone_features(b, images) for b in BACKBONES}

    methods = ['siamese MobileNetV2', 'siamese EfficientNetB0', 'plain MobileNetV2', 'plain EfficientNetB0']
    acc = {m: [] for m in methods}
    recall = {m: np.zeros(len(CLASSES)) for m in methods}
    epochs = {m: [] for m in methods}
    tests = []

    if args.early_stop:
        log('epochs picked by early stopping on validation leaves (patience 10, max {})'.format(MAX_EPOCHS))
    else:
        log('fixed epochs: siamese {}, plain {}'.format(EPOCHS, PLAIN_EPOCHS))
    log()

    for seed in SEEDS:
        held_out = pick_leaves(names, seed)
        is_train = lambda n: leaf_id(n) not in held_out
        train_names = [n for n in names if is_train(n)]
        test_names = [n for n in names if not is_train(n)]
        true = np.array([CLASSES.index(n.split('_')[0]) for n in test_names])

        is_val = None
        if args.early_stop:
            # validation leaves come out of the train leaves, the test leaves are not touched
            val_leaves = pick_leaves(train_names, seed + 100)
            is_val = lambda n: leaf_id(n) in val_leaves

        tf.random.set_seed(seed)
        pred = {}
        for b in BACKBONES:
            pred['siamese ' + b], used = run_siamese(feats[b], index, roles, is_train, is_val, test_names, seed)
            epochs['siamese ' + b].append(used)
            pred['plain ' + b], used = run_plain(feats[b], index, train_names, test_names, is_val)
            epochs['plain ' + b].append(used)

        right = {m: pred[m] == true for m in methods}
        for m in methods:
            acc[m].append(np.mean(right[m]))
            recall[m] += [np.mean(right[m][true == c]) for c in range(len(CLASSES))]
        tests.append((seed, right))
        log('seed {:2d} ({} test images): '.format(seed, len(test_names)) +
            '  '.join('{} {:.3f}'.format(m, acc[m][-1]) for m in methods))

    log()
    log('4 class test accuracy over {} leaf splits (mean +- std)'.format(len(SEEDS)))
    for m in methods:
        log('  {:24s} {:.3f} +- {:.3f}   (min {:.3f}, max {:.3f})'.format(
            m, np.mean(acc[m]), np.std(acc[m]), np.min(acc[m]), np.max(acc[m])))

    log()
    log('recall per class, mean over splits')
    log('  {:24s}'.format('') + ''.join('{:>9s}'.format(c) for c in CLASSES))
    for m in methods:
        log('  {:24s}'.format(m) + ''.join('{:9.3f}'.format(v / len(SEEDS)) for v in recall[m]))

    if args.early_stop:
        log()
        log('epochs used per split')
        for m in methods:
            if m.startswith('siamese'):
                for cls in CLASSES:
                    log('  {:24s} {:8s} {}'.format(m, cls, [e[cls] for e in epochs[m]]))
            else:
                log('  {:24s} {:8s} {}'.format(m, '', epochs[m]))

    log()
    log('McNemar test per split (b = only first right, c = only second right)')
    pairs = [('siamese MobileNetV2', 'siamese EfficientNetB0'),
             ('siamese MobileNetV2', 'plain MobileNetV2'),
             ('siamese EfficientNetB0', 'plain EfficientNetB0'),
             ('plain MobileNetV2', 'plain EfficientNetB0')]
    for m1, m2 in pairs:
        log('  {} vs {}'.format(m1, m2))
        for seed, right in tests:
            b, c, p = mcnemar(right[m1], right[m2])
            log('    seed {:2d}: b={:3d} c={:3d} p={:.4f}'.format(seed, b, c, p))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
