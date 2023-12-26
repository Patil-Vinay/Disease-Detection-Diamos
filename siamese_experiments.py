# Tries changes to the Siamese setup one at a time on the same 5 leaf splits as experiments.py.
# All of them use one shared model for the 4 classes on pooled (averaged) backbone features:
#   shared      - |a - b| of the pooled features into one sigmoid
#   embedding   - + a learned 128 value embedding before the distance
#   all_refs    - + compare with all train images instead of 10 (this one is in Siamese_AllClasses.ipynb)
#   top5        - + only average the 5 best matches of each class
#   hard_neg    - + negatives picked as the closest of 5 random ones
#   plain       - plain softmax classifier on the same pooled features, for comparison
# Epochs are picked on validation leaves (20% of the train leaves), then the model is
# trained again on all train leaves and tested once on the test leaves.
#
#   python siamese_experiments.py --size 100
#   python siamese_experiments.py --size 224
#   python siamese_experiments.py --size 224 --fine-tune   (EfficientNetB0 + all_refs, top layers unfrozen)
import os
import random
import argparse

import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.applications import EfficientNetB0

from experiments import list_images, load_images, backbone_features, SEEDS
from split_data import CLASSES, leaf_id, pick_leaves

VARIANTS = ['shared', 'embedding', 'all_refs', 'top5', 'hard_neg', 'plain']


class L1Dist(layers.Layer):
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def pair_model(dim, emb_size):
    a = layers.Input(shape=(dim,))
    b = layers.Input(shape=(dim,))
    emb = layers.Dense(emb_size) if emb_size else None
    ea, eb = (emb(a), emb(b)) if emb else (a, b)
    out = layers.Dense(1, activation='sigmoid')(L1Dist()(ea, eb))
    model = models.Model([a, b], out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='binary_crossentropy')
    return model, emb


def head_weights(model):
    w, b = model.layers[-1].get_weights()
    return w[:, 0], b[0]


def embed(emb, X):
    return emb(X).numpy() if emb is not None else X


def classify(E, w, b, labels, refs, query, top=None):
    # score of each query against the reference images of each class
    out = np.zeros((len(query), len(CLASSES)))
    for c in range(len(CLASSES)):
        R = E[[i for i in refs if labels[i] == c]]
        for k, q in enumerate(query):
            s = 1 / (1 + np.exp(-(np.abs(R - E[q]) @ w + b)))
            out[k, c] = np.sort(s)[-top:].mean() if top else s.mean()
    return np.argmax(out, axis=1)


class Pairs:
    def __init__(self, labels, leaves, rng):
        self.labels, self.leaves, self.rng = labels, leaves, rng

    def sample(self, idx, n, hard=None):
        # half same class (always a different leaf), half other class
        # hard = (k, E, w, b): pick the negative that scores highest out of k random ones
        by_c = {c: [i for i in idx if self.labels[i] == c] for c in range(len(CLASSES))}
        rng = self.rng
        A, B, Y = [], [], []
        for _ in range(n // 2):
            c = rng.randrange(len(CLASSES))
            a = rng.choice(by_c[c])
            p = rng.choice(by_c[c])
            while self.leaves[p] == self.leaves[a]:
                p = rng.choice(by_c[c])
            others = [k for k in range(len(CLASSES)) if k != c]
            if hard:
                k, E, w, b = hard
                cands = [rng.choice(by_c[rng.choice(others)]) for _ in range(k)]
                neg = cands[int(np.argmax(np.abs(E[cands] - E[a]) @ w + b))]
            else:
                neg = rng.choice(by_c[rng.choice(others)])
            A += [a, a]
            B += [p, neg]
            Y += [1.0, 0.0]
        return np.array(A), np.array(B), np.array(Y)


def refs_of(idx, labels, n=None):
    if n is None:
        return list(idx)
    out = []
    for c in range(len(CLASSES)):
        ids = sorted(i for i in idx if labels[i] == c)
        random.Random(0).shuffle(ids)
        out += ids[:n]
    return out


def train_siamese(X, labels, pairs, train_idx, variant, epochs, val=None):
    # with val = (fit refs, val queries) returns the val accuracy after every epoch
    emb_size = None if variant == 'shared' else 128
    n_refs = 10 if variant in ['shared', 'embedding'] else None
    top = 5 if variant in ['top5', 'hard_neg'] else None
    model, emb = pair_model(X.shape[1], emb_size)
    hist = []
    for _ in range(epochs):
        hard = None
        if variant == 'hard_neg':
            hard = (5, embed(emb, X)) + head_weights(model)
        A, B, Y = pairs.sample(train_idx, 4000, hard)
        model.fit([X[A], X[B]], Y, batch_size=32, epochs=1, verbose=0)
        if val is not None:
            refs, query = val
            pred = classify(embed(emb, X), *head_weights(model), labels, refs_of(refs, labels, n_refs), query, top)
            hist.append(np.mean(pred == labels[query]))
    return model, emb, hist, n_refs, top


def train_plain(X, labels, fit, epochs, val=None):
    inp = layers.Input(shape=(X.shape[1],))
    model = models.Model(inp, layers.Dense(len(CLASSES), activation='softmax')(inp))
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    vd = (X[val], labels[val]) if val is not None else None
    h = model.fit(X[fit], labels[fit], validation_data=vd, batch_size=16, epochs=epochs, verbose=0)
    return model, h.history.get('val_accuracy')


def run_variant(variant, X, labels, pairs, train, fit, val, test):
    if variant == 'plain':
        _, hist = train_plain(X, labels, fit, 60, val)
        best = int(np.argmax(hist)) + 1
        model, _ = train_plain(X, labels, train, best)
        pred = np.argmax(model.predict(X[test], verbose=0), axis=1)
        return np.mean(pred == labels[test]), max(hist), best

    *_, hist, _, _ = train_siamese(X, labels, pairs, fit, variant, 40, (fit, val))
    best = int(np.argmax(hist)) + 1
    model, emb, _, n_refs, top = train_siamese(X, labels, pairs, train, variant, best)
    pred = classify(embed(emb, X), *head_weights(model), labels, refs_of(train, labels, n_refs), test, top)
    return np.mean(pred == labels[test]), max(hist), best


def fine_tune(images, X, labels, pairs, fit, val, test, train, size):
    # 1. embedding + head on the frozen features, best epoch on val (up to 40)
    model, emb, hist, _, _ = train_siamese(X, labels, pairs, fit, 'all_refs', 0)
    best, best_w = -1, None
    for _ in range(40):
        A, B, Y = pairs.sample(fit, 4000)
        model.fit([X[A], X[B]], Y, batch_size=32, epochs=1, verbose=0)
        acc = np.mean(classify(embed(emb, X), *head_weights(model), labels, fit, val) == labels[val])
        if acc > best:
            best, best_w = acc, model.get_weights()
    model.set_weights(best_w)

    # 2. same thing on images with the last 30 backbone layers unfrozen and augmentation
    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(size, size, 3))
    base.trainable = True
    for l in base.layers[:-30]:
        l.trainable = False
    for l in base.layers:
        if isinstance(l, layers.BatchNormalization):
            l.trainable = False
    aug = models.Sequential([layers.RandomFlip('horizontal_and_vertical'),
                             layers.RandomRotation(0.2),
                             layers.RandomBrightness(0.2, value_range=(0, 255))])
    inp = layers.Input((size, size, 3))
    emb2 = layers.Dense(128)
    embed_model = models.Model(inp, emb2(layers.GlobalAveragePooling2D()(base(aug(inp), training=False))))
    emb2.set_weights(emb.get_weights())
    a = layers.Input((size, size, 3))
    b = layers.Input((size, size, 3))
    head = layers.Dense(1, activation='sigmoid')
    full = models.Model([a, b], head(L1Dist()(embed_model(a), embed_model(b))))
    head.set_weights(model.layers[-1].get_weights())
    full.compile(optimizer=tf.keras.optimizers.Adam(2e-5), loss='binary_crossentropy')

    def embed_all(idx):
        E = np.zeros((len(images), 128), dtype='float32')
        E[idx] = embed_model.predict(images[idx], batch_size=64, verbose=0)
        return E

    # epoch 0 is the frozen model, so fine tuning only wins if val goes up
    used = sorted(set(train + test))
    E = embed_all(used)
    hist = [np.mean(classify(E, *head_weights(full), labels, fit, val) == labels[val])]
    frozen_test = np.mean(classify(E, *head_weights(full), labels, train, test) == labels[test])
    best, best_w = hist[0], full.get_weights()
    for ep in range(10):
        A, B, Y = pairs.sample(fit, 2000)
        order = np.random.default_rng(ep).permutation(len(Y))
        for s in range(0, len(Y), 32):
            k = order[s:s + 32]
            full.train_on_batch([images[A[k]], images[B[k]]], Y[k])
        E = embed_all(fit + val)
        hist.append(np.mean(classify(E, *head_weights(full), labels, fit, val) == labels[val]))
        if hist[-1] > best:
            best, best_w = hist[-1], full.get_weights()
    full.set_weights(best_w)
    E = embed_all(used)
    ft_test = np.mean(classify(E, *head_weights(full), labels, train, test) == labels[test])
    return hist, frozen_test, int(np.argmax(hist)), ft_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='King_Saud')
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--fine-tune', action='store_true')
    args = parser.parse_args()

    roles, paths = list_images(args.data)
    names = sorted(paths)
    labels = np.array([CLASSES.index(n.split('_')[0]) for n in names])
    leaves = [leaf_id(n) for n in names]
    images = load_images(names, paths, 'images_{}.npz'.format(args.size), args.size)

    lines = []

    def log(s=''):
        print(s, flush=True)
        lines.append(s)

    backbones = ['EfficientNetB0'] if args.fine_tune else ['MobileNetV2', 'EfficientNetB0']
    for backbone in backbones:
        # pooled backbone features
        X = backbone_features(backbone, images).mean(axis=(1, 2))
        results = {v: [] for v in VARIANTS}
        ft_rows = []

        for seed in SEEDS:
            held_out = pick_leaves(names, seed)
            train = [i for i in range(len(names)) if leaves[i] not in held_out]
            test = [i for i in range(len(names)) if leaves[i] in held_out]
            val_leaves = pick_leaves([names[i] for i in train], seed + 100)
            val = [i for i in train if leaves[i] in val_leaves]
            fit = [i for i in train if leaves[i] not in val_leaves]
            pairs = Pairs(labels, leaves, random.Random(seed))
            tf.random.set_seed(seed)

            if args.fine_tune:
                hist, frozen_test, best_ep, ft_test = fine_tune(images, X, labels, pairs, fit, val, test, train, args.size)
                ft_rows.append((hist[0], frozen_test, max(hist), best_ep, ft_test))
                log('seed {:2d}: val per epoch {}'.format(seed, ' '.join('{:.3f}'.format(h) for h in hist)))
                log('         frozen val {:.3f} test {:.3f} | fine tuned best epoch {} val {:.3f} test {:.3f}'.format(
                    hist[0], frozen_test, best_ep, max(hist), ft_test))
                continue

            for v in VARIANTS:
                results[v].append(run_variant(v, X, labels, pairs, train, fit, val, test))
                t, va, ep = results[v][-1]
                log('{} {}px seed {:2d} {:9s} test {:.3f}  val {:.3f}  epochs {}'.format(
                    backbone, args.size, seed, v, t, va, ep))

        log()
        if args.fine_tune:
            r = np.array(ft_rows)
            log('frozen      val {:.3f}  test {:.3f} +- {:.3f}'.format(r[:, 0].mean(), r[:, 1].mean(), r[:, 1].std()))
            log('fine tuned  val {:.3f}  test {:.3f} +- {:.3f}  best epochs {}'.format(
                r[:, 2].mean(), r[:, 4].mean(), r[:, 4].std(), [int(e) for e in r[:, 3]]))
        else:
            log('{} {}px, mean over {} splits'.format(backbone, args.size, len(SEEDS)))
            for v in VARIANTS:
                t = np.array([x[0] for x in results[v]])
                va = np.array([x[1] for x in results[v]])
                log('  {:9s} test {:.3f} +- {:.3f}   val {:.3f}'.format(v, t.mean(), t.std(), va.mean()))
        log()

    out = 'results/siamese_finetune.txt' if args.fine_tune else 'results/siamese_{}px.txt'.format(args.size)
    os.makedirs('results', exist_ok=True)
    with open(out, 'w', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
