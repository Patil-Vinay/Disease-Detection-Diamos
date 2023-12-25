# Classifies a leaf image with the shared Siamese model from Siamese_AllClasses.ipynb.
# The leaf is compared with the embeddings of all train images (saved by the notebook),
# the class with the highest average score wins.
#
#   python predict.py leaf.jpg
#   python predict.py leaf.jpg --backbone MobileNetV2
#   python predict.py --test        (runs on every image in King_Saud/Test)
import os
import glob
import argparse

import numpy as np
import tensorflow as tf
from keras.layers import Layer

CLASSES = ['curl', 'healthy', 'slug', 'spot']
SIZE = 224


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def load_image(path):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    return tf.cast(tf.round(tf.image.resize(img, (SIZE, SIZE))), tf.uint8).numpy()


def scores(query_emb, ref_emb, ref_labels, w, b):
    out = np.zeros((len(query_emb), len(CLASSES)))
    for c in range(len(CLASSES)):
        refs = ref_emb[ref_labels == c]
        for i, q in enumerate(query_emb):
            out[i, c] = (1 / (1 + np.exp(-(np.abs(refs - q) @ w + b)))).mean()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?')
    parser.add_argument('--backbone', default='EfficientNetB0', choices=['MobileNetV2', 'EfficientNetB0'])
    parser.add_argument('--data', default='King_Saud')
    parser.add_argument('--models', default='models')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    model = tf.keras.models.load_model(os.path.join(args.models, 'siamese_{}.h5'.format(args.backbone)),
                                       custom_objects={'L1Dist': L1Dist}, compile=False)
    embed = model.get_layer('embed')
    w, b = model.get_layer('head').get_weights()
    refs = np.load(os.path.join(args.models, 'refs_{}.npz'.format(args.backbone)))
    ref_emb, ref_labels = refs['embeddings'], refs['labels']

    if args.image:
        emb = embed.predict(load_image(args.image)[None], verbose=0)
        s = scores(emb, ref_emb, ref_labels, w[:, 0], b[0])[0]
        for cls, v in zip(CLASSES, s):
            print('{:8s} {:.3f}'.format(cls, v))
        print('prediction:', CLASSES[int(np.argmax(s))])
        return

    if not args.test:
        parser.error('give an image or use --test')

    # every held out image once (the same file can be in more than one folder)
    files = {}
    for f in sorted(glob.glob(os.path.join(args.data, 'Test', '*', '*', '*.jpg'))):
        files.setdefault(os.path.basename(f).replace('(1)', ''), f)
    names = sorted(files)

    images = np.stack([load_image(files[n]) for n in names])
    emb = embed.predict(images, batch_size=64, verbose=0)
    pred = np.argmax(scores(emb, ref_emb, ref_labels, w[:, 0], b[0]), axis=1)
    true = np.array([CLASSES.index(n.split('_')[0]) for n in names])

    print(args.backbone, 'on', len(names), 'test images')
    print('accuracy: {:.4f}'.format(np.mean(pred == true)))
    print()
    print('rows = true class, columns = predicted')
    print(' ' * 9 + ''.join('{:>9s}'.format(c) for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        row = [np.sum((true == i) & (pred == j)) for j in range(len(CLASSES))]
        print('{:9s}'.format(cls) + ''.join('{:9d}'.format(v) for v in row))


if __name__ == '__main__':
    main()
