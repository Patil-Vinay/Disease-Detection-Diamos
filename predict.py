# Uses the four siamese models together to pick a class for a leaf image.
# Each model compares the image with a few train images of its own class,
# the class whose model gives the highest average score wins.
#
#   python predict.py leaf.jpg
#   python predict.py leaf.jpg --backbone EfficientNetB0
#   python predict.py --test        (runs on every image in King_Saud/Test)
import os
import glob
import random
import argparse

import numpy as np
import tensorflow as tf
from keras.layers import Layer

CLASSES = ['curl', 'healthy', 'slug', 'spot']


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def load_image(path, backbone):
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (100, 100))
    # same scaling as in the notebooks
    if backbone == 'MobileNetV2':
        img = img / 127.5 - 1
    return img.numpy()


def class_of(path):
    return os.path.basename(path).split('_')[0]


def reference_images(data_dir, cls, n, backbone):
    # train images of this class, from its anchor folder
    files = sorted(glob.glob(os.path.join(data_dir, 'Train', cls, 'anchor', '*.jpg')))
    random.Random(0).shuffle(files)
    return np.stack([load_image(f, backbone) for f in files[:n]])


def scores(models, refs, images):
    # images: (N, 100, 100, 3), returns (N, 4) average score per class
    out = np.zeros((len(images), len(CLASSES)))
    for c, cls in enumerate(CLASSES):
        n_ref = len(refs[cls])
        a = np.repeat(images, n_ref, axis=0)
        b = np.tile(refs[cls], (len(images), 1, 1, 1))
        s = models[cls].predict([a, b], batch_size=64, verbose=0)
        out[:, c] = s.reshape(len(images), n_ref).mean(axis=1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', nargs='?')
    parser.add_argument('--backbone', default='MobileNetV2', choices=['MobileNetV2', 'EfficientNetB0'])
    parser.add_argument('--data', default='King_Saud')
    parser.add_argument('--models', default='models')
    parser.add_argument('--refs', type=int, default=10, help='train images per class to compare with')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    models = {}
    refs = {}
    for cls in CLASSES:
        path = os.path.join(args.models, '{}_{}.h5'.format(args.backbone, cls.capitalize()))
        models[cls] = tf.keras.models.load_model(path, custom_objects={'L1Dist': L1Dist}, compile=False)
        refs[cls] = reference_images(args.data, cls, args.refs, args.backbone)

    if args.image:
        s = scores(models, refs, load_image(args.image, args.backbone)[None])[0]
        for cls, v in zip(CLASSES, s):
            print('{:8s} {:.3f}'.format(cls, v))
        print('prediction:', CLASSES[int(np.argmax(s))])
        return

    if not args.test:
        parser.error('give an image or use --test')

    # every held out image once (the same file can be in more than one folder)
    files = {}
    for f in glob.glob(os.path.join(args.data, 'Test', '*', '*', '*.jpg')):
        files.setdefault(os.path.basename(f).replace('(1)', ''), f)
    files = sorted(files.values())

    images = np.stack([load_image(f, args.backbone) for f in files])
    pred = np.argmax(scores(models, refs, images), axis=1)
    true = np.array([CLASSES.index(class_of(f)) for f in files])

    print(args.backbone, 'on', len(files), 'test images')
    print('accuracy: {:.4f}'.format(np.mean(pred == true)))
    print()
    print('rows = true class, columns = predicted')
    print(' ' * 9 + ''.join('{:>9s}'.format(c) for c in CLASSES))
    for i, cls in enumerate(CLASSES):
        row = [np.sum((true == i) & (pred == j)) for j in range(len(CLASSES))]
        print('{:9s}'.format(cls) + ''.join('{:9d}'.format(v) for v in row))


if __name__ == '__main__':
    main()
