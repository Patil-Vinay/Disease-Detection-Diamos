# Moves a held out set of leaves from King_Saud/Train to King_Saud/Test.
# Same layout is kept: <class>/anchor, <class>/positive, <class>/negative.
# Run once before training: python split_data.py
import os
import random
import shutil
import argparse

CLASSES = ['curl', 'healthy', 'slug', 'spot']

# curl and healthy have 10 augmented copies of each leaf numbered in a row
# (curl_4..curl_13 is one leaf, healthy_2..healthy_11 is one leaf).
# They have to stay on the same side of the split.
FIRST_NUM = {'curl': 4, 'healthy': 2}


def leaf_id(name):
    # curl_57.jpg -> ('curl', 5), slug_184(1).jpg -> ('slug', 184)
    cls, num = os.path.splitext(name)[0].replace('(1)', '').split('_')
    num = int(num)
    if cls in FIRST_NUM:
        return cls, (num - FIRST_NUM[cls]) // 10
    return cls, num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='King_Saud')
    parser.add_argument('--test', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_dir = os.path.join(args.data, 'Train')
    test_dir = os.path.join(args.data, 'Test')
    if os.path.exists(test_dir):
        print(test_dir, 'already exists, delete it first to split again')
        return

    # all leaves of each class, from every folder they show up in
    leaves = {c: set() for c in CLASSES}
    for folder in CLASSES:
        for role in ['anchor', 'positive', 'negative']:
            for name in os.listdir(os.path.join(train_dir, folder, role)):
                if name.endswith('.jpg'):
                    cls, leaf = leaf_id(name)
                    leaves[cls].add(leaf)

    random.seed(args.seed)
    held_out = set()
    for cls in CLASSES:
        ids = sorted(leaves[cls])
        random.shuffle(ids)
        n = round(len(ids) * args.test)
        held_out.update((cls, i) for i in ids[:n])
        print(cls, len(ids), 'leaves,', n, 'held out')

    for folder in CLASSES:
        for role in ['anchor', 'positive', 'negative']:
            src = os.path.join(train_dir, folder, role)
            dst = os.path.join(test_dir, folder, role)
            os.makedirs(dst, exist_ok=True)
            moved = 0
            for name in os.listdir(src):
                if name.endswith('.jpg') and leaf_id(name) in held_out:
                    shutil.move(os.path.join(src, name), os.path.join(dst, name))
                    moved += 1
            left = len([n for n in os.listdir(src) if n.endswith('.jpg')])
            print(f'{folder}/{role}: {left} train, {moved} test')


if __name__ == '__main__':
    main()
