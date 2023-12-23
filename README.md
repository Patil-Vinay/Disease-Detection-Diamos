# Disease-Detection-Diamos

Pear leaf disease classification on the DiaMOS Plant dataset using Siamese networks.
A pretrained MobileNetV2 or EfficientNetB0 is used as a frozen feature extractor and only a small head on top is trained.

Classes: **curl**, **healthy**, **slug**, **spot**

## How it works

There is one Siamese network per class. It takes two leaf images and outputs a score between 0 and 1 for "both images are from this class".

Both images go through the same backbone. The absolute difference of the two feature maps is flattened and passed to a Dense layer with one sigmoid output.

- backbone: MobileNetV2 or EfficientNetB0 with ImageNet weights, frozen
- input size 100x100
- loss: binary cross entropy, Adam with learning rate 1e-4, batch size 16

Training pairs come from the `anchor`, `positive` and `negative` folders of each class:

- anchor + positive (same class) gives label 1
- anchor + negative (images of the other 3 classes) gives label 0

To classify a single leaf, `predict.py` runs all 4 models. Each one compares the leaf with 10 train images of its own class and the class with the highest average score is the prediction.

## Dataset

Images are from DiaMOS Plant (Fenu and Malloci, Agronomy 2021), arranged like this:

```
King_Saud/
  Train/
    curl/     anchor/  positive/  negative/
    healthy/  anchor/  positive/  negative/
    slug/     anchor/  positive/  negative/
    spot/     anchor/  positive/  negative/
```

Curl and healthy have very few leaves in DiaMOS (54 and 43), so each of those leaves was saved as 10 augmented copies (brightness, colour, noise, rotation). Slug and spot images are separate photos.

`split_data.py` moves 20% of the **leaves** of each class to `King_Saud/Test`, keeping the same folder layout. All augmented copies of a leaf go to the same side, so the test set only has leaves the models never saw while training.

The dataset and the trained models are not in this repo (see `.gitignore`).

## Files

| File | What it does |
|------|--------------|
| `split_data.py` | makes the train/test split by leaf |
| `MobileNetV2_<Class>.ipynb` | trains and tests the Siamese network for one class with MobileNetV2 |
| `EfficientNetB0_<Class>.ipynb` | same with EfficientNetB0 |
| `predict.py` | uses the 4 models of a backbone together to classify a leaf |

Epochs: 20 for curl and healthy, 50 for slug, 30 for spot.

## How to run

```
pip install -r requirements.txt
```

1. Put the `King_Saud` folder next to the notebooks (on Colab, mount Drive and change `DATA_DIR` in the notebook).
2. Split once:
   ```
   python split_data.py
   ```
3. Run the 4 notebooks of a backbone. Each one saves its model to `models/<Backbone>_<Class>.h5`.
4. Classify a leaf or test on all held out images:
   ```
   python predict.py leaf.jpg
   python predict.py leaf.jpg --backbone EfficientNetB0
   python predict.py --test
   ```

## Results

All numbers are on the held out test leaves (seed 42 split). The test sets are small, so treat them as rough.

Pair verification (test accuracy of each notebook):

| Backbone | curl | healthy | slug | spot |
|----------|------|---------|------|------|
| MobileNetV2 | 0.947 | 0.816 | 0.804 | 0.756 |
| EfficientNetB0 | 0.926 | 0.750 | 0.859 | 0.756 |

4 class classification with `predict.py --test` (386 test images):

| Backbone | Accuracy |
|----------|----------|
| MobileNetV2 | 0.863 |
| EfficientNetB0 | 0.824 |

MobileNetV2 confusion matrix (rows = true class, columns = predicted):

|         | curl | healthy | slug | spot |
|---------|------|---------|------|------|
| curl    | 98 | 0 | 1 | 0 |
| healthy | 6 | 76 | 1 | 1 |
| slug    | 1 | 3 | 85 | 11 |
| spot    | 2 | 10 | 17 | 74 |

Most mistakes are between slug and spot, since slug damage can look like small spots.
The curl and healthy test images are 11 and 9 leaves with their augmented copies, so those rows are based on few real leaves.

## Notes

- MobileNetV2 reaches 100% train accuracy, so it overfits. A validation split with early stopping or augmentation during training would probably help.
- Only the head is trained. Unfreezing the last backbone blocks or using bigger images than 100x100 are things to try next.

## Reference

G. Fenu and F. M. Malloci, "DiaMOS Plant: A Dataset for Diagnosis and Monitoring Plant Disease", Agronomy, 2021.
