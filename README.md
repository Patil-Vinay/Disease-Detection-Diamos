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
| `experiments.py` | repeats everything on 5 leaf splits and compares with a plain classifier |
| `results/` | output of `experiments.py` |

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
5. Repeat on 5 leaf splits (uses Train and Test together and makes its own splits):
   ```
   python experiments.py
   python experiments.py --early-stop
   ```

## Results on one split

These are from the notebooks and `predict.py` on the seed 42 split. They come from a single training run, see the next section for how much they move.

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

## Results on 5 splits

`experiments.py` picks 5 different sets of test leaves (seed 42 is the split above plus seeds 1 to 4) and trains everything again on each one. It also trains a **plain classifier** on the same frozen backbone (Flatten and Dense with 4 softmax outputs) to see if the Siamese setup is worth it. The backbone is frozen so its features are computed once and only the heads are trained. Pairs, epochs and the 10 reference images are the same as in the notebooks and `predict.py`.

4 class test accuracy, mean and std over the 5 splits:

| Method | Fixed epochs | Early stopping |
|--------|--------------|----------------|
| Siamese MobileNetV2 | 0.810 +- 0.033 | 0.800 +- 0.018 |
| Siamese EfficientNetB0 | 0.785 +- 0.035 | 0.788 +- 0.031 |
| Plain MobileNetV2 | 0.862 +- 0.031 | 0.833 +- 0.021 |
| Plain EfficientNetB0 | 0.837 +- 0.028 | 0.830 +- 0.019 |

Fixed epochs are the notebook ones (20/20/50/30) and 30 for the plain classifier. With early stopping 20% of the train leaves are used as validation, training stops when the validation loss has not improved for 10 epochs (max 100) and the model is trained again on all train leaves for the best epoch count. The test leaves are never used for this.

Recall per class with fixed epochs, mean over splits:

| Method | curl | healthy | slug | spot |
|--------|------|---------|------|------|
| Siamese MobileNetV2 | 0.941 | 0.858 | 0.782 | 0.672 |
| Plain MobileNetV2 | 0.941 | 0.841 | 0.832 | 0.833 |

Full output with the McNemar test for every split is in `results/`.

What this shows:

- The plain classifier does as well or better than the Siamese networks in almost every split. With fixed epochs it is about 5 points higher with both backbones and the McNemar test says the difference is real (p < 0.01) in 2 of the 5 splits for MobileNetV2 and 3 of 5 for EfficientNetB0. Most of the gain is on spot.
- There is no clear winner between MobileNetV2 and EfficientNetB0. MobileNetV2 is a bit ahead on average but the gap is inside the spread between splits and with early stopping none of the per split differences are significant.
- Training the same Siamese model again on the same split changes the result a lot. On the seed 42 split `experiments.py` gets 0.813 for Siamese MobileNetV2 while the notebook run got 0.863. So the single split numbers above are on the lucky side.
- More epochs do not help. Early stopping picks very different epochs from split to split (for example 4 to 98 for curl) because the validation sets are small and the validation loss is flat. The accuracy ends up about the same as with the fixed epochs. For the plain classifier it stops after 2 to 11 epochs, when the validation loss starts going up, but the fixed 30 epochs give better accuracy.
- Spot is the hardest class. Slug and spot both show small dark marks, which are hard to see at 100x100.

## Notes

- MobileNetV2 reaches 100% train accuracy, so the heads overfit the training pairs.
- Only the head is trained. Unfreezing the last backbone blocks or using bigger images than 100x100 (mainly for spot vs slug) are things to try next.

## Reference

G. Fenu and F. M. Malloci, "DiaMOS Plant: A Dataset for Diagnosis and Monitoring Plant Disease", Agronomy, 2021.
