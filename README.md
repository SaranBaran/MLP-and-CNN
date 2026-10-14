# HW2: CNNs

Classifying cat, deer, and dog images from CIFAR-10 with an MLP and a CNN, plus a hand-written convolution layer (`ManualConv2d`).

## How to run

From the main project folder, with the course environment active:

    conda activate csci2470
    python3 code/assignment.py

`main()` trains and tests whichever model is selected (`model = CNN(classes)` or `model = MLP(classes)`) and saves its test predictions to `predictions_cnn.npy` or `predictions_mlp.npy`.

Environment: TensorFlow 2.12.0 / Keras 2.12.0, matching the course `requirements.txt`.

## Models

**MLP** (`mlp.py`): flatten → 3 Dense layers (256 units, ReLU) → Dense output (3 logits).

**CNN** (`cnn.py`): 3 convolution blocks (3×3 kernels, 32 → 64 → 64 filters, "same" padding, stride 1), each with batch normalization and ReLU, with max pooling after the first two blocks → flatten → 2 Dense layers (320 units, ReLU), each followed by dropout (0.4) → Dense output (3 logits).

Training: Adam (learning rate 1e-3), batch size 64, 20 epochs for the CNN, inputs shuffled every epoch, random left-right flips on training batches only.

**ManualConv2d** (`manual_convolution.py`): my own convolution with stride 1 and SAME/VALID padding, written with TensorFlow operations and two loops over output positions. When `is_testing=True`, the CNN runs its third convolution layer through `ManualConv2d`, using that layer's trained weights. It passes all tests in `local_test.py`.

## Results (cat / deer / dog test set)

| Model | Test accuracy |
|---|---|
| MLP | 60% |
| CNN | 78.1% (with `ManualConv2d` in place of the third conv layer at test time) |

The CNN reached 85.8% accuracy on a 3,000-image slice of the training set, so the gap between training and test accuracy is moderate.

## Known bugs / notes

- `model.loss_list` is not filled during training, so `visualize_loss` has nothing to plot.
- Running this stencil on TensorFlow 2.16+ (Keras 3) breaks `model.loss`: Keras 3 sets its own `loss` attribute to `None` on every model, hiding `CifarModel.loss`. Using the course-pinned TensorFlow 2.12.0 fixes this.