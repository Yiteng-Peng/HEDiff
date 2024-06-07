# HEDiff

## Paper

**Testing and Understanding Deviation Behaviors in HE-hardened Machine Learning Models**

## Run

1. Train models with `plain_models.py` and `plain_models_tf.py`.
   If you want to use HElayers, you need to convert torch model into onnx with `fhe_onnx_convert` first.
2. Run `diff_{name}.py` for Margin-guided Differential Testing. `name`: 'tenseal' for TenSEAL, 'zama' for Concrete-ML, 'helayer' for HElayers.
3. If needed, run other files for the statistic information about deviation inputs in FHE-ML models.

## Datasets List

MNIST: http://yann.lecun.com/exdb/mnist/

DIGITS: https://scikit-learn.org/stable/auto_examples/datasets/plot_digits_last_image.html

Credit: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

Bank: https://archive.ics.uci.edu/dataset/222/bank+marketing

Code for data processing is in `./dataset`
