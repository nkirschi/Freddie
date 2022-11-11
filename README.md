# Freddie

This is the code repository for the ECML paper *[Deep Active Learning for Detection of Mercury’s Bow Shock and Magnetopause Crossings](https://2022.ecmlpkdd.org/wp-content/uploads/2022/09/sub_1177.pdf)*.

## Cite

If you use the code or models in this repository for your research, please cite our paper:

```
@inproceedings{julka_deep_active_learning_mercury_2022,
  title = {Deep Active Learning for Detection of Mercury’s Bow Shock and Magnetopause Crossings},
  author = {Julka, Sahib and Kirschstein, Nikolas and Granitzer, Michael and Lavrukhin, Alexander and Amerstorfer, Ute},
  booktitle = {European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD)},
  year = {2022}
}
```

## Installation

This project is written in pure Python. To deploy it on your machine, use

```bash
pip install -e .
```

in the repository's root directory. While the development mode `-e` is not strictly necessary, we recommend it since it ensures any modifications to the code take effect immediately.

## Scripts

The project's three main scripts are located in the `src/` directory:

- `preprocessing.py`: Data preprocessing as described in the paper.
- `ordinary_training.py`: Standard passive training procedure.
- `active_learning.py`: Implementation of the active learning algorithm devised in the paper.

## Configuration

The `config/` directory holds the configuration files to parametrize the training scripts:

- `hyperparams.yaml`: Hyperparameters affecting the task, models and training process.
- `techparams.yaml`: Technical parameters configuring the training environment only.

## Model Architectures

For our study, we consider a total of six architecture categories, with corresponding generic implementations in `src/learning/models.py`. The concrete instances used in our experiments are as follows:

- **MLP:** For a baseline, we employ a simple *multi-layer perceptron* with two dense hidden layers having 128 neurons each that solely operates on the flattened window. Therefore, the MLP is completely agnostic to the structure of the input, which consists of a time dimension and a channel dimension.
- **CNN:** To leverage the spatial structure in the data, we utilise a classical *convolutional neural network (CNN)*. First, the input passes through three successive 1D convolutional layers with subsequent max-pooling. The 1D convolutions all employ "same" padding and can also be thought of as 2D convolutions with full channel width. We find that increasing the convolution kernel sizes with deeper layers works well. In our case, the succession is 3-5-7. The convolutional block's activations are then flattened and passed to a single dense layer to yield the required output shape. The dense layer is the reason for employing max-pooling, since the latter reduces the parameter number in this final layer, which however is still rather high.
- **FCNN:** Due to our sequence-to-sequence task, the vanilla CNN has most of its parameters in the final dense layer instead of the convolution filter kernels. We circumvent this issue with a *fully-connected convolutional neural network (FCNN)* that replaces the dense layer: First, a convolution with kernel size of one, being equivalent to a cheap point-wise dense layer, converts the channel size to the required flattened output size. Second, a global average pooling layer reduces each resulting channel across the time dimension to a single value. The FCNN has no choice but to learn to use different channels for different features, since different time steps within the same channel will eventually be combined. It further learns entirely through convolutional layers, which also speeds up training.
- **RNN:** So-called *recurrent neural networks (RNN)* are widely used to exploit temporal relationships that might follow a systematic rule. Hence, we add an RNN to the competition, consisting of three stacked LSTMs. Since our task definition allows the model to utilise the entire input window without restrictions, all LSTMs are bi-directional. As RNNs are suitable out of the box for sequence modeling, we require only a few additional components. Before the LSTMs, we add zero padding that extends the input series by the desired amount of additional future steps to classify. In the end, a point-wise dense layer reduces the dimensionality from the internal LSTM state size to the number of classes.
- **CRNN:** To exploit both spatial and temporal relationships in the data, we combine the convolutional components with the recurrent components into a single architecture. The hybrid combination called a *convolutional recurrent neural network (CRNN)* has demonstrated advantages in several time series tasks. In our setting, a convolutional block is followed by a recurrent stack. We dispense with the pooling layers used in the CNN and FCNN for two main reasons: first, there is no memory-related need for reducing the sequence length since RNNs are agnostic to it, and second, so as to preserve the original sequence length.
- **CANN:** Recurrent networks are known to be slow to train and, often fail to capture long-term dependencies, even with LSTMs. Hence, we try an experimental architecture, replacing the second, recurrent part of the CRNN with attention mechanisms. For  consistent terminology, we call this model a *convolutional attentional neural network (CANN)*. Loosely following the structure of a Transformer encoder, it employs several multi-head self-attention layers separated by point-wise dense layers.

## Inference Plots

As mentioned in the paper, the plots for model inference on the entire test set reside in this repository under the `inference_plots/` folder.

- `inference_plots/groundtruth/` contains the ground-truth orbit annotations.
- `inference_plots/prediction/` contains the orbit-scaled model predictions.

## Project Structure

The entire project folder structure is outlined below:

```yaml
├── config                                # training configuration files
│   ├── hyperparams.yaml                  # config for hyperparameters that affect results
│   └── techparams.yaml                   # config for technical parameters that do not affect results
├── (data)                                # MESSENGER magnetometer data (NOT included in repo!)
│   ├── (eval)                            # evaluation set orbits, created by preprocessing.py
│   │   └── ...                           # not included in repo...
│   ├── (raw)                             # raw orbits, without any preprocessing
│   │   └── ...                           # not included in repo...
│   ├── (test)                            # test set orbits, created by preprocessing.py
│   │   └── ...                           # not included in repo...
│   ├── (train)                           # training set orbits, created by preprocessing.py
│   │   └── ...                           # not included in repo...
│   ├── (classes.csv)                     # class distribution information, preprocessing.py
│   ├── (labels.csv)                      # orbit-wise bow shock and magnetopause crossing annotations
│   ├── (statistics.csv)                  # descriptive statistics, produced by preprocessing.py
│   └── (validity.csv)                    # orbit-wise validity information, produced by preprocessing.py
├── inference_plots                       # plots of the CRNN inference predictions for the entire test set
│   ├── groundtruth                       # ground-truth orbit annotations for comparison
│   │   └── ...                           # see yourself in the repo...
│   └── prediction                        # CRNN orbit inference predictions
│       └── ...                           # see yourself in the repo...
├── notebooks                             # jupyter notebooks for exploring specific aspects
│   ├── data                              # notebooks for verifying assumptions about the data
│   │   └── ...                           # see yourself in the repo...
│   ├── model                             # notebooks helpful for analyzing models
│   │   └── ...                           # see yourself in the repo...
│   └── technical                         # notebooks to analyze technical implementation details
│       └── ...                           # see yourself in the repo...
├── (runs)                                # training runs with checkpoints and results (NOT included in repo!)
│       └── ...                           # not included in repo...
├── src                                   # the main source code directory
│   ├── callbacks                         # custom callbacks for the central Fitter class
│   │   ├── best_model_callback.py        # callback tracking the best model version over all epochs
│   │   ├── checkpointing_callback.py     # callback saving a model checkpoint after each epoch 
│   │   ├── early_stopping_callback.py    # callback stopping the training process on absent improvement
│   │   ├── metric_logging_callback.py    # callback logging the evaluation metrics into local JSON files
│   │   └── wandb_callback.py             # callback logging various information to the WandB tracker
│   ├── learning                          # core functionality for the deep learning procedure
│   │   ├── datasets.py                   # abstractions of the windowed MESSENGER magnetometer dataset 
│   │   ├── fitter.py                     # central training logic, extendable by callbacks
│   │   └── models.py                     # definition of the used model architectures
│   ├── modules                           # reusable components used for modeling
│   │   ├── attentional_stack.py          # stack of attentional layers with point-wise linear layers
│   │   ├── convolutional_stack.py        # stack of convolutional layers with pooling
│   │   ├── linear_stack.py               # stack of linear/dense layers
│   │   ├── multihead_self_attention.py   # multi-head self attention layer (does not exist in PyTorch)
│   │   ├── projector.py                  # module projecting the input to one of its components
│   │   ├── recurrent_stack.py            # stack of bidirectional LSTMs          
│   │   ├── residual_block.py             # generic residual addition block
│   │   └── transposer.py                 # module transposing the input on two dimensions
│   ├── utils                             # useful functionality for various tasks 
│   │   ├── constants.py                  # important project-wide constants
│   │   ├── harry_plotter.py              # plotting functions based on matplotlib
│   │   ├── io.py                         # I/O related helper functions
│   │   ├── timer.py                      # context manager for timing source code execution
│   │   └── torchutils.py                 # handy abbreviations for common PyTorch workflows
│   ├── active_learning.py                # active learning algorithm from the paper
│   ├── ordinary_training.py              # standard passive learning procedure
│   └── preprocessing.py                  # data preprocessing as described in the paper
├── .gitignore                            # files to exclude from git tracking
├── LICENSE.txt                           # the full license which this project employs
├── README.md                             # this README file :)
├── requirements.txt                      # required Python packages for this project
├── setup.cfg                             # setup configuration when installing the project as package
└── setup.py                              # setup script called by pip install
```
