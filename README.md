# Freddie

This is the code repository accompanying the Bachelor's thesis of _Nikolas Kirschstein_.

> Deep Active Learning with Artificial Neural Networks for Automatic Detection of Mercury’s Bow Shock and Magnetopause Crossing Signatures in NASA’s MESSENGER Magnetometer Observations

We are currently turning my thesis into a journal paper. The final version of the source code will be available on _Zenodo_ as part of the publication.

## Installation

This project is written in pure Python. To deploy it on your machine, use

```bash
pip install -e .
```

in the repository's root directory. While the development mode `-e` is not strictly necessary, we recommend it since it ensures any modifications to the code take effect immediately.

## Scripts

The project's three main scripts are located in the `src/` directory:

- `preprocessing.py`: Data preprocessing as described in the thesis.
- `ordinary_training.py`: Standard passive training procedure.
- `active_learning.py`: Implementation of the active learning algorithm devised in the thesis.

## Configuration

The `config/` directory holds the configuration files to parametrize the training scripts:

- `hyperparams.yaml`: Hyperparameters affecting the task, models and training process.
- `techparams.yaml`: Technical parameters configuring the training environment only.

## Inference Plots

As mentioned in the thesis, the plots for model inference on the entire test set reside in this repository under the `inference_plots/` folder.

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
│   ├── active_learning.py                # active learning algorithm from the thesis
│   ├── ordinary_training.py              # standard passive learning procedure
│   └── preprocessing.py                  # data preprocessing as described in the thesis
├── .gitignore                            # files to exclude from git tracking
├── LICENSE.txt                           # the full license which this project employs
├── README.md                             # this README file :)
├── requirements.txt                      # required Python packages for this project
├── setup.cfg                             # setup configuration when installing the project as package
└── setup.py                              # setup script called by pip install
```