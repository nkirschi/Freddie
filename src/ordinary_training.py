"""
This is the main training script.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"

###############################################################################

from utils import training

hparams, tparams = training.load_config()
model = training.construct_model(hparams)
training.perform_train(model, hparams, tparams)
