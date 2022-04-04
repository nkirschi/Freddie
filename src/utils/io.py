"""
This file contains utility functions for working with the file system, terminal etc.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Prototype"


import ast
import os
import yaml

from os import PathLike
from pathlib import Path
from utils import constants as const


def resolve_path(path):
    """
    Resolves the given path relative to the project root.

    Parameters
    ----------
    path: PathLike
        A path relative to the project root.

    Returns
    -------
    Path
        The resolved absolute path of the target on the file system.
    """

    return Path(__file__).parent.parent.parent.joinpath(path)


def ensure_directory(path):
    """
    Ensures that the directory resembled by the given path exists.

    Parameters
    ----------
    path: PathLike
        The path of the concerning directory.
    """

    Path(path).mkdir(parents=True, exist_ok=True)


def create_run_directory(hparams):
    """
    Sets up a new run directory and stores the given hyperparameters there.

    Parameters
    ----------
    hparams: dict
        A dictionary of hyperparameters to store in the new run directory.

    Returns
    -------
    run_path : str
        The path to the newly created directory on the file system.
    next_id : int
        The numeric ID of the new run.
    """

    path = resolve_path(const.RUNS_DIR)               # runs directory
    run_ids = list(map(int, next(os.walk(path))[1]))  # subfolders as ints
    next_id = (max(run_ids) + 1) if run_ids else 0    # highest id plus one
    run_path = path / const.RUN_NAME(next_id)         # path for new run

    run_path.mkdir()                                     # create directory for new run
    (run_path / const.CKPT_SUBDIR).mkdir()               # create subdirectory for checkpoints
    with open(run_path / const.HPARAMS_FILE, "w") as f:  # save hyperparameter config
        yaml.dump(hparams, f)

    return run_path, next_id


def parse_cli_keyword_args(args):
    """
    Parses command line keyword arguments that were given in the format '--key=value'.

    Parameters
    ----------
    args : list
        A list of command line arguments in the described format.

    Returns
    -------
    dict
        The parsed command line arguments as a dictionary.
    """

    keyvalpairs = [x.lstrip("--").split("=") for x in args]
    return {key: parse_type(val) for key, val in keyvalpairs}


def parse_type(string):
    """
    Casts the given string to the best suited type.

    Parameters
    ----------
    string : str
        The string to interpret as python object.

    Returns
    -------
        The casted version of the string or itself if no casting was applicable.
    """

    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string
