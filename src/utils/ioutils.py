import os
import ast
import sys
import yaml

from pathlib import Path
from os import PathLike

import constants as const


def resolve_path(path: PathLike) -> Path:
    return Path(__file__).parent.parent.parent.joinpath(path)


def ensure_directory(path: PathLike) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def create_run_directory(hparams):
    path = resolve_path(const.RUNS_DIR)               # runs directory
    run_ids = list(map(int, next(os.walk(path))[1]))  # subfolders as ints
    next_id = (max(run_ids) + 1) if run_ids else 0    # highest id plus one
    run_path = path / const.RUN_NAME(next_id)         # path for new run

    run_path.mkdir()                                     # create directory for new run
    (run_path / const.CKPT_SUBDIR).mkdir()               # create subdirectory for checkpoints
    with open(run_path / const.HPARAMS_FILE, "w") as f:  # save hyperparameter config
        yaml.dump(hparams, f)

    return run_path, next_id


def parse_cli_keyword_args():
    keyvalpairs = [x.lstrip("--").split("=") for x in sys.argv[1:]]
    return {key: parse_type(val) for key, val in keyvalpairs}


def parse_type(string):
    try:
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        return string
