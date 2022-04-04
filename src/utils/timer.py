"""
This file contains a class for measuring execution durations.
"""

__author__ = "Nikolas Kirschstein"
__copyright__ = "Copyright 2021, Nikolas Kirschstein, All rights reserved."
__license__ = "Apache License 2.0"
__version__ = "1.0.0"
__maintainer__ = "Nikolas Kirschstein"
__email__ = "nikolas.kirschstein@gmail.com"
__status__ = "Production"


import time


class Timer:
    """Timer for code performance measurements."""

    def __init__(self, name=None, text="Elapsed time: {:.6f} seconds"):
        """
        Initializes the timer with the given parameters.

        Parameters
        ----------
        name : str, default None
            The display name of this timer.
        text : str, default "Elapsed time: {:.6f} seconds"
            A monadic format string for reporting elapsed time.
        """

        self._time = None
        self._name = f"[{name}] " if name else ""
        self._text = text

    def start(self):
        """Starts the timer immediately."""

        if self._time is not None:
            raise ValueError("Timer is already running. You need to .stop() it first!")

        self._time = time.perf_counter()

    def stop(self):
        """Stops the timer and report elapsed time."""

        if self._time is None:
            raise ValueError("Timer is not running. You need to .start() it first!")

        elapsed = time.perf_counter() - self._time
        self._time = None
        print(self._name + self._text.format(elapsed))

    def __enter__(self):
        """Starts timer as context manager."""

        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stops context-managed timer."""

        self.stop()
