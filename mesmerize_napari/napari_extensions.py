from mesmerize_core import CaimanSeriesExtensions
from functools import partial
from typing import Union, List, Optional
from subprocess import Popen
import pandas as pd
from mesmerize_core.batch_utils import (
    HAS_PYQT,
)
from mesmerize_core.utils import IS_WINDOWS

if HAS_PYQT:
    from PyQt5 import QtCore

@pd.api.extensions.register_series_accessor("caiman_napari")
class CaimanNapariSeriesExtensions(CaimanSeriesExtensions):
    """
    Extensions specifically for caiman-related functions in napari
    """
    def __init__(self, s: pd.Series):
        self._series = s
        self.process: [Union, QtCore.QProcess, Popen] = None

    def _run_qprocess(
        self,
        runfile_path: str,
        callbacks_finished: List[callable],
        callback_std_out: Optional[callable] = None,
    ) -> QtCore.QProcess:

        # Create a QProcess
        self.process = QtCore.QProcess()
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        # Set the callback function to read the stdout
        if callback_std_out is not None:
            self.process.readyReadStandardOutput.connect(
                partial(callback_std_out, self.process)
            )

        # connect the callback functions for when the process finishes
        if callbacks_finished is not None:
            for f in callbacks_finished:
                self.process.finished.connect(f)

        parent_path = self._series.paths.resolve(self._series.input_movie_path).parent

        # Set working dir for the external process
        self.process.setWorkingDirectory(str(parent_path))

        # Start the external process
        if IS_WINDOWS:
            self.process.start("powershell.exe", [runfile_path])
        else:
            self.process.start(runfile_path)

        return self.process
