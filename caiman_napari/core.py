import pandas
import pandas as pd
import pathlib
from typing import *
import os
from PyQt5 import QtCore
from common import make_runfile, IS_WINDOWS
from functools import partial
import _cnmf, _mcorr
from uuid import uuid4, UUID
from pathlib import Path

CURRENT_BATCH_PATH: pathlib.Path = None  # only one batch at a time for now


ALGO_MODULES = \
    {
        'cnmf': _cnmf,
        'mcorr': _mcorr
    }


def load_batch(path: Union[str, pathlib.Path]) -> pd.DataFrame:
    global CURRENT_BATCH_PATH

    df = pd.read_pickle(
        pathlib.Path(path)
    )

    CURRENT_BATCH_PATH = pathlib.Path(path)

    df.caiman.path = path

    return df


def _get_item_uuid(item: Union[int, str, UUID]) -> UUID:
    pass


def create_batch(path: str = None):
    df = pandas.DataFrame(columns=['algo', 'input_movie_path', 'params', 'outputs', 'uuid'])
    df.caiman.path = path

    df.to_pickle(path)

    global CURRENT_BATCH_PATH
    CURRENT_BATCH_PATH = path

    return df


@pd.api.extensions.register_dataframe_accessor("caiman")
class CaimanDataFrameExtensions:
    """
    Extensions for caiman related functions
    """
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self.path = None

    def add_item(self, algo: str, input_movie_path: str, params: dict):
        s = pd.Series(
            {
                'algo': algo,
                'input_movie_path': input_movie_path,
                'params': params,
                'outputs': [],
                'uuid': str(uuid4())
            }
        )

        self._df.loc[self._df.index.size] = s

        self._df.to_pickle(self.path)

        print(self._df)


@pd.api.extensions.register_series_accessor("caiman")
class CaimanSeriesExtensions:
    """
    Extensions for caiman stuff
    """
    def __init__(self, s: pd.Series):
        self._series = s
        self.process: QtCore.QProcess = None

    def run(
            self, callbacks_finished: List[callable],
            callback_std_out: Optional[callable] = None
    ):
        self.process = QtCore.QProcess()
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        if callback_std_out is not None:
            self.process.readyReadStandardOutput.connect(partial(self.callback_std_out, self.process))

        for f in callbacks_finished:
            self.process.finished.connect(f)

        parent_path = Path(self._series.input_movie_path).parent
        runfile_path = str(parent_path.joinpath(self._series['uuid'] + '.runfile'))

        runfile = make_runfile(
            module_path=os.path.abspath(ALGO_MODULES[self._series['algo']].__file__),
            filename=runfile_path,
            args_str=f'{CURRENT_BATCH_PATH} {self._series.uuid}'
        )

        self.process.setWorkingDirectory(os.path.dirname(self._series.input_movie_path))

        if IS_WINDOWS:
            self.process.start('powershell.exe', [runfile])
        else:
            print(runfile)
            self.process.start(runfile)
