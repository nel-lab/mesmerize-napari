import os
from .algorithms import *
from .utils import make_runfile, IS_WINDOWS
import pandas as pd
import pathlib
from pathlib import Path
from typing import *
from PyQt5 import QtCore
from functools import partial
from uuid import uuid4, UUID
from subprocess import Popen


# Start of Core Utilities
CURRENT_BATCH_PATH: pathlib.Path = None  # only one batch at a time for now
CURRENT_DATA_PATH: pathlib.Path = None


ALGO_MODULES = \
    {
        'cnmf': cnmf,
        'mcorr': mcorr,
        'cnmfe': cnmfe,
    }


DATAFRAME_COLUMNS = ['algo', 'name', 'input_movie_path', 'params', 'outputs', 'uuid']


def load_batch(batch_file: Union[str, pathlib.Path], input_data_path: Union[str, pathlib.Path]) -> pd.DataFrame:
    global CURRENT_BATCH_PATH
    global CURRENT_DATA_PATH

    df = pd.read_pickle(
        pathlib.Path(batch_file)
    )

    CURRENT_BATCH_PATH = pathlib.Path(batch_file)

    df.caiman.path = batch_file

    return df


def _get_item_uuid(item: Union[int, str, UUID]) -> UUID:
    pass


def create_batch(path: str = None):
    if pathlib.Path(path).is_file():
        raise FileExistsError(
            f'Batch file already exists at specified location: {path}'
        )

    df = pd.DataFrame(columns=DATAFRAME_COLUMNS)
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

    def add_item(self, algo: str, name: str, input_movie_path: str, params: dict):
        """
        Add an item to the DataFrame to organize parameters
        that can be used to run a CaImAn algorithm

        Parameters
        ----------
        algo: str
            Name of the algorithm to run, see `ALGO_MODULES` dict

        name: str
            User set name for the batch item

        input_movie_path: str
            Full path to the input movie

        params:
            Parameters for running the algorithm with the input movie

        """

        global CURRENT_DATA_PATH
        if CURRENT_DATA_PATH is not None:
            input_movie_path = Path(input_movie_path).relative_to(CURRENT_DATA_PATH).as_posix()

        # Create a pandas Series (Row) with the provided arguments
        s = pd.Series(
            {
                'algo': algo,
                'name': name,
                'input_movie_path': input_movie_path,
                'params': params,
                'outputs': None,  # to store dict of output information, such as output file paths
                'uuid': str(uuid4())  # unique identifier for this combination of movie + params
            }
        )

        # Add the Series to the DataFrame
        self._df.loc[self._df.index.size] = s

        # Save DataFrame to disk
        self._df.to_pickle(self.path)

    def remove_item(self, index):
        # Drop selected index
        self._df.drop([index], inplace=True)
        # Reset indeces so there are no 'jumps'
        self._df.reset_index(drop=True, inplace=True)
        # Save new df to disc
        self._df.to_pickle(self.path)


@pd.api.extensions.register_series_accessor("caiman")
class CaimanSeriesExtensions:
    """
    Extensions for caiman stuff
    """
    def __init__(self, s: pd.Series):
        self._series = s
        self.process: QtCore.QProcess = None

    def _run_qprocess(self):
        pass

    def _run_subprocess(self):
        parent_path = Path(self._series.input_movie_path).parent

        # Create the runfile in the same dir using this Series' UUID as the filename
        runfile_path = str(parent_path.joinpath(self._series['uuid'] + '.runfile'))

        # make the runfile
        runfile = make_runfile(
            module_path=os.path.abspath(ALGO_MODULES[self._series['algo']].__file__), # caiman algorithm
            filename=runfile_path,  # path to create runfile
            args_str=f'{CURRENT_BATCH_PATH} {self._series.uuid}'  # batch file path (which contains the params) and UUID are passed as args
        )

        self.process = Popen(runfile, cwd=os.path.dirname(self._series.input_movie_path))
        self.process.wait()

    def submit_slurm(self):
        parent_path = Path(self._series.input_movie_path).parent

        # Create the runfile in the same dir using this Series' UUID as the filename
        runfile_path = str(parent_path.joinpath(self._series['uuid'] + '.runfile'))

        if CURRENT_DATA_PATH is not None:
            args_str = f'{CURRENT_BATCH_PATH} {self._series.uuid} {CURRENT_DATA_PATH}'
        else:
            f'{CURRENT_BATCH_PATH} {self._series.uuid}'

        # make the runfile
        runfile = make_runfile(
            module_path=os.path.abspath(ALGO_MODULES[self._series['algo']].__file__), # caiman algorithm
            filename=runfile_path,  # path to create runfile
            args_str=args_str  # batch file path (which contains the params) and UUID are passed as args
        )

        submission_command = f'sbatch --ntasks=1 --cpus-per-task=16 --mem=90000 --wrap="{runfile}"'

        Popen(submission_command.split(' '))

    def run(
            self, callbacks_finished: List[callable],
            callback_std_out: Optional[callable] = None
    ):
        """--cpus-per-task=16 --cpus-per-task=16
        Run a CaImAn algorithm in an external process.

        NoRMCorre, CNMF, or CNMFE will be run for this Series.
        Each Series (DataFrame row) has a `input_movie_path` and `params` for the algorithm

        Parameters
        ----------
        callbacks_finished: List[callable]
            List of callback functions that are called when the external process has finished.

        callback_std_out: Optional[callable]
            callback function to pipe the stdout
        """
        # Create a QProcess
        self.process = QtCore.QProcess()
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        # Set the callback function to read the stdout
        if callback_std_out is not None:
            self.process.readyReadStandardOutput.connect(partial(callback_std_out, self.process))

        # connect the callback functions for when the process finishes
        for f in callbacks_finished:
            self.process.finished.connect(f)

        # Get the dir that contains the input movie
        parent_path = Path(self._series.input_movie_path).parent

        # Create the runfile in the same dir using this Series' UUID as the filename
        runfile_path = str(parent_path.joinpath(self._series['uuid'] + '.runfile'))

        # make the runfile
        runfile = make_runfile(
            module_path=os.path.abspath(ALGO_MODULES[self._series['algo']].__file__), # caiman algorithm
            filename=runfile_path,  # path to create runfile
            args_str=f'{CURRENT_BATCH_PATH} {self._series.uuid}'  # batch file path (which contains the params) and UUID are passed as args
        )

        # Set working dir for the external process
        self.process.setWorkingDirectory(os.path.dirname(self._series.input_movie_path))

        # Start the external process
        if IS_WINDOWS:
            self.process.start('powershell.exe', [runfile])
        else:
            print(runfile)
            self.process.start(runfile)

        return self.process
