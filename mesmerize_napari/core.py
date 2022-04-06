import os
import numpy as np
from .algorithms import *
from .utils import make_runfile, IS_WINDOWS, MESMERIZE_LRU_CACHE, validate_path
import pandas as pd
import pathlib
from pathlib import Path
from typing import *
from PyQt5 import QtCore
from functools import partial, wraps
from uuid import uuid4, UUID
from subprocess import Popen
from functools import lru_cache
from caiman import load_memmap
from caiman.source_extraction.cnmf.cnmf import CNMF, load_CNMF
from caiman.utils.visualization import get_contours as caiman_get_contours


# Start of Core Utilities
CURRENT_BATCH_PATH: pathlib.Path = None  # only one batch at a time
PARENT_DATA_PATH: pathlib.Path = None


ALGO_MODULES = \
    {
        'cnmf': cnmf,
        'mcorr': mcorr,
        'cnmfe': cnmfe,
    }


QPROCESS_BACKEND = 'qprocess'  #: QProcess backend for use in napari
SUBPROCESS_BACKEND = 'subprocess'  #: subprocess backend, for use output a Qt application such as a notebook
SLURM_BACKEND = 'slurm'  #: SLURM backend, not yet implemented


COMPUTE_BACKENDS =\
[
    QPROCESS_BACKEND,
    SUBPROCESS_BACKEND,
    SLURM_BACKEND
]


DATAFRAME_COLUMNS = ['algo', 'name', 'input_movie_path', 'params', 'outputs', 'uuid']


def set_parent_data_path(path: Union[Path, str]) -> Path:
    """
    Set the global `PARENT_DATA_PATH`

    Parameters
    ----------
    path: Union[Path, str]
        Full parent data path
    """
    global PARENT_DATA_PATH
    path = validate_path(path)
    PARENT_DATA_PATH = Path(path)
    return PARENT_DATA_PATH


def get_parent_data_path() -> Path:
    """
    Get the global `PARENT_DATA_PATH`
    Returns
    -------
    Path
        global `PARENT_DATA_PATH`

    """
    global PARENT_DATA_PATH
    return PARENT_DATA_PATH


def load_batch(batch_file: Union[str, pathlib.Path]) -> pd.DataFrame:
    """
    Load the batch pickle file, also sets the global `CURRENT_BATCH_PATH`

    Parameters
    ----------
    batch_file: Union[str, Path])

    Returns
    -------

    """
    global CURRENT_BATCH_PATH
    batch_file = validate_path(batch_file)

    df = pd.read_pickle(
        pathlib.Path(batch_file)
    )

    CURRENT_BATCH_PATH = pathlib.Path(batch_file)

    df.caiman.path = batch_file

    return df


def _get_item_uuid(item: Union[int, str, UUID]) -> UUID:
    pass


def create_batch(path: str = None, remove_existing: bool = False) -> pd.DataFrame:
    """
    Create a new batch DataFrame

    Parameters
    ----------
    path: str
        path to save the new batch DataFrame

    remove_existing: bool
        If `True`, remove an existing batch DataFrame file if it exists at the given `path`

    Returns
    -------
    pd.DataFrame
        New empty batch DataFrame

    """
    path = validate_path(path)

    if pathlib.Path(path).is_file():
        if remove_existing:
            os.remove(path)
        else:
            raise FileExistsError(
                f'Batch file already exists at specified location: {path}'
            )

    if not Path(path).parent.is_dir():
        os.makedirs(Path(path).parent)

    df = pd.DataFrame(columns=DATAFRAME_COLUMNS)
    df.caiman.path = path

    df.to_pickle(path)

    global CURRENT_BATCH_PATH
    CURRENT_BATCH_PATH = path

    return df


def get_full_data_path(path: Union[Path, str]) -> Path:
    path = Path(path)
    if PARENT_DATA_PATH is not None:
        return PARENT_DATA_PATH.joinpath(path)

    return path


@pd.api.extensions.register_dataframe_accessor("caiman")
class CaimanDataFrameExtensions:
    """
    Extensions for caiman related functions
    """
    def __init__(self, df: pd.DataFrame):
        self._df = df
        self.path = None

    def uloc(self, u: Union[str, UUID]) -> pd.Series:
        """
        Return the series corresponding to the passed UUID
        """
        df_u = self._df.loc[self._df['uuid'] == str(u)]

        if df_u.index.size == 0:
            raise KeyError("Item with given UUID not found in dataframe")
        elif df_u.index.size > 1:
            raise KeyError(f"Duplicate items with given UUID found in dataframe, something is wrong\n"
                           f"{df_u}")

        return df_u.squeeze()

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

        global PARENT_DATA_PATH
        input_movie_path = Path(input_movie_path)
        validate_path(input_movie_path)

        if PARENT_DATA_PATH is not None:
            input_movie_path = input_movie_path.relative_to(PARENT_DATA_PATH)

        input_movie_path = str(input_movie_path)

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


def validate(algo: str = None):
    def dec(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self._series['outputs'] is None:
                raise ValueError("Item has not been run")

            if algo is not None:
                if algo not in self._series['algo']:
                    raise ValueError(f"<{algo} extension called for a <{self._series}> item")

            if not self._series['outputs']['success']:
                raise ValueError("Cannot load output of an unsuccessful item")
            return func(self, *args, **kwargs)

        return wrapper

    return dec


@pd.api.extensions.register_series_accessor("caiman")
class CaimanSeriesExtensions:
    """
    Extensions for caiman stuff
    """
    def __init__(self, s: pd.Series):
        self._series = s
        self.process: [Union, QtCore.QProcess, Popen] = None

    def _run_qprocess(
            self,
            runfile_path: str,
            callbacks_finished: List[callable],
            callback_std_out: Optional[callable] = None
    ) -> QtCore.QProcess:

        # Create a QProcess
        self.process = QtCore.QProcess()
        self.process.setProcessChannelMode(QtCore.QProcess.MergedChannels)

        # Set the callback function to read the stdout
        if callback_std_out is not None:
            self.process.readyReadStandardOutput.connect(partial(callback_std_out, self.process))

        # connect the callback functions for when the process finishes
        for f in callbacks_finished:
            self.process.finished.connect(f)

        # Set working dir for the external process
        self.process.setWorkingDirectory(os.path.dirname(self._series.input_movie_path))

        # Start the external process
        if IS_WINDOWS:
            self.process.start('powershell.exe', [runfile_path])
        else:
            self.process.start(runfile_path)

        return self.process

    def _run_subprocess(
            self,
            runfile_path: str,
            callbacks_finished: List[callable] = None,
            callback_std_out: Optional[callable] = None
    ):
        global PARENT_DATA_PATH

        # Get the dir that contains the input movie
        parent_path = get_full_data_path(Path(self._series.input_movie_path).parent)

        self.process = Popen(runfile_path, cwd=parent_path)
        return self.process

    def _run_slurm(
            self,
            runfile_path: str,
            callbacks_finished: List[callable],
            callback_std_out: Optional[callable] = None
    ):
        submission_command = f'sbatch --ntasks=1 --cpus-per-task=16 --mem=90000 --wrap="{runfile_path}"'

        Popen(submission_command.split(' '))

    def run(
            self,
            backend: str,
            callbacks_finished: List[callable],
            callback_std_out: Optional[callable] = None
    ):
        """
        Run a CaImAn algorithm in an external process using the chosen backend

        NoRMCorre, CNMF, or CNMFE will be run for this Series.
        Each Series (DataFrame row) has a `input_movie_path` and `params` for the algorithm

        Parameters
        ----------
        backend: str
            One of the available backends

        callbacks_finished: List[callable]
            List of callback functions that are called when the external process has finished.

        callback_std_out: Optional[callable]
            callback function to pipe the stdout
        """
        if backend not in COMPUTE_BACKENDS:
            raise KeyError(f'Invalid `backend`, choose from the following backends:\n'
                           f'{COMPUTE_BACKENDS}')

        global PARENT_DATA_PATH

        # Get the dir that contains the input movie
        parent_path = get_full_data_path(Path(self._series.input_movie_path).parent)

        # Create the runfile in the same dir using this Series' UUID as the filename
        runfile_path = str(parent_path.joinpath(self._series['uuid'] + '.runfile'))

        args_str = f'--batch-path {CURRENT_BATCH_PATH} --uuid {self._series.uuid}'
        if PARENT_DATA_PATH is not None:
             args_str += f' --data-path {PARENT_DATA_PATH}'

        # make the runfile
        runfile = make_runfile(
            module_path=os.path.abspath(ALGO_MODULES[self._series['algo']].__file__), # caiman algorithm
            filename=runfile_path,  # path to create runfile
            args_str=args_str
        )

        self.process = getattr(self, f"_run_{backend}")(runfile, callbacks_finished, callback_std_out)

        return self.process

    @validate()
    def get_input_movie_path(self) -> Path:
        """
        Returns
        -------
        Path
            full path to the input movie file
        """

        return get_full_data_path(self._series['input_movie_path'])

    @validate()
    def get_correlation_image(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            correlation image
        """
        path = get_full_data_path(self._series['outputs']['corr-img-path'])
        return np.load(str(path))

    @validate()
    def get_projection(self, proj_type: str):
        pass

    # TODO: finish the copy_data() extension
    # def copy_data(self, new_parent_dir: Union[Path, str]):
    #     """
    #     Copy all data associated with this series to a different parent dir
    #     """
    #     movie_path = get_full_data_path(self._series['input_movie_path'])
    #     output_paths = []
    #     for p in self._series['outputs']


@pd.api.extensions.register_series_accessor("cnmf")
class CNMFExtensions:
    """
    Extensions for managing CNMF output data
    """
    def __init__(self, s: pd.Series):
        self._series = s

    def get_cnmf_memmap(self) -> np.ndarray:
        """
        Get the CNMF memmap

        Returns
        -------
        np.ndarray
            numpy memmap array used for CNMF
        """
        path = get_full_data_path(self._series['outputs']['cnmf-memmap-path'])
        # Get order f images
        Yr, dims, T = load_memmap(str(path))
        images = np.reshape(Yr.T, [T] + list(dims), order='F')
        return images

    def get_input_memmap(self) -> np.ndarray:
        """
        Return the F-order memmap if the input to the
        CNMF batch item was a mcorr output memmap

        Returns
        -------
        np.ndarray
            numpy memmap array of the input
        """
        movie_path = str(self._series.caiman.get_input_movie_path())
        if movie_path.endswith('mmap'):
            Yr, dims, T = load_memmap(movie_path)
            images = np.reshape(Yr.T, [T] + list(dims), order='F')
            return images
        else:
            raise TypeError(f"Input movie for CNMF was not a memmap, path to input movie is:\n"
                            f"{movie_path}")

    # TODO: Cache this globally so that a common upper cache limit is valid for ALL batch items
    @validate('cnmf')
    def get_output_path(self) -> Path:
        """
        Returns
        -------
        Path
            Path to the Caiman CNMF hdf5 output file
        """
        return get_full_data_path(self._series['outputs']['cnmf-hdf5-path'])

    #@lru_cache(MESMERIZE_LRU_CACHE)
    @validate('cnmf')
    def get_output(self) -> CNMF:
        """
        Returns
        -------
        CNMF
            Returns the Caiman CNMF object
        """
        # Need to create a cache object that takes the item's UUID and returns based on that
        # collective global cache
        return load_CNMF(self.get_output_path())

    # TODO: Make the ``ixs`` parameter for spatial stuff optional
    @validate('cnmf')
    def get_spatial_masks(self, ixs_components: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Get binary masks of the spatial components at the given `ixs`

        Basically created from cnmf.estimates.A

        Parameters
        ----------
        ixs_components: np.ndarray
            numpy array containing integer indices for which you want spatial masks

        threshold: float
            threshold

        Returns
        -------
        np.ndarray
            shape is [dim_0, dim_1, n_components]

        """
        cnmf_obj = self.get_output()

        dims = cnmf_obj.dims
        if dims is None:
            dims = cnmf_obj.estimates.dims

        masks = np.zeros(shape=(dims[0], dims[1], len(ixs_components)), dtype=bool)

        for n, ix in enumerate(ixs_components):
            s = cnmf_obj.estimates.A[:, ix].toarray().reshape(cnmf_obj.dims)
            s[s >= threshold] = 1
            s[s < threshold] = 0

            masks[:, :, n] = s.astype(bool)

        return masks

    # TODO: Cache this globally so that a common upper cache limit is valid for ALL batch items
    @staticmethod
    @lru_cache(5)
    def _get_spatial_contour_coors(cnmf_obj: CNMF):
        dims = cnmf_obj.dims
        if dims is None:  # I think that one of these is `None` if loaded from an hdf5 file
            dims = cnmf_obj.estimates.dims

        # need to transpose these
        dims = dims[1], dims[0]

        contours = caiman_get_contours(
            cnmf_obj.estimates.A,
            dims,
            swap_dim=True
        )

        return contours

    @validate('cnmf')
    def get_spatial_contours(self, ixs_components: np.ndarray) -> List[dict]:
        """
        Get the contours for the spatial footprints

        Parameters
        ----------
        ixs_components: np.ndarray
            indices for which to return spatial contours

        Returns
        -------

        """
        cnmf_obj = self.get_output()
        contours = self._get_spatial_contour_coors(cnmf_obj)

        contours_selection = list()
        for i in range(len(contours)):
            if i in ixs_components:
                contours_selection.append(contours[i])

        return contours_selection

    @validate('cnmf')
    def get_spatial_contour_coors(self, ixs_components: np.ndarray) -> List[np.ndarray]:
        contours = self.get_spatial_contours(ixs_components)

        coordinates = []
        for contour in contours:
            coors = contour['coordinates']
            coordinates.append(coors[~np.isnan(coors).any(axis=1)])

        return coordinates

    @validate('cnmf')
    def get_temporal_components(self, ixs_components: np.ndarray = None, add_background: bool = True) -> np.ndarray:
        """
        Get the temporal components for this CNMF item

        Parameters
        ----------
        ixs_components: np.ndarray
            indices for which to return temporal components, ``cnmf.estimates.C``

        add_background: bool
            if ``True``, add the temporal background, basically ``cnmf.estimates.C + cnmf.estimates.f``

        Returns
        -------

        """
        cnmf_obj = self.get_output()

        if ixs_components is None:
            ixs_components = np.arange(0, cnmf_obj.estimates.C.shape[0])

        C = cnmf_obj.estimates.C[ixs_components]
        f = cnmf_obj.estimates.f

        if add_background:
            return C + f
        else:
            return C

    # TODO: Cache this globally so that a common upper cache limit is valid for ALL batch items
    @validate('cnmf')
    def get_reconstructed_movie(self, ixs_frames: Tuple[int, int] = None, add_background: bool = True) -> np.ndarray:
        """
        Return the reconstructed movie, (A * C) + (b * f)

        Parameters
        ----------
        ixs_frames: Tuple[int, int]
            (start_frame, stop_frame), return frames in this range including the ``start_frame``, upto and not
            including the ``stop_frame``

        add_background: bool
            if ``True``, add the spatial & temporal background, b * f

        Returns
        -------
        np.ndarray
            shape is [n_frames, x_pixels, y_pixels]
        """
        cnmf_obj = self.get_output()

        if ixs_frames is None:
            ixs_frames = (0, cnmf_obj.estimates.C.shape[1])

        dn = (cnmf_obj.estimates.A.dot(cnmf_obj.estimates.C[:, ixs_frames[0]:ixs_frames[1]]))

        if add_background:
            dn += (cnmf_obj.estimates.b.dot(cnmf_obj.estimates.f[:, ixs_frames[0]:ixs_frames[1]]))
        return dn.reshape(cnmf_obj.dims + (-1,), order='F').transpose([2, 0, 1])


@pd.api.extensions.register_series_accessor("mcorr")
class MCorrExtensions:
    """
    Extensions for managing motion correction outputs
    """
    def __init__(self, s: pd.Series):
        self._series = s

    @validate('mcorr')
    def get_output_path(self) -> Path:
        """
        Get the path to the motion corrected output memmap file

        Returns
        -------
        Path
            path to the motion correction output memmap file
        """
        return get_full_data_path(self._series['outputs']['mcorr-output-path'])

    @validate('mcorr')
    def get_output(self) -> np.ndarray:
        """
        Get the motion corrected output as a memmaped numpy array, allows fast random-access scrolling.

        Returns
        -------
        np.ndarray
            memmap numpy array of the motion corrected movie
        """
        path = self.get_output_path()
        Yr, dims, T = load_memmap(str(path))
        mc_movie = np.reshape(Yr.T, [T] + list(dims), order='F')
        return mc_movie
