import os
from glob import glob
import pandas as pd
from ..core import create_batch, load_batch, CaimanDataFrameExtensions, CaimanSeriesExtensions,\
    set_parent_data_path, get_parent_data_path
from ..core import ALGO_MODULES, DATAFRAME_COLUMNS
from uuid import uuid4
from typing import *
import pytest
import requests
from tqdm import tqdm
from .params import test_params
from uuid import UUID
from pathlib import Path
import shutil


tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)


def get_tmp_filename():
    return os.path.join(tmp_dir, f'{uuid4()}.pickle')


def clear_tmp():
    shutil.rmtree(tmp_dir)
    shutil.rmtree(data_dir)


def download_data(algo: str):
    """
    Download the large network files from Zenodo
    """
    url = {
        'mcorr': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/demoMovie.tif',
        'cnmf': None,
        'cnmfe': None,
    }.get(algo)

    print(f'Downloading test data from: {url}')

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    os.mkdir(os.path.join(data_dir, 'input_movies'))

    path = os.path.join(data_dir, 'input_movies', f'{algo}.tif')

    with open(path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise ConnectionError("Couldn't download test test")

    return Path(path)


def teardown_module():
    clear_tmp()


def _create_tmp_batch() -> Tuple[pd.DataFrame, str]:
    fname = get_tmp_filename()
    df = create_batch(fname)

    return df, fname


def test_create_batch():
    df, fname = _create_tmp_batch()

    for c in DATAFRAME_COLUMNS:
        assert c in df.columns

    # test that existing batch is not overwritten
    with pytest.raises(FileExistsError):
        create_batch(fname)


def test_load_batch():
    pass


def test_all_algos():
    set_parent_data_path(data_dir)

    ALGO_MODULES = {'mcorr': None}  # temporary until other algo tests are ready

    df, batch_path = _create_tmp_batch()
    for algo in list(ALGO_MODULES.keys()):
        print(f"Testing {algo}")
        input_movie_path = download_data(algo)

        df.caiman.add_item(
            algo=algo,
            name=f'test-{algo}',
            input_movie_path=input_movie_path,
            params=test_params[algo]
        )

        assert df.iloc[-1]['algo'] == algo
        assert df.iloc[-1]['name'] == f'test-{algo}'
        assert df.iloc[-1]['params'] == test_params[algo]
        assert df.iloc[-1]['outputs'] is None
        try:
            UUID(df.iloc[-1]['uuid'])
        except:
            pytest.fail("Something wrong with setting UUID for batch items")

        parent_data_path = get_parent_data_path()

        assert df.iloc[-1]['input_movie_path'] == f'input_movies/{algo}.tif'
        assert parent_data_path.joinpath(df.iloc[-1]['input_movie_path']) == input_movie_path

        df.iloc[-1].caiman._run_subprocess()

        df = load_batch(batch_path)

        assert parent_data_path.joinpath(
            df.iloc[-1]['outputs']['mcorr_output']
        ) == \
            parent_data_path.joinpath(
                'input_movies',
                f'{df.iloc[-1]["uuid"]}-mcorr_els__d1_60_d2_80_d3_1_order_F_frames_2000_.mmap'
            )

        assert df.iloc[-1]['outputs']['success'] is True
        assert df.iloc[-1]['outputs']['traceback'] is None


def test_remove_item():
    pass

