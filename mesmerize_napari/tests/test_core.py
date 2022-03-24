import os
from glob import glob
import pandas as pd
from ..core import create_batch, load_batch, CaimanDataFrameExtensions, CaimanSeriesExtensions,\
    set_parent_data_path, get_parent_data_path, get_full_data_path
from ..core import ALGO_MODULES, DATAFRAME_COLUMNS, SUBPROCESS_BACKEND
from uuid import uuid4
from typing import *
import pytest
import requests
from tqdm import tqdm
from .params import test_params
from uuid import UUID
from pathlib import Path
import shutil
from caiman.paths import caiman_datadir


tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
vid_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'videos')

os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(vid_dir, exist_ok=True)

def get_tmp_filename():
    return os.path.join(tmp_dir, f'{uuid4()}.pickle')


def clear_tmp():
    shutil.rmtree(tmp_dir)

    test = os.listdir(vid_dir)
    for item in test:
        if item.endswith(".npy") | item.endswith(".mmap") | item.endswith(".runfile"):
            os.remove(os.path.join(vid_dir, item))



def get_datafile(fname: str):
    local_path = Path(os.path.join(vid_dir, f'{fname}.tif'))
    if local_path.is_file():
        return local_path
    else:
        download_data(fname)


def download_data(fname: str):
    """
    Download the large network files from Zenodo
    """
    url = {
        'mcorr': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/demoMovie.tif',
        'cnmf': None,
        'cnmfe': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/data_endoscope.tif',
    }.get(fname)

    print(f'Downloading test data from: {url}')

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    path = os.path.join(vid_dir, f'{fname}.tif')
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


def test_mcorr():
    algo = 'mcorr'
    df, batch_path = _create_tmp_batch()
    print(f"Testing mcorr")
    input_movie_path = get_datafile(algo)
    print(input_movie_path)
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

    assert df.iloc[-1]['input_movie_path'] == os.path.join(vid_dir, f'{algo}.tif')

    # set_parent_data_path(vid_dir)
    # df.iloc[-1].caiman._run_subprocess()
    process = df.iloc[-1].caiman.run(
        backend=SUBPROCESS_BACKEND,
        callbacks_finished=None)
    process.wait()

    df = load_batch(batch_path)
    print(df)
    print(df.iloc[-1]['outputs']['traceback'])
    assert df.iloc[-1]['outputs']['success'] is True
    assert df.iloc[-1]['outputs']['traceback'] is None
    assert os.path.join(vid_dir, df.iloc[-1]['outputs']['mcorr-output-path']
                        ) == \
        os.path.join(vid_dir,
        f'{df.iloc[-1]["uuid"]}-mcorr_els__d1_60_d2_80_d3_1_order_F_frames_2000_.mmap')

    assert Path(os.path.join(vid_dir, df.iloc[-1]['outputs']['mcorr-output-path']
                        )) == \
        get_full_data_path(df.iloc[-1]['outputs']['mcorr-output-path']
                           )== \
        Path(os.path.join(vid_dir,
        f'{df.iloc[-1]["uuid"]}-mcorr_els__d1_60_d2_80_d3_1_order_F_frames_2000_.mmap'))

def test_cnmf():
    algo = 'mcorr'
    df, batch_path = _create_tmp_batch()
    print(f"Testing mcorr")
    input_movie_path = get_datafile(algo)
    print(input_movie_path)
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

    assert df.iloc[-1]['input_movie_path'] == os.path.join(vid_dir, f'{algo}.tif')

    process = df.iloc[-1].caiman.run(
        backend=SUBPROCESS_BACKEND,
        callbacks_finished=None)
    process.wait()

    df = load_batch(batch_path)
    print(df)
    print(df.iloc[-1]['outputs']['traceback'])
    assert df.iloc[-1]['outputs']['success'] is True
    assert df.iloc[-1]['outputs']['traceback'] is None
    assert os.path.join(vid_dir, df.iloc[-1]['outputs']['mcorr-output-path']
                        ) == \
        os.path.join(vid_dir,
        f'{df.iloc[-1]["uuid"]}-mcorr_els__d1_60_d2_80_d3_1_order_F_frames_2000_.mmap')

    assert Path(os.path.join(vid_dir, df.iloc[-1]['outputs']['mcorr-output-path']
                        )) == \
        get_full_data_path(df.iloc[-1]['outputs']['mcorr-output-path']
                           )== \
        Path(os.path.join(vid_dir,
        f'{df.iloc[-1]["uuid"]}-mcorr_els__d1_60_d2_80_d3_1_order_F_frames_2000_.mmap'))

    algo = 'cnmf'
    print("Testing cnmf")
    input_movie_path = df.iloc[-1]['outputs']['mcorr-output-path']
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
    print('df input path:', df.iloc[-1]['input_movie_path'])
    assert Path(df.iloc[-1]['input_movie_path']) == input_movie_path

    process = df.iloc[-1].caiman.run(
        backend=SUBPROCESS_BACKEND,
        callbacks_finished=None)
    process.wait()

    df = load_batch(batch_path)
    print(df)
    # Confirm output path is as expected

#
# def test_remove_item():
#     pass