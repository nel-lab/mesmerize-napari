import os
from glob import glob
import pandas as pd
from ..core import create_batch, load_batch, CaimanDataFrameExtensions, CaimanSeriesExtensions
from ..core import ALGO_MODULES, DATAFRAME_COLUMNS
from uuid import uuid4
from typing import *


tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(tmp_dir, exist_ok=True)


def get_tmp_filename():
    return os.path.join(tmp_dir, f'{uuid4()}.pickle')


def clear_tmp():
    files = glob(os.path.join(tmp_dir, '*'))
    for f in files:
        os.remove(f)


def teardown_module():
    clear_tmp()
    os.removedirs(tmp_dir)


def _create_batch() -> Tuple[pd.DataFrame, str]:
    fname = get_tmp_filename()
    df = create_batch(fname)

    return df, fname


def test_create_batch():
    df = _create_batch()[0]

    for c in DATAFRAME_COLUMNS:
        assert c in df.columns


def test_load_batch():
    pass


def test_add_item_all_algos():
    pass


def test_remove_item():
    pass


def test_run_all_algos():
    pass

