"""Performs CNMF in a separate process"""

import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.utils.utils import load_dict_from_hdf5
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.motion_correction import MotionCorrect
from caiman.utils.utils import download_demo
import psutil
import json
from tqdm import tqdm
import sys
import pandas as pd


def main(batch_path, uuid):
    df = pd.read_pickle(batch_path)
    item = df[df['uuid'] == uuid].squeeze()

    input_movie_path = item['input_movie_path']
    params = item['params']

    # adapted from current demo notebook
    n_processes = psutil.cpu_count() - 1
    print("starting mc")
    # Start cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local',
        n_processes=n_processes,
        single_thread=False
    )

    cnmf_params = CNMFParams(params_dict=params)

    fname_new = cm.save_memmap(
        [input_movie_path],
        base_name='memmap_',
        order='C',
        dview=dview
    )

    print('making memmap')

    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')

    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local',
        n_processes=None,
        single_thread=False
    )

    print("performing CNMF")
    cnm = cnmf.CNMF(
        n_processes,
        params=cnmf_params,
        dview=dview
    )

    print("fitting images")
    cnm = cnm.fit(images)

    print('refitting')
    cnmf_obj = cnm.refit(images, dview=dview)

    print("Eval")
    cnmf_obj.estimates.evaluate_components(images, cnmf_obj.params, dview=dview)

    cnmf_obj.save(uuid + '.hdf5')


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
