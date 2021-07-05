"""Performs CNMF in a separate process"""

import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.utils.utils import load_dict_from_hdf5
from caiman.source_extraction.cnmf.params import CNMFParams
import psutil
import json
from tqdm import tqdm
import sys


params = \
{
    "fr": 30,
    "decay_time": 0.4,
    "p": 1,
    "nb": 2,
    "rf": 15,
    "K": 4,
    "stride": 6,
    "method_init": "greedy_roi",
    "rolling_sum": True,
    "only_init": True,
    "ssub": 1,
    "tsub": 1,
    "merge_thr": 0.85,
    "min_SNR": 2.0,
    "rval_thr": 0.85,
    "use_cnn": True,
    "min_cnn_thr": 0.99,
    "cnn_lowest": 0.1
}

cnmf_params = CNMFParams(params_dict=params)


def main(path):
    # adapted from current demo notebook
    n_processes = psutil.cpu_count() - 1
    print("starting cnmf")

    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local',
        n_processes=n_processes,
        single_thread=False
    )

    fname_new = cm.save_memmap(
        [path],
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

    cnmf_obj.save(path + '.results.hdf5')


if __name__ == "__main__":
    main(sys.argv[1])