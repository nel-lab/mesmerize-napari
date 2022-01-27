import pathlib
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
import pickle
from caiman.utils.visualization import get_contours as caiman_get_contours
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman_napari.utils import *
from caiman.summary_images import local_correlations_movie_offline
import os
import traceback
from napari.viewer import Viewer
import time
from time import sleep

def main(batch_path, uuid):
    df = pd.read_pickle(batch_path)
    item = df[df['uuid'] == uuid].squeeze()

    input_movie_path = item['input_movie_path']
    params = item['params']
    print("cnmfe params:", params)

    #adapted from current demo notebook
    n_processes = psutil.cpu_count() - 1
    # Start cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local',
        n_processes=n_processes,
        single_thread=False
     )

    try:
        fname_new = cm.save_memmap(
            [input_movie_path],
            base_name='memmap_',
            order='C',
            dview=dview
        )

        print('making memmap')
        gSig = params['cnmfe_kwargs']['gSig'][0]

        Yr, dims, T = cm.load_memmap(fname_new)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')

        # in fname new load in memmap order C

        cn_filter, pnr = cm.summary_images.correlation_pnr(
            images, swap_dim=False, gSig=gSig
        )

        if not params['do_cnmfe']:
             pickle.dump(cn_filter, open(uuid + '_cn_filter.pikl', 'wb'), protocol=4)
             pickle.dump(pnr, open(uuid + '_pnr.pikl', 'wb'), protocol=4)

             output_file_list = \
             [
                 uuid + '_pnr.pikl',
                 uuid + '_cn_filter.pikl'
             ]
             print(output_file_list)

             d = dict()
             d.update(
                 {
                     "cnmfe_outputs": output_file_list,
                     "cnmfe_memmap": fname_new,
                     "success": True,
                     "traceback": None
                 }
             )

        else:
            cnmfe_params_dict = \
                {
                    #"method_init": 'corr_pnr',
                    "n_processes": n_processes,
                    "only_init": True,    # for 1p
                    #"center_psf": True,         # for 1p
                    "normalize_init": False     # for 1p
                }
            # TODO: figure out cause for _flapack module error when including above dict params - method_init and center_psf
            tot = {**cnmfe_params_dict, **params['cnmfe_kwargs']}
            cnmfe_params_dict = CNMFParams(params_dict=tot)
            cnm = cnmf.CNMF(
                n_processes=n_processes,
                dview=dview,
                params=cnmfe_params_dict
            )
            print("Performing CNMFE")
            cnm = cnm.fit(images)
            print("evaluating components")
            #cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
            # TODO: figure out cause for 'error while saving cnn_preds' when running above function

            output_path = str(pathlib.Path(batch_path).parent.joinpath(f"{uuid}.hdf5").resolve())
            cnm.save(output_path)

            d = dict()
            d.update(
                {
                    "cnmfe_outputs": output_path,
                    "cnmfe_memmap": fname_new,
                    "success": True,
                    "traceback": None
                }
            )

    except:
        d = {"success": False, "traceback": traceback.format_exc()}
    # Add dictionary to output column of series
    df.loc[df['uuid'] == uuid, 'outputs'] = [d]
    # save dataframe to disc
    df.to_pickle(batch_path)
def load_output(viewer, batch_item: pd.Series):
    pass

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])